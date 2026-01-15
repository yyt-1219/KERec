import torch
import torch.nn as nn
from modules import DualEncoder, LayerNorm, CosinePredictionHead


class SASRecModel(nn.Module):
    def __init__(self, args):
        super(SASRecModel, self).__init__()
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.item_encoder = DualEncoder(args)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args = args
        self.output_head = CosinePredictionHead(args)

        self.apply(self.init_weights)

    # Positional Embedding
    def add_position_embedding(self, sequence):
        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_embeddings(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)
        return sequence_emb

    # model same as SASRec
    # def forward(self, input_ids):
    def forward(self, input_ids):
        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        if self.args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        sequence_emb = self.add_position_embedding(input_ids)

        item_encoded_layers = self.item_encoder(sequence_emb, extended_attention_mask, output_all_encoded_layers=True)
        sequence_output = item_encoded_layers[-1]

        with torch.no_grad():
            dynamics = self.item_encoder.layer[0].taylor_extractor(sequence_emb)
        aug_sequence_emb = self.kinematic_noise_augmentation(sequence_emb, dynamics, noise_scale=self.args.noise_scale)

        aug_item_encoded_layers = self.item_encoder(aug_sequence_emb, extended_attention_mask,
                                                    output_all_encoded_layers=True)
        aug_sequence_output = aug_item_encoded_layers[-1]

        logits = self.output_head(sequence_output[:, -1, :], self.item_embeddings.weight)

        return logits, sequence_output, aug_sequence_output

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def kinematic_noise_augmentation(self, sequence_emb, dynamics, attention_mask=None, noise_scale=0.2, drift_clip=2.0,
                                     rand_mix=0.3, eps=1e-8):
        drift = torch.norm(dynamics, dim=-1, keepdim=True)  # [B, L, 1]

        if attention_mask is not None:
            m = attention_mask.unsqueeze(-1).float()
            drift_mean = (drift * m).sum(dim=1, keepdim=True) / m.sum(dim=1, keepdim=True).clamp_min(1.0)
        else:
            drift_mean = drift.mean(dim=1, keepdim=True)

        drift_norm = (drift / (drift_mean + eps)).clamp(0.0, drift_clip)  # [B, L, 1]

        dyn_dir = dynamics / torch.norm(dynamics, dim=-1, keepdim=True).clamp_min(eps)

        rand = torch.randn_like(dyn_dir)
        rand_dir = rand / torch.norm(rand, dim=-1, keepdim=True).clamp_min(eps)

        direction = (1.0 - rand_mix) * dyn_dir + rand_mix * rand_dir
        direction = direction / torch.norm(direction, dim=-1, keepdim=True).clamp_min(eps)

        drop_mask = (torch.rand_like(drift_norm) < noise_scale).float()  # [B, L, 1]

        noise = drift_norm * drop_mask * direction  # [B, L, H]

        if attention_mask is not None:
            noise = noise * attention_mask.unsqueeze(-1).float()

        return sequence_emb + noise
