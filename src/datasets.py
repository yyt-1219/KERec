import random
import torch
from torch.utils.data import Dataset
from utils import neg_sample
import copy


class SASRecDataset(Dataset):
    def __init__(self, args, user_seq, test_neg_items=None, data_type="train"):
        self.args = args
        self.user_seq = user_seq
        self.test_neg_items = test_neg_items
        self.data_type = data_type
        self.max_len = args.max_seq_length

    def mask_input_ids(self, input_ids, p):
        mask_indices = []
        for token in input_ids:
            if token == 0:
                mask_indices.append(0)
            else:
                mask_indices.append(1 if random.random() < p else 0)
        return mask_indices

    def _data_sample_rec_task(self, user_id, items, input_ids, target_pos, answer):
        copied_input_ids = copy.deepcopy(input_ids)
        target_neg = []
        seq_set = set(items)
        for _ in input_ids:
            target_neg.append(neg_sample(seq_set, self.args.item_size))

        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        target_pos = [0] * pad_len + target_pos
        target_neg = [0] * pad_len + target_neg

        input_ids = input_ids[-self.max_len:]
        target_pos = target_pos[-self.max_len:]
        target_neg = target_neg[-self.max_len:]

        assert len(input_ids) == self.max_len
        assert len(target_pos) == self.max_len
        assert len(target_neg) == self.max_len

        attention_mask = [1.0 if token > 0 else 0.0 for token in input_ids]

        if self.test_neg_items is not None:
            test_samples = self.test_neg_items[index]
            cur_rec_tensors = (
                torch.tensor(user_id, dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(test_samples, dtype=torch.long),
            )
        else:
            cur_rec_tensors = (
                torch.tensor(user_id, dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
            )

        return cur_rec_tensors

    def _add_noise_interactions(self, items):
        copied_sequence = copy.deepcopy(items)
        insert_nums = max(int(self.args.noise_ratio * len(copied_sequence)), 0)
        if insert_nums == 0:
            return copied_sequence
        insert_idx = random.choices([i for i in range(len(copied_sequence))], k=insert_nums)
        inserted_sequence = []
        for index, item in enumerate(copied_sequence):
            if index in insert_idx:
                item_id = random.randint(1, self.args.item_size - 2)
                while item_id in copied_sequence:
                    item_id = random.randint(1, self.args.item_size - 2)
                inserted_sequence += [item_id]
            inserted_sequence += [item]
        return inserted_sequence

    def __getitem__(self, index):
        user_id = index
        items = self.user_seq[index]

        assert self.data_type in {"train", "valid", "test"}

        if self.data_type == "train":
            input_ids = items[:-3]
            target_pos = items[1:-2]
            answer = [0]  # no use

        elif self.data_type == "valid":
            input_ids = items[:-2]
            target_pos = items[1:-1]
            answer = [items[-2]]

        else:
            items_with_noise = self._add_noise_interactions(items)
            input_ids = items_with_noise[:-1]
            target_pos = items_with_noise[1:]
            answer = [items_with_noise[-1]]

        if self.data_type == "train" or self.data_type == "valid":
            return self._data_sample_rec_task(user_id, items, input_ids, target_pos, answer)
        else:
            return self._data_sample_rec_task(user_id, items_with_noise, input_ids, target_pos, answer)

    def __len__(self):
        return len(self.user_seq)


def DS(i_file, o_file, max_len):
    """
    :param i_file: original data
    :param o_file: output data
    :max_len: the max length of the sequence
    :return:
    """
    with open(i_file, "r+") as fr:
        data = fr.readlines()
    aug_d = {}
    # training, validation, and testing
    max_save_len = max_len + 3
    # save
    max_keep_len = max_len + 2
    for d_ in data:
        u_i, item = d_.split(' ', 1)
        item = item.split(' ')
        item[-1] = str(eval(item[-1]))
        aug_d.setdefault(u_i, [])
        start = 0
        j = 3
        if len(item) > max_save_len:
            # training, validation, and testing
            while start < len(item) - max_keep_len:
                j = start + 4
                while j < len(item):
                    if start < 1 and j - start < max_save_len:
                        aug_d[u_i].append(item[start:j])
                        j += 1
                    else:
                        aug_d[u_i].append(item[start:start + max_save_len])
                        break
                start += 1
        else:
            while j < len(item):
                aug_d[u_i].append(item[start:j + 1])
                j += 1
    with open(o_file, "w+") as fw:
        for u_i in aug_d:
            for i_ in aug_d[u_i]:
                fw.write(u_i + " " + ' '.join(i_) + "\n")
