import argparse


def get_config():
    parser = argparse.ArgumentParser()
    # system args
    parser.add_argument("--data_dir", default="../data/", type=str)
    parser.add_argument("--output_dir", default="output/", type=str)
    parser.add_argument("--data_name", default="Sports_and_Outdoors", type=str)
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")

    # robustness experiments
    parser.add_argument("--noise_ratio", default=0.0, type=float,
                        help="percentage of negative interactions ln-in a sequence - robustness analysis", )

    ## contrastive learning task args
    parser.add_argument("--temperature", default=1.0, type=float,
                        help="softmax temperature (default:  1.0) - not studied.")
    parser.add_argument("--sim", default='dot', type=str, help="the calculate ways of the similarity.")

    # model args
    parser.add_argument("--model_name", default="KERec", type=str)
    parser.add_argument("--hidden_size", type=int, default=64, help="hidden size of transformer model")
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="number of layers")
    parser.add_argument("--num_attention_heads", default=2, type=int)
    parser.add_argument("--hidden_act", default="gelu", type=str)  # gelu relu
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5, help="attention dropout p")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.5, help="hidden dropout p")
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument("--max_seq_length", default=50, type=int)

    # train args
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument("--batch_size", type=int, default=256, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=2025, type=int)
    # loss weight
    parser.add_argument("--rec_weight", type=float, default=1, help="weight of prediction task")
    parser.add_argument("--alpha", type=float, default=0.1, help="weight of high-pass filter")
    parser.add_argument("--cl_weight", type=float, default=0.1, help="weight of contrastive learning task")
    parser.add_argument("--noise_scale", type=float, default=0.1,
                        help="scale of noise added to the sequence embeddings")
    parser.add_argument("--scales", type=str, default='1,2,4')
    parser.add_argument("--num_samples", type=int, default=5, help="number of dropout samples")

    # ablation experiments
    parser.add_argument("--f_neg", action="store_true", help="delete the FNM component (both ln-in cicl and ficl)")

    # learning related
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")

    return parser.parse_args()
