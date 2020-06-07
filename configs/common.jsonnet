# if you don't use cuda, cuda_device=-1
local cuda_device = -1;

# https://arxiv.org/pdf/2004.07437.pdf
# Our models consists of 12 self-attention
# layers, with 512 hidden size, 2048 filter size, and
# 8 attention heads per layer. We use 0.1 dropout
# for regularization. 

{
    "direction" : "ja-en",
    "train_data_path": "datasets/small_parallel_enja/train",
    "validation_data_path": "datasets/small_parallel_enja/test",

    "embedding_dim" : 8,
    "feedforward_hidden_dim" : 8,
    "num_layers" : 2,

    "batch_size" : 8,
    "min_count" : 2,

    "num_epochs" : 10,
    "patience": 3,
    "cuda_device" : cuda_device,
}