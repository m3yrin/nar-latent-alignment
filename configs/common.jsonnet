# if you don't use cuda, cuda_device=-1
local cuda_device = 0;

# https://arxiv.org/pdf/2004.07437.pdf
# Our models consists of 12 self-attention
# layers, with 512 hidden size, 2048 filter size, and
# 8 attention heads per layer. We use 0.1 dropout
# for regularization. 

{
    "direction" : "ja-en",
    "train_data_path": "datasets/small_parallel_enja/train",
    "validation_data_path": "datasets/small_parallel_enja/test",

    "embedding_dim" : 128,
    "feedforward_hidden_dim" : 2048,
    "num_layers" : 12,

    "batch_size" : 128,
    "min_count" : 2,

    "num_epochs" : 150,
    "patience": 10,
    "cuda_device" : cuda_device,
}