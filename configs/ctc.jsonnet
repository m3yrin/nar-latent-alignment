# https://arxiv.org/pdf/2004.07437.pdf
# Our models consists of 12 self-attention
# layers, with 512 hidden size, 2048 filter size, and
# 8 attention heads per layer. We use 0.1 dropout
# for regularization. 

local embedding_dim = 256;

# Stacked self attention
local hidden_dim = 512;
local feedforward_hidden_dim = 2048;
local num_layers = 12;
local num_attention_heads = 8;


local num_epochs = 100;
local batch_size = 64;
local learning_rate = 0.001;

local SPECIAL_BLANK_TOKEN = "@@BLANK@@";
local min_count = 2;
local patience = 2;

# if you don't use cuda, cuda_device=-1
local cuda_device=0;


{
    "dataset_reader": {
        "type": "tanaka_corpus_reader"
    },
    "train_data_path": "datasets/small_parallel_enja/train",
    "validation_data_path": "datasets/small_parallel_enja/dev",
    "model": {
        "type": "latent_alignment_ctc",
        "source_embedder": {
            "type": "basic",
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": embedding_dim
                }
            }
        },
        "net": {
            "type": "stacked_self_attention",
            "input_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "projection_dim": hidden_dim,
            "feedforward_hidden_dim": feedforward_hidden_dim,
            "num_layers": num_layers,
            "num_attention_heads": num_attention_heads
        }
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys": [
            [
                "source_tokens",
                "num_tokens"
            ]
        ],
        "batch_size": batch_size
    },
    "trainer": {
        "optimizer": {
            "type": "adam",
            "lr": learning_rate
        },
        "patience": patience,
        "num_epochs": num_epochs,
        "cuda_device": cuda_device
    },
    "vocabulary": {
        "min_count": {
            "source_tokens": min_count,
            "target_tokens": min_count
        },
        "tokens_to_add": {
            "target_tokens": [
                SPECIAL_BLANK_TOKEN
            ]
        }
    }
}