local COMMON = import 'common.jsonnet';

local direction = COMMON['direction'];
local train_data_path = COMMON['train_data_path'];
local validation_data_path = COMMON['validation_data_path'];

local embedding_dim = COMMON['embedding_dim'];
local feedforward_hidden_dim = COMMON['feedforward_hidden_dim'];
local num_layers = COMMON['num_layers'];
local num_epochs = COMMON['num_epochs'];
local batch_size = COMMON['batch_size'];
local patience = COMMON['patience'];
local min_count = COMMON['min_count'];
local cuda_device = COMMON['cuda_device'];


## for AR
# encoder net is bidirectional, so target input become double.
local target_embedding_dim = embedding_dim * 2;
# small_parallel_enja filtering sentences length to be 4 to 16 words.
local max_decoding_steps = 20;
local decoder_num_attention_heads = 8;
local learning_rate_ar = 0.001;

{
    "dataset_reader": {
      "type": "small_parallel_enja_reader",
      "direction" : direction,
      "add_start_end_tokens" : true,
    },
    "train_data_path": train_data_path,
    "validation_data_path": validation_data_path,
    "model": {
        "type": "composed_seq2seq",
        "source_text_embedder": {
            "type": "basic",
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": embedding_dim
                }
            }
        },
        "encoder": {
            "type": "bidirectional_language_model_transformer",
            "input_dim": embedding_dim,
            "hidden_dim": feedforward_hidden_dim,
            "num_layers": num_layers
        },
        "decoder": {
            "type": "auto_regressive_seq_decoder",
            "target_namespace": "target_tokens",
            "decoder_net": {
                "type": "stacked_self_attention",
                "decoding_dim": target_embedding_dim,
                "target_embedding_dim": target_embedding_dim,
                "feedforward_hidden_dim": feedforward_hidden_dim,
                "num_layers": num_layers,
                "num_attention_heads": decoder_num_attention_heads
            },
            "max_decoding_steps": max_decoding_steps,
            "target_embedder": {
                "vocab_namespace": "target_tokens",
                "embedding_dim": target_embedding_dim
            },
            "tensor_based_metric": {
                "type": "bleu",
                # "exclude_indices": [0, 2, 3], 
                #   as is in https://github.com/allenai/allennlp/blob/052353ed62e3a54fd7b39a660e65fc5dd2f91c7d/allennlp/models/encoder_decoders/simple_seq2seq.py#L95
                #   assuming {0: '@@PADDING@@',1: '@@UNKNOWN@@',2: '@start@',3: '@end@', ...}
                #   I don't know how to pass these indices from config like "vocab.get_token_index(self.vocab._padding_token, self._target_namespace)"
                "exclude_indices": [0, 2, 3], 
            }
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
            "lr": learning_rate_ar
        },
        "patience": patience,
        "validation_metric": "+BLEU",
        "num_epochs": num_epochs,
        "cuda_device": cuda_device,
        "learning_rate_scheduler": {
            "type": "exponential",
            "gamma": 0.98
        }
    },
    "vocabulary": {
        "min_count": {
            "source_tokens": min_count,
            "target_tokens": min_count
        }
    }
}