
# common settings
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

# for CTC
local learning_rate_ctc = 0.001;
local SPECIAL_BLANK_TOKEN = "@@BLANK@@";

{
    "dataset_reader": {
      "type": "small_parallel_enja_reader",
      "direction" : direction,
      "add_start_end_tokens" : false, # the model doesn't use BOS/EOS for now.
    },
    "train_data_path": train_data_path,
    "validation_data_path": validation_data_path,
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
            "type": "bidirectional_language_model_transformer",
            "input_dim": embedding_dim,
            "hidden_dim": feedforward_hidden_dim,
            "num_layers": num_layers,
        }
        #"net": {
        #    "type": "stacked_self_attention",
        #    "input_dim": embedding_dim,
        #    "hidden_dim": hidden_dim,
        #    "projection_dim": hidden_dim,
        #    "feedforward_hidden_dim": feedforward_hidden_dim,
        #    "num_layers": num_layers,
        #    "num_attention_heads": num_attention_heads
        #}
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
            "lr": learning_rate_ctc
        },
        "patience": patience,
        "validation_metric": "+BLEU",
        "num_epochs": num_epochs,
        "cuda_device": cuda_device,
        "learning_rate_scheduler": {
            "type": "exponential",
            "gamma": 0.98
        },
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
