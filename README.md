# nar-latent-alignment [WIP]
Unofficial implementation of "Non-Autoregressive Machine Translation with Latent Alignments" https://arxiv.org/abs/2004.07437


* Built on AllenNLP (0.9.0)
* Dataset (En-Ja)
    * A small parallel corpus for English-Japanese translation task provided by @odashi
    * see detail at https://github.com/odashi/small_parallel_enja

## Memo
* CTC model only
   * Imputer model is under construction.
* The CTC model is inspired heavily by Libovicky and Helcl (2018)
   * Jindrich Libovicky and Jindrich Helcl. 2018. End-toEnd Non-Autoregressive Neural Machine Translation with Connectionist Temporal Classification. In EMNLP.
   * https://arxiv.org/abs/1811.04719
* Distillation is not tested.

## TODO
* Imputer model

## Examples (CTC)
0. Download datasets
```
$ cd datasets
$ git clone https://github.com/odashi/small_parallel_enja.git
```

1. training
```
$ allennlp train -f --include-package src -s tmp configs/ctc.jsonnet
```

2. evaluation
```
$ allennlp evaluate --output-file tmp/output_test.json --include-package src tmp/model.tar.gz datasets/small_parallel_enja/test
```

3. prediction
    1. make json inputs
        ```
        $ python datasets/make_json.py -I datasets/small_parallel_enja/test.ja -O datasets/test.ja.json
        ```
    2. predict
        ```
        $ allennlp predict --output-file tmp/output_pred.json --include-package src --predictor small_parallel_enja_predictor  tmp/model.tar.gz datasets/test.ja.json
        ```
