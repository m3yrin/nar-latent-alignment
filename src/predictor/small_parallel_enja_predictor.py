from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor

from allennlp.data.tokenizers import Token


@Predictor.register('small_parallel_enja_predictor')
class SmallParallelEnJaPredictor(Predictor):
    def predict(self, source: str) -> JsonDict:
        return self.predict_json({"source" : source})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"source": "..."}``.
        """
        source = json_dict["source"]
        return self._dataset_reader.text_to_instance([Token(w) for w in source])