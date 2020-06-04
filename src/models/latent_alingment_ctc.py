from typing import Dict, List, Tuple, Iterable

import numpy
from overrides import overrides
import torch
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTMCell

from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.modules import Attention, TextFieldEmbedder, Seq2SeqEncoder
from allennlp.models.model import Model

from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import PytorchTransformer

from allennlp.nn import util
from allennlp.training.metrics import BLEU

from ..modules import LinearUpsample


@Model.register("latent_alignment_ctc")
class LatentAignmentCTC(Model):
    """
    This `SimpleSeq2Seq` class is a `Model` which takes a sequence, encodes it, and then
    uses the encoded representations to decode another sequence.  You can use this as the basis for
    a neural machine translation system, an abstractive summarization system, or any other common
    seq2seq problem.  The model here is simple, but should be a decent starting place for
    implementing recent models for these tasks.

    # Parameters

    vocab : `Vocabulary`, required
        Vocabulary containing source and target vocabularies. They may be under the same namespace
        (`tokens`) or the target tokens can have a different namespace, in which case it needs to
        be specified as `target_namespace`.
    source_embedder : `TextFieldEmbedder`, required
        Embedder for source side sequences
    encoder : `Seq2SeqEncoder`, required
        The encoder of the "encoder/decoder" model
    max_decoding_steps : `int`
        Maximum length of decoded sequences.
    target_namespace : `str`, optional (default = 'tokens')
        If the target side vocabulary is different from the source side's, you need to specify the
        target's namespace here. If not, we'll assume it is "tokens", which is also the default
        choice for the source side, and this might cause them to share vocabularies.
    target_embedding_dim : `int`, optional (default = source_embedding_dim)
        You can specify an embedding dimensionality for the target side. If not, we'll use the same
        value as the source embedder's.
    attention : `Attention`, optional (default = None)
        If you want to use attention to get a dynamic summary of the encoder outputs at each step
        of decoding, this is the function used to compute similarity between the decoder hidden
        state and encoder outputs.
    beam_size : `int`, optional (default = None)
        Width of the beam for beam search. If not specified, greedy decoding is used.
    scheduled_sampling_ratio : `float`, optional (default = 0.)
        At each timestep during training, we sample a random number between 0 and 1, and if it is
        not less than this value, we use the ground truth labels for the whole batch. Else, we use
        the predictions from the previous time step for the whole batch. If this value is 0.0
        (default), this corresponds to teacher forcing, and if it is 1.0, it corresponds to not
        using target side ground truth labels.  See the following paper for more information:
        [Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks. Bengio et al.,
        2015](https://arxiv.org/abs/1506.03099).
    use_bleu : `bool`, optional (default = True)
        If True, the BLEU metric will be calculated during validation.
    ngram_weights : `Iterable[float]`, optional (default = (0.25, 0.25, 0.25, 0.25))
        Weights to assign to scores for each ngram size.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        source_embedder: TextFieldEmbedder,
        upsample: nn.Module = None, 
        net: Seq2SeqEncoder = None,
        target_namespace: str = "target_tokens",
        target_embedding_dim: int = None,
        use_bleu: bool = True,
        bleu_ngram_weights: Iterable[float] = (0.25, 0.25, 0.25, 0.25),
    ) -> None:
        super().__init__(vocab)
        self._target_namespace = target_namespace
        
        self._pad_index = self.vocab.get_token_index(self.vocab._padding_token, 
                                                    self._target_namespace)

        if use_bleu:
            self._bleu = BLEU(bleu_ngram_weights, exclude_indices={self.pad_index})
        else:
            self._bleu = None

        self._source_embedder = source_embedder
        source_embedding_dim = _source_embedder.get_output_dim()

        self._upsample = upsample or LinearUpsample(source_embedding_dim, s = 3)
        self._net = net or PytorchTransformer(input_dim = source_embedding_dim * 3,
                                              num_layers = 12,
                                              feedforward_hidden_dim = 1024,
                                              num_attention_heads = 8)
        
        num_classes = self.vocab.get_vocab_size(self._target_namespace)
        target_embedding_dim = self._net.get_output_dim()

        self._output_projection = torch.nn.Linear(target_embedding_dim, num_classes)
        
        

    @overrides
    def forward(
        self,  # type: ignore
        source_tokens: TextFieldTensors,
        target_tokens: TextFieldTensors = None,
    ) -> Dict[str, torch.Tensor]:

        """
        Make foward pass with decoder logic for producing the entire target sequence.

        # Parameters

        source_tokens : `TextFieldTensors`
           The output of `TextField.as_array()` applied on the source `TextField`. This will be
           passed through a `TextFieldEmbedder` and then through an encoder.
        target_tokens : `TextFieldTensors`, optional (default = None)
           Output of `Textfield.as_array()` applied on target `TextField`. We assume that the
           target tokens are also represented as a `TextField`.

        # Returns

        Dict[str, torch.Tensor]
        """

        # shape: (batch_size, max_input_sequence_length, encoder_input_dim)
        embedded_input = self._source_embedder(source_tokens)
        # shape: (batch_size, max_input_sequence_length)
        source_mask = util.get_text_field_mask(source_tokens)

        # source_upsampled : shape : (batch_size, max_input_sequence_length, encoder_input_dim * self.s)
        # source_mask_upsampled : shape : (batch_size, max_input_sequence_length)
        source_upsampled, source_mask_upsampled = self._upsample(embedded_input, source_mask)


        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        net_output = self._net(source_upsampled, source_mask_upsampled)
        output_dict = {"source_mask_upsampled": source_mask_upsampled, "net_output": net_output}

        alignment_logits = self._output_projection(net_output)
        output_dict["alignment_logits"] = alignment_logits

        if target_tokens:
            # Compute loss.
            loss = self._get_loss(output_dict, target_tokens)
            output_dict["loss"] = loss

        if not self.training:
            alignments = alignment_logits.detach().cpu().argmax(2)
            predictions = self.beta_inverse(alignments)
            output_dict["predictions"] = predictions

            if target_tokens and self._bleu:
                self._bleu(output_dict['predictions'], target_tokens["tokens"]["tokens"])

        return output_dict

    # TODO: too cheap. need pallalel processing
    def beta_inverse(self, a:torch.Tensor):
        """
        a : size (batch, sequence)
        """
        max_length = a.size(1)
        outputs = []

        for sequence in a.to_list():
            output = []
            for token in sequence:
                if len(output) == 0:
                    output.append(token)
                    continue
                if token == output[-1]:
                    continue
                else:
                    output.append(token)
            
            pad_list = [self._pad_index] * (max_length - len(output))
            outputs.append(output + pad_list)
            
        return torch.LongTensor(outputs)

    @staticmethod
    def _get_loss(
        output_dict: Dict[str, torch.Tensor], target_tokens: TextFieldTensors) -> torch.Tensor:

        targets = target_tokens["tokens"]["tokens"]
        target_mask = util.get_text_field_mask(target_tokens)
        
        """
        # discarding <EOS>
        # shape: (batch_size, target_length)
        relevant_targets = targets[:, 1:].contiguous()
        # shape: (batch_size, target_length)
        relevant_mask = target_mask[:, 1:].contiguous()
        """

        # shape: (batch_size, input_length, target_size)
        alignment_logits = output_dict["alignment_logits"]

        # shape: (batch_size, input_length)
        source_mask_upsampled = output_dict["source_mask_upsampled"]

        return util.sequence_ctc_loss_with_logits(alignments, source_mask_upsampled, targets, target_mask)


    @overrides
    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Finalize predictions.
        """
        predicted_indices = output_dict["predictions"]
        if not isinstance(predicted_indices, numpy.ndarray):
            predicted_indices = predicted_indices.detach().cpu().numpy()
        
        all_predicted_tokens = []

        for indices in predicted_indices:
            indices = list(indices)

            # remove pad
            if self._pad_index in indices:
                indices = indices[: indices.index(self._pad_index)]

            # lookup
            predicted_tokens = [
                self.vocab.get_token_from_index(x, namespace=self._target_namespace)
                for x in indices
            ]
            all_predicted_tokens.append(predicted_tokens)
        
        output_dict["predicted_tokens"] = all_predicted_tokens

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        if self._bleu and not self.training:
            all_metrics.update(self._bleu.get_metric(reset=reset))
        return all_metrics
