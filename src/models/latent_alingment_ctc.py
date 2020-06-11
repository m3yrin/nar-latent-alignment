from typing import Dict, List, Tuple

import numpy
from overrides import overrides
import torch
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTMCell

from allennlp.common.checks import ConfigurationError
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.attention import LegacyAttention
from allennlp.modules import Attention, TextFieldEmbedder, Seq2SeqEncoder
from allennlp.modules.seq2seq_encoders import StackedSelfAttentionEncoder
from allennlp.modules.similarity_functions import SimilarityFunction
from allennlp.models.model import Model
from allennlp.modules.token_embedders import Embedding
from allennlp.nn import util
from allennlp.nn.beam_search import BeamSearch
from allennlp.training.metrics import BLEU

from ..modules.upsample import LinearUpsample
from ..utils.loss import sequence_ctc_loss_with_logits

SPECIAL_BLANK_TOKEN = "@@BLANK@@"


@Model.register("latent_alignment_ctc")
class LatentAignmentCTC(Model):

    def __init__(
        self,
        vocab: Vocabulary,
        source_embedder: TextFieldEmbedder,
        upsample: torch.nn.Module = None, 
        net: Seq2SeqEncoder = None,
        target_namespace: str = "target_tokens",
        target_embedding_dim: int = None,
        use_bleu: bool = True,
    ) -> None:
        super(LatentAignmentCTC, self).__init__(vocab)
        self._target_namespace = target_namespace
        
        self._pad_index = self.vocab.get_token_index(self.vocab._padding_token, 
                                                    self._target_namespace)
        self._blank_index = self.vocab.get_token_index(SPECIAL_BLANK_TOKEN, 
                                                       self._target_namespace)

        if use_bleu:
            self._bleu = BLEU(exclude_indices={self._pad_index, self._blank_index})
        else:
            self._bleu = None

        self._source_embedder = source_embedder
        source_embedding_dim = source_embedder.get_output_dim()

        self._upsample = upsample or LinearUpsample(source_embedding_dim, s = 3)
        self._net = net or StackedSelfAttentionEncoder(input_dim = source_embedding_dim,
                                                       hidden_dim = 128,
                                                       projection_dim = 128,
                                                       feedforward_hidden_dim = 512,
                                                       num_layers = 4,
                                                       num_attention_heads = 4)
        
        num_classes = self.vocab.get_vocab_size(self._target_namespace)
        target_embedding_dim = self._net.get_output_dim()

        self._output_projection = torch.nn.Linear(target_embedding_dim, num_classes)
        
        

    @overrides
    def forward(
        self,  # type: ignore
        source_tokens:  Dict[str, torch.LongTensor],
        target_tokens:  Dict[str, torch.LongTensor] = None,
    ) -> Dict[str, torch.Tensor]:
    
        
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
                self._bleu(output_dict['predictions'], target_tokens["tokens"])
                
                #output_dict = self.decode(output_dict)
                #print(output_dict["predicted_tokens"])

        return output_dict

    # TODO: too cheap. need pallalel processing
    def beta_inverse(self, a:torch.Tensor):
        """
        a : size (batch, sequence)
        """
        max_length = a.size(1)
        outputs = []

        for sequence in a.tolist():
            output = []
            for token in sequence:
                if token == self._blank_index:
                    continue
                elif len(output) == 0:
                    output.append(token)
                    continue
                elif token == output[-1]:
                    continue
                else:
                    output.append(token)
            
            pad_list = [self._pad_index] * (max_length - len(output))
            outputs.append(output + pad_list)
            
        return torch.LongTensor(outputs)

    # @staticmethod
    def _get_loss(self, 
        output_dict: Dict[str, torch.Tensor], target_tokens: Dict[str, torch.LongTensor]) -> torch.Tensor:

        targets = target_tokens["tokens"]
        target_mask = util.get_text_field_mask(target_tokens)
        
        # shape: (batch_size, input_length, target_size)
        alignment_logits = output_dict["alignment_logits"]

        # shape: (batch_size, input_length)
        source_mask_upsampled = output_dict["source_mask_upsampled"]

        #return util.sequence_cross_entropy_with_logits(alignment_logits, targets, source_mask_upsampled)
        return sequence_ctc_loss_with_logits(alignment_logits, source_mask_upsampled, targets, target_mask, self._blank_index)


    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
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
        
        # provide "tokens" and "predicted_tokens" for output.
        output_dict["predicted_tokens"] = all_predicted_tokens
        del output_dict["alignment_logits"], output_dict['source_mask_upsampled'], output_dict['net_output']
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        if self._bleu and not self.training:
            all_metrics.update(self._bleu.get_metric(reset=reset))
        return all_metrics
