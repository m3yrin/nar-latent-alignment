"""
Assorted utilities for working with neural networks in AllenNLP.
"""
# pylint: disable=too-many-lines
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar, Union
import logging
import torch
from allennlp.common.checks import ConfigurationError

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def sequence_cross_entropy_with_logits(
    logits: torch.FloatTensor,
    logit_mask: Union[torch.FloatTensor, torch.BoolTensor],
    targets: torch.LongTensor,
    target_mask: Union[torch.FloatTensor, torch.BoolTensor]
) -> torch.FloatTensor:

    # lengths : (batch_size, )
    # calculated by counting number of !mask
    logit_lengths = (~ logit_mask.bool()).long().sum(1)
    target_lengths = (~ target_mask.bool()).long().sum(1)

    # log_logits : (T, batch_size, n_class), this kind of shape is required for ctc_loss
    log_logits = logits.log_softmax(2).transpose(0, 1)
    # targets (batch_size, S)
    targets = targets.long()

    loss = F.ctc_loss(log_logits, targets, logit_lengths, target_lengths)
    return loss


