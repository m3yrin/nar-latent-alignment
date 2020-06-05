import torch.nn as nn
import torch.nn.functional as F

# pylint: disable=too-many-lines
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar, Union
import logging
import torch
from allennlp.common.checks import ConfigurationError

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class LinearUpsample(nn.Module):
    def __init__(self, input_size: torch.LongTensor, s: torch.LongTensor = 3) -> None:

        super(LinearUpsample, self).__init__()

        self._input_size = input_size
        self._s = s
        self.mlp = torch.nn.Linear(input_size, input_size * s)
        nn.init.xavier_normal_(self.mlp.weight)
        

    def forward(self, x, mask):
        (batch_size, sequence_length, input_size) = x.shape

        _x = self.mlp(x)
        _x = _x.reshape(batch_size, sequence_length * self._s, input_size)

        _mask = mask.unsqueeze(-1).expand(-1, -1,self._s)
        _mask = _mask.reshape(batch_size,-1)

        assert _x.size(1) == _mask.size(1)

        return _x, _mask