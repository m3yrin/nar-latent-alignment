import os

from typing import Dict, List, Iterator
from overrides import overrides
import logging

from allennlp.data import Instance
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField
from allennlp.common.util import START_SYMBOL, END_SYMBOL

logger = logging.getLogger(__name__)

@DatasetReader.register("small_parallel_enja_reader")
class SmallParallelEnJaReader(DatasetReader):
    def __init__(self, 
                 direction: str = "ja-en",
                 source_token_indexers: Dict[str, TokenIndexer] = None,
                 target_token_indexers: Dict[str, TokenIndexer] = None, 
                 add_start_end_tokens: bool = False,
                 lazy: bool = False) -> None:
        
        super().__init__(lazy)
        self.direction = direction
        self.source_token_indexers = source_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.target_token_indexers = target_token_indexers or {"tokens": SingleIdTokenIndexer(namespace='target_tokens')}
        self.add_start_end_tokens = add_start_end_tokens

    @overrides
    def text_to_instance(self, 
                         tokens_src: List[Token],
                         tokens_tgt: List[Token] = None) -> Instance:
        

        if self.add_start_end_tokens:
            tokens_src.insert(0, Token(START_SYMBOL))
            tokens_src.append(Token(END_SYMBOL))

            if tokens_tgt:
                tokens_tgt.insert(0, Token(START_SYMBOL))
                tokens_tgt.append(Token(END_SYMBOL))



        field_src = TextField(tokens_src, self.source_token_indexers)
        fields = {"source_tokens": field_src}
        
        if tokens_tgt:
            field_tgt = TextField(tokens_tgt, self.target_token_indexers)
            fields["target_tokens"] = field_tgt

        return Instance(fields)

    @overrides
    def _read(self, data_prefix: str) -> Iterator[Instance]:

        if self.direction == "ja-en":
            src_path = data_prefix + ".ja"
            tgt_path = data_prefix + ".en"
        elif self.direction == "en-ja":
            src_path = data_prefix + ".en"
            tgt_path = data_prefix + ".ja"
        else:
            raise NotImplementedError()

        logger.info("Source file: %s", src_path)
        f_src = open(src_path, encoding='utf-8')
        
        if os.path.exists(tgt_path):
            logger.info("Target file: %s", src_path)
            f_tgt = open(tgt_path, encoding='utf-8')
        else:
            logger.info("Target file is not found.")
            f_tgt = None
        
        # read files
        for line_src in f_src:
            words_src = line_src.split()

            if f_tgt is not None:
                words_tgt = f_tgt.readline().split()
                yield self.text_to_instance([Token(w) for w in words_src], 
                                            [Token(w) for w in words_tgt])
            else:
                yield self.text_to_instance([Token(w) for w in words_src])

        # end of _read, close i/o
        f_src.close()
        if f_tgt:
            f_src.close()