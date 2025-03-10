from typing import List
import torch
from transformers import RobertaTokenizerFast
from backend.transformer.bidirectional_transformer import BidirectionalTransformer
from backend.embedding_system.embedding_layer_config import EmbeddingLayerConfig


class EmbeddingSystem:

    def __init__(
        self,
        tokenizer: RobertaTokenizerFast,
        embedding_model: BidirectionalTransformer,
        embedding_layer_config_list: List[EmbeddingLayerConfig]
    ):
        self._tokenizer = tokenizer
        self._embedding_model = embedding_model
        self._embedding_layer_config_list = embedding_layer_config_list


    def _tokenize_text(self, text, max_len, stride, find_snippet_bounds):
        if find_snippet_bounds:
            tokenized_text = self._tokenizer(

            )
        else:
            pass




