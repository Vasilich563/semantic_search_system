from typing import List
import torch
from transformers import RobertaTokenizerFast
from backend.transformer.bidirectional_transformer import BidirectionalTransformer
from backend.embedding_system.snippet_bounds import SnippetBounds

class EmbeddingSystem:

    def __init__(self, tokenizer: RobertaTokenizerFast, embedding_model: BidirectionalTransformer):
        self._tokenizer = tokenizer
        self._embedding_model = embedding_model

        self._level_1_max_len = 16
        self._level_1_stride = 8

        self._level_2_max_len = 64
        self._level2_stride = 32

        self._level3_max_len = 256
        self._level3_stride = 128


    @staticmethod
    def _get_window_start_end_mapping(offset_mapping):
        snippet_bounds = []
        for i in range(offset_mapping.shape[0]):
            # offset_mappin[i][j] is array of len(2)
            # offset_mapping[i][0] is always [0, 0] and maps to [CLS] special token
            start = offset_mapping[i][1][0]
            # offset_mapping[i][j] is always [0, 0] and maps to [SEP] special token if j = (offset_mapping.shape[1] - 1)
            j = offset_mapping.shape[1] - 2
            while offset_mapping[i][j][1] == 0 and j > 0:  # j == 0 is maps to [CLS] special token
                j -= 1  # offset_mapping[i][j] = [0, 0] maps to [PAD], but the part of the text is needed
            end = offset_mapping[i][j][1]
            # offset_mapping[i][j][1] is the index of the element after the last element of the window
            # for the last window offset_mapping[i][j][1] == len(original_text)
            snippet_bounds.append(SnippetBounds(start, end))
        return snippet_bounds



    def _tokenize_text(self, text, max_len, stride, find_snippet_bounds):
        if find_snippet_bounds:
            tokenized_text = self._tokenizer(

            )
        else:
            pass




