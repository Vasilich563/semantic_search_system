import torch
from transformers import RobertaTokenizerFast
from backend.transformer.bidirectional_transformer import BidirectionalTransformer
from backend.embedding_system.snippet_bounds import SnippetBounds
from backend.embedding_system.db_crud import DBCrud


class EmbeddingSystem:

    def __init__(self, tokenizer: RobertaTokenizerFast, embedding_model: BidirectionalTransformer, db_crud: DBCrud):
        self._tokenizer: RobertaTokenizerFast = tokenizer
        self._embedding_model: BidirectionalTransformer = embedding_model
        self._embedding_model.eval()

        self._db_crud = db_crud

        self._level_1_max_len = 16
        self._level_1_stride = 8

        self._level_2_max_len = 64
        self._level_2_stride = 32

        self._level_3_max_len = 256
        self._level_3_stride = 128


    @staticmethod
    def _get_snippet_bounds(offset_mapping):
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


    def _tokenize_text(self, text, max_len, stride, return_snippet_bounds):
        with torch.no_grad():
            tokenized_text = self._tokenizer(
                text, padding="max_length", truncation=True, max_length=max_len, stride=stride,
                return_overflowing_tokens=True, return_tensors="pt", return_offsets_mapping=return_snippet_bounds
            )
            if return_snippet_bounds:
                snippet_bounds = self._get_snippet_bounds(tokenized_text["offset_mapping"])
                return tokenized_text["input_ids"], tokenized_text["attention_mask"], snippet_bounds
            else:
                return tokenized_text["input_ids"], tokenized_text["attention_mask"]


    def _count_text_embeddings(self, text_input_ids, text_attention_mask, mean_along_batch):
        with torch.no_grad():
            text_input_ids = text_input_ids.to(self._embedding_model.device)
            text_attention_mask = text_attention_mask.to(self._embedding_model.device)
            text_embedding_batch = self._embedding_model.forward(text_input_ids, text_attention_mask)

            if mean_along_batch:
                return [text_embedding_batch.mean(dim=0).detach().cpu().tolist()]
            else:
                return text_embedding_batch.detach().cpu().tolist()


    @staticmethod
    def _windows_before_next_level(next_level_max_len, cur_level_max_len, cur_level_stride):
        return ((next_level_max_len - cur_level_max_len) // cur_level_stride) + 1


    async def index_new_text(self, text, text_path):
        text_input_ids, text_attention_mask, snippet_bounds = self._tokenize_text(
            text, self._level_1_max_len, self._level_1_stride, return_snippet_bounds=True
        )
        text_embeddings = self._count_text_embeddings(text_input_ids, text_attention_mask, mean_along_batch=False)
        # TODO save

        text_input_ids, text_attention_mask, snippet_bounds = self._tokenize_text(
            text, self._level_2_max_len, self._level_2_stride, return_snippet_bounds=True
        )
        text_embeddings = self._count_text_embeddings(text_input_ids, text_attention_mask, mean_along_batch=False)
        # TODO save

        text_input_ids, text_attention_mask, snippet_bounds = self._tokenize_text(
            text, self._level_3_max_len, self._level_3_stride, return_snippet_bounds=True
        )
        text_embeddings = self._count_text_embeddings(text_input_ids, text_attention_mask, mean_along_batch=False)
        # TODO save


    async def handle_user_query(self, query, limit=25):
        input_ids, attention_mask = self._tokenize_text(
            query, self._level_1_max_len, self._level_1_stride, return_snippet_bounds=False
        )
        if input_ids.shape[0] > self._windows_before_next_level(self._level_2_max_len, self._level_1_max_len, self._level_1_stride):
            input_ids, attention_mask = self._tokenize_text(
                query, self._level_2_max_len, self._level_2_stride, return_snippet_bounds=False
            )
            if input_ids.shape[0] > self._windows_before_next_level(self._level_3_max_len, self._level_2_max_len, self._level_2_stride):
                input_ids, attention_mask = self._tokenize_text(
                    query, self._level_3_max_len, self._level_3_stride, return_snippet_bounds=False
                )
        # TODO query


    async def remove_document(self, document_path):
        # TODO remove
        pass


    async def update_document(self, document_path, new_text):
        # TODO delete old
        # TODO write new
        pass


    async def change_document_path(self, old_document_path, new_document_path):
        pass


    async def get_text_by_name(self, document_name):










