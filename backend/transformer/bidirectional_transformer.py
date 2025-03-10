import torch
from torch import nn
from encoder import Encoder
from positional_encoding import PositionalEncoding


def make_mask(hugging_face_mask, device, dtype):
    mask = torch.zeros_like(hugging_face_mask, dtype=dtype, device=device, requires_grad=False)
    mask [hugging_face_mask == 0] = -torch.inf
    mask = mask.unsqueeze(-2).unsqueeze(-2)  # unsqueeze(-2) to deal with heads of attention, next unsqueeze(-2) to deal with every token in sequence
    # Mask shape is [Batch-size, 1, 1, D-model]
    return mask




class BidirectionalTransformer(nn.Module):

    def __init__(
        self, vocab_size, max_len, num_layers, d_model, num_attention_heads, d_ffn_hidden, dropout_p, device, dtype,
        padding_index=None
    ):
        super().__init__()

        self._encoder = Encoder(num_layers, d_model, num_attention_heads, d_ffn_hidden, dropout_p, device, dtype)
        self._embeddings = nn.Embedding(vocab_size, d_model, padding_idx=padding_index, device=device, dtype=dtype)
        self._positional_encoding = PositionalEncoding(d_model, dropout_p, max_len, device, dtype)
        self._output_linear = nn.Linear(d_model, vocab_size, bias=True, device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype




    def forward(self, x, hugging_face_mask=None):
        if hugging_face_mask is not None:
            mask = make_mask(hugging_face_mask, self.device, self.dtype)
        else:
            mask = None
        hidden_state = self._embeddings(x)
        hidden_state = self._positional_encoding(hidden_state)
        hidden_state = self._encoder(hidden_state, mask=mask)

        return hidden_state


    def train_forward(self, x, hugging_face_mask=None):
        hidden_state = self.forward(x, hugging_face_mask)
        output_tokens = self._output_linear(hidden_state)  # output_tokens has shape [masked_tokens, vocab_size]
        output_tokens = nn.functional.log_softmax(output_tokens, dim=-1)
        return output_tokens
