import torch
from torch import nn
from encoder import Encoder
from positional_encoding import PositionalEncoding


def make_mask(tokens, pad, dtype):
    pad_indices = (tokens == pad)
    mask = torch.zeros_like(tokens, dtype=dtype, requires_grad=False)
    mask [pad_indices] = -torch.inf
    mask = mask.unsqueeze(-2).unsqueeze(-1)  # unsqueeze(-2) to deal with heads of attention, (-1) - for d_model
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


    def forward(self, x, mask=None):
        hidden_state = self._embeddings(x)
        hidden_state = self._positional_encoding(hidden_state)
        hidden_state = self._encoder(hidden_state, mask=mask)

        return hidden_state


    def train_forward(self, x, mask=None):
        hidden_state = self.forward(x, mask)
        output_tokens = self._output_linear(hidden_state)  # output_tokens has shape [masked_tokens, vocab_size]
        output_tokens = nn.functional.log_softmax(output_tokens, dim=-1)
        return output_tokens
