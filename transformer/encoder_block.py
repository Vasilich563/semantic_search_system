from torch import nn
from multi_head_attention import MultiHeadAttention
from feed_forward_network import FeedForwardNetwork


class EncoderBlock(nn.Module):

    def __init__(self, d_model, num_attention_heads, d_ffn_hidden, device, dtype):
        super().__init__()

        self._multi_head_attention = MultiHeadAttention(d_model, num_attention_heads, device, dtype)
        self._attention_norm = nn.LayerNorm(d_model, device=device, dtype=dtype)

        self._feed_forward_network = FeedForwardNetwork(d_model, d_ffn_hidden, device, dtype)
        self._ffn_norm = nn.LayerNorm(d_model, device=device, dtype=dtype)


    def forward(self, x, dropout_p=0, mask=None):
        residual = x
        x = self._multi_head_attention(query_x=x, key_x=x, value_x=x, dropout_p=dropout_p, mask=mask)
        x = nn.functional.dropout(x, p=dropout_p, training=self.training)

        x = residual + x
        x = self._attention_norm(x)

        residual = x
        x = self._feed_forward_network(x, dropout_p=dropout_p)
        x = nn.functional.dropout(x, p=dropout_p, training=self.training)

        x = residual + x
        x = self._ffn_norm(x)

        return x



