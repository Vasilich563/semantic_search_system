import torch
from torch import nn
from encoder_block import EncoderBlock


class Encoder(nn.Module):

    def __init__(self, num_layers, d_model, num_attention_heads, d_ffn_hidden, dropout_p, device, dtype):
        super().__init__()
        self._blocks = nn.ModuleList()
        self._dropout_p = dropout_p

        for _ in range(num_layers):
            self._blocks.append(
                EncoderBlock(d_model, num_attention_heads, d_ffn_hidden, device, dtype)
            )


    def forward(self, x, mask):
        for block in self._blocks:
            x = block(x, mask=mask, dropout_p=self._dropout_p)

        return x



