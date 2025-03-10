import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout_p, max_len, device, dtype):
        super().__init__()

        self._dropout_p = dropout_p
        self.positional_encoding = torch.zeros(max_len, d_model, requires_grad=False, device=device, dtype=dtype)
        position = torch.arange(0, max_len, requires_grad=False, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, requires_grad=False, device=device, dtype=dtype) * (-math.log(10000.) / d_model)
        )

        self.positional_encoding[:, 0::2] = torch.sin(position * div_term)
        self.positional_encoding[:, 1::2] = torch.cos(position * div_term)
        #self.positional_encoding = self.positional_encoding.unsqueeze(0)


    def forward(self, x):
        x = x + self.positional_encoding[:x.size(-2), :]
        x = nn.functional.dropout(x, p=self._dropout_p, training=self.training)
        return x