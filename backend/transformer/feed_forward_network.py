from torch import nn


class FeedForwardNetwork(nn.Module):

    def __init__(self, d_model, d_hidden, device, dtype):
        super().__init__()

        self._input_linear = nn.Linear(d_model, d_hidden, bias=True, device=device, dtype=dtype)
        self._input_linear_activation = nn.GELU()
        self._output_linear = nn.Linear(d_hidden, d_model, bias=True, device=device, dtype=dtype)

        self._init_weights()


    def _init_weights(self):
        nn.init.kaiming_uniform_(self._input_linear.weight)
        nn.init.kaiming_uniform_(self._output_linear.weight)


    def forward(self, x, dropout_p=0):
        x = self._input_linear(x)
        x = self._input_linear_activation(x)
        x = nn.functional.dropout(x, p=dropout_p, training=self.training)
        x = self._output_linear(x)
        return x




