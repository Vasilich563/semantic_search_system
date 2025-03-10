class EmbeddingLayerConfig:

    def __init__(self, sequence_length, stride):
        self._sequence_length = sequence_length
        self._stride = stride

    @property
    def sequence_length(self):
        return self._sequence_length

    @property
    def stride(self):
        return self._stride
