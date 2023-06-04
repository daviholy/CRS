from torch import Tensor
from torch.nn import BatchNorm1d, Embedding, Linear, Module, Sequential, SiLU
from torch.nn.functional import sigmoid


class SkipGram(Module):
    def __init__(self, vocabulary_size: int, embedding_size=100) -> None:
        super().__init__()
        self.eval()

        self._embedding = Embedding(
            vocabulary_size, embedding_size, scale_grad_by_freq=True, max_norm=1
        )
        self._linear = Linear(embedding_size, vocabulary_size)

    def forward(self, x: Tensor):
        x = self._embedding(x)
        return self._linear(x) if self.training else x


class Classifier(Module):
    def __init__(self, embedd_size: int, output_size: int) -> None:
        super().__init__()
        self.eval()

        self._hidden_linear = Sequential(
            Linear(embedd_size, output_size // 2),
            BatchNorm1d(output_size // 2),
            SiLU(),
        )
        self._output_layer = Linear(output_size // 2, output_size)

    def forward(self, x: Tensor):
        x = self._hidden_linear(x)
        x = self._output_layer(x)
        return x if self.training else sigmoid(x)
