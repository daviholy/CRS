from torch.nn import Module, Embedding, Linear, Sequential
from torch.nn.functional import softmax
from torch import Tensor

class skip_gram(Module):

    def __init__(self, output_size: int) -> None:
        super().__init__()
        self._embedding = Sequential(Embedding(output_size, output_size))
        self._linear = Linear(output_size,output_size)

    def forward(self, x: Tensor):
        x = self._embedding(x)
        x = self._linear(x)
        if not self.training:
            x = softmax(x)
        return x


