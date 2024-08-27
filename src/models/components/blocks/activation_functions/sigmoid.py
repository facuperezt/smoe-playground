import torch

__all__ = [
    "ShiftedSigmoid",
]

class ShiftedSigmoid(torch.nn.Module):
    def __init__(self, shift: float = -1, scale: float = 2):
        super().__init__()
        self.shift = shift
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x) * self.scale + self.shift