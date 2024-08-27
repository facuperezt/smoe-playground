from typing import Tuple
import torch

__all__ = [
    "CustomizableSmoeLastLayerActivations",
]

class CustomizableSmoeLastLayerActivations(torch.nn.Module):
    def __init__(self, group_sizes: Tuple[int], activations: Tuple):
        super().__init__()
        self.group_sizes = group_sizes
        self.group_activation = activations

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = []
        _base = 0
        for group_size, act in zip(self.group_sizes, self.group_activation):
            out.append(act(x[:, _base:_base + group_size]))
            _base += group_size
        return torch.cat(out, dim=1)