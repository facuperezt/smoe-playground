from typing import Dict, List, Optional
import torch


__all__ = [
    "ResNetBlock",
]


class ResNetBlock(torch.nn.Module):
    def __init__(self, in_channels: int, filter_sizes: List[int], hidden_dims: List[int], out_channels: int,
                 residual: bool = True, dropout: float = 0.1, batchnorm_parameters: Optional[Dict] = None):
        """ResNet Block with variable amount of hidden layers.

        Args:
            in_channels (int): input channels
            filter_sizes (List[int]): Filter (kernel) sizes FOR EACH layer. Needs to have at least length 2. For example [3, 4] makes {in_channels <(3x3)> hidden_dim <(4x4)> out_channels}
            hidden_dims (List[int]): Number of dimension in each hidden layer, needs to have length of at least 1
            out_channels (int): output channels
            residual (bool, optional): Whether to apply residual. Defaults to True.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
            batchnorm_parameters (Optional[Dict], optional): Parameters for torch.nn.BatchNorm2d. Defaults to None.
        """
        super().__init__()
        if batchnorm_parameters is None:
            batchnorm_parameters = {}
        assert len(hidden_dims) > 0, "Need at least one hidden layer, just use a Conv2d layer otherwise."
        assert len(filter_sizes) == len(hidden_dims) + 1
        self.residual = residual
        convs = [torch.nn.Conv2d(in_channels, out_channels=hidden_dims[0], kernel_size=filter_sizes[0])]
        for in_ch, out_ch, filter_size in zip(hidden_dims[:-1], hidden_dims[1:], filter_sizes[1:-1]):
            convs.append(torch.nn.Conv2d(in_ch, out_ch, filter_size, bias=False, padding=filter_size//2))
            convs.append(torch.nn.BatchNorm2d(out_ch, **batchnorm_parameters))
            convs.append(torch.nn.ReLU())
        convs.append(torch.nn.Conv2d(hidden_dims[-1], out_channels, filter_sizes[-1], padding=filter_sizes[-1]//2))
        convs.append(torch.nn.BatchNorm2d(out_channels, **batchnorm_parameters))
        self.convs = torch.nn.Sequential(*convs)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.convs(x)
        x = self.relu(x)
        if self.residual:
            x = residual + x
        x = self.dropout(x)
        return x