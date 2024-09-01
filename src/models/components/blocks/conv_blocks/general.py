from typing import Literal, Tuple, Union
import torch

__all__ = [
    "GeneralConvBlock",
]

class GeneralConvBlock(torch.nn.Module):
    """
    General Convolutional Block with the following order:
    l: Layer
    b: BatchNorm
    a: Activation
    d: Dropout
    r: Residual
    
    Args:
    in_channels (int): Number of input channels
    out_channels (int): Number of output channels
    kernel_size (Tuple[int, int]): Kernel size
    stride (int): Stride
    padding (int): Padding
    batch_norm (bool): Whether to use batch normalization
    bias (bool): Whether to use bias
    residual (bool): Whether to use residual connection
    dropout (float): Dropout rate
    order (str): Order of operations
    activation (str): Activation function, one of "relu", "swish", "lrelu". Alternatively, pass an initialized Module like torch.nn.ReLU()

    Returns:
    torch.Tensor: Output tensor
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, int] = (3, 3),
                 stride: int = 1,
                 padding: int = 1,
                 batch_norm: bool = False,
                 bias: bool = False,
                 residual: bool = True,
                 dropout: float = 0.0,
                 order: str = "lbadr",
                 activation: Union[torch.nn.Module, Literal["relu", "swish", "lrelu", "gelu"]] = "relu",
                 ):
        super().__init__()
        if type(activation) == str:
            activation = {
                "relu": torch.nn.ReLU(),
                "swish": torch.nn.SiLU(),
                "lrelu": torch.nn.LeakyReLU(),
                "gelu": torch.nn.GELU()
            }[activation]
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        if batch_norm:
            self.bn = torch.nn.BatchNorm2d(out_channels)
        else:
            self.bn = torch.nn.Identity()
        self.activation = activation
        if not residual:
            order = order.replace("r", "")
        if residual and in_channels != out_channels:
            self.downsample_residual = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        else:
            self.downsample_residual = torch.nn.Identity()
        self.dropout = torch.nn.Dropout(dropout)
        self.order = order
        self._apply_order_dict = {
            "l": self.conv,
            "b": self.bn,
            "a": self.activation,
            "d": self.dropout,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _x = x
        for order in self.order:
            if order == "r":
                _x = self.downsample_residual(_x)
                x = x + _x
            else:
                x = self._apply_order_dict[order](x)
        return x
