import logging
from typing import Literal, Tuple, Union
import torch


class GeneralLinearBlock(torch.nn.Module):
    """
    General Linear Block with the following order:
    l: Layer
    b: BatchNorm
    a: Activation
    d: Dropout
    r: Residual
    
    Args:
    in_features (int): Number of input features
    out_features (int): Number of output features
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
                 in_features: int,
                 out_features: int,
                 batch_norm: bool = False,
                 bias: bool = False,
                 residual: bool = False,
                 dropout: float = 0.0,
                 order: str = "lbadr",
                 activation: Union[torch.nn.Module, Literal["relu", "swish", "lrelu"]] = "relu",
                 ):
        super().__init__()
        if type(activation) == str:
            activation = {
                "relu": torch.nn.ReLU(),
                "swish": torch.nn.SiLU(),
                "lrelu": torch.nn.LeakyReLU()
            }[activation]
        self.fc = torch.nn.Linear(in_features, out_features, bias=bias)

        if batch_norm:
            self.bn = torch.nn.BatchNorm1d(out_features)
        else:
            self.bn = torch.nn.Identity()
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = activation
        if residual and in_features != out_features:
            logging.warning("Residual connection is not possible with different input and output features. Disabling residual connection.")
            residual = False
        if not residual:
            order = order.replace("r", "")
        self.order = order
        self._apply_order_dict = {
            "l": self.fc,
            "b": self.bn,
            "a": self.activation,
            "d": self.dropout,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _x = x
        for order in self.order:
            if order == "r":
                x = x + _x
            else:
                x = self._apply_order_dict[order](x)
        return x
