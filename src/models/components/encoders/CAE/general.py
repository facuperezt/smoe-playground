from functools import partial
from logging import warning
from typing import Dict, List, Literal, Optional, Tuple, Union
import torch

from src.models.components.blocks import GeneralConvBlock, GeneralLinearBlock, SmoeActivations, ShiftedSigmoid


__all__ = [
    'GeneralCAE',
]


class CAE(torch.nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 n_kernels: int = 4,
                 block_size: int = 16,
                 hidden_dims_conv: List = None,
                 hidden_dims_lin: List = None,
                 conv_block: torch.nn.Module = GeneralConvBlock,
                 lin_block: torch.nn.Module = GeneralLinearBlock,
                 **kwargs) -> None:
        super().__init__()
        self.latent_dim = 7 * n_kernels
        if hidden_dims_conv is None:
            # Same as Elvira's model?
            hidden_dims_conv = [16, 32, 64, 128, 256, 512, 1024]
        if hidden_dims_lin is None:
            hidden_dims_lin = [1024, 512,  256, 128, 64, 24]

        self._block_size = block_size
        ret = self._build_conv_layers(in_channels, hidden_dims_conv, conv_block, **kwargs.get("conv_args", {}))
        
        if len(ret) == 2:
            self.conv, in_channels = ret
        else:
            self.conv, in_channels, block_size = ret
            self._block_size = block_size
        self.lin, in_channels = self._build_lin_layers(in_channels*block_size**2, hidden_dims_lin, lin_block)

        self.encoder = torch.nn.Sequential(
            self.conv,
            torch.nn.Flatten(),
            self.lin
        )

        self.fc = torch.nn.Linear(in_channels, self.latent_dim)

        self.output_nonlinearities = SmoeActivations(
            (2*n_kernels, 1*n_kernels, 4*n_kernels),
            (torch.nn.Sigmoid(), torch.nn.Sigmoid(), torch.nn.Identity())
            )
    
    def _build_conv_layers(
                self,
                in_channels: int,
                hidden_dims_conv: List[int],
                conv_block: GeneralConvBlock,
                curr_block_size: Optional[int] = None,
                downsample: bool = False,
                min_block_size: int = 2,
                block_size_reduction: int = 2,
            ) -> torch.nn.Sequential:
        if int(block_size_reduction) != block_size_reduction:
            warning(f"Block size reduction got corrected from {block_size_reduction} to {int(block_size_reduction)}")
            block_size_reduction = int(block_size_reduction)
        if curr_block_size is None:
            curr_block_size = self._block_size
        conv_modules = []
        for h_dim in hidden_dims_conv:
            if (
                    downsample and
                    in_channels < h_dim and
                    curr_block_size//block_size_reduction >= min_block_size and 
                    in_channels != 1
                ):
                curr_block_size = curr_block_size // block_size_reduction
                _stride = block_size_reduction
            else:
                _stride = 1
            layer = conv_block(in_channels=in_channels, out_channels=h_dim, stride=_stride)
            conv_modules.append(layer)
            in_channels = h_dim
        return torch.nn.Sequential(*conv_modules), in_channels, curr_block_size
    
    def _build_lin_layers(self, in_channels: int, hidden_dims_lin: List[int], lin_block: GeneralLinearBlock) -> torch.nn.Sequential:
        lin_modules = []
        for h_dim in hidden_dims_lin:
            layer = lin_block(in_channels, h_dim)
            lin_modules.append(layer)
            in_channels = h_dim
        return torch.nn.Sequential(*lin_modules), in_channels

    def encode(self, input: torch.Tensor) -> List[torch.Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = input
        # result = self.encoder(result)
        for layer in self.encoder:
            result = layer(result)
        return self.fc(result)
    
    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(x.shape) == 3:
            x = x[:, None, :, :]
        x = self.encode(x)
        return self.output_nonlinearities(x)

class GeneralCAE(CAE):
    def __init__(
            self,
            in_channels: int = 1,
            n_kernels: int = 4,
            block_size: int = 16,
            hidden_dims_conv: List = None,
            hidden_dims_lin: List = None,
            kernels_outside: bool = False,
            negative_experts: bool = False,
            downsample: Optional[Dict] = None,
            batch_norm: Optional[Dict[str, bool]] = None,
            bias: Optional[Dict[str, bool]] = None,
            residual: Optional[Dict[str, bool]] = None,
            dropout: Optional[Dict[str, float]] = None,
            order: str = "lbad",
            activation: Union[torch.nn.Module, Literal["relu", "swish", "lrelu"]] = "relu",
            **kwargs
        ) -> None:
        if downsample is None:
            downsample = {"active": False}
        if batch_norm is None:
            batch_norm = {"conv": False, "lin": False}
        if bias is None:
            bias = {"conv": True, "lin": True}
        if residual is None:
            residual = {"conv": False, "lin": False}
        if dropout is None:
            dropout = {"conv": 0.0, "lin": 0.0}
        if not isinstance(activation, dict):
            activation = {"conv": activation, "lin": activation}
        if not isinstance(order, dict):
            order = {"conv": order, "lin": order}
        ### Convolutional layers ###
        conv_block = partial(GeneralConvBlock, batch_norm=batch_norm["conv"], bias=bias["conv"], residual=residual["conv"], dropout=dropout["conv"], order=order["conv"], activation=activation["conv"])
        lin_block = partial(GeneralLinearBlock, batch_norm=batch_norm["lin"], bias=bias["lin"], residual=residual["lin"], dropout=dropout["lin"], order=order["lin"], activation=activation["lin"])
        if downsample["active"]:
            self._build_conv_layers = partial(
                self._build_conv_layers,
                downsample=downsample["active"],
                min_block_size=downsample.get("min_block_size", 2),
                block_size_reduction=downsample.get("block_size_reduction", 2)
            )
        super().__init__(in_channels, n_kernels, block_size, hidden_dims_conv, hidden_dims_lin, conv_block, lin_block, **kwargs)
        
        ### Output non-linearities ###
        if kernels_outside:
            xy_nonlinearity = ShiftedSigmoid(**kwargs.get("xy_nonlinearity_params", {"shift": -1.5, "scale": 2.5}))
        else:
            xy_nonlinearity = torch.nn.Sigmoid()
        if negative_experts:
            nu_nonlinearity = ShiftedSigmoid(**kwargs.get("nu_nonlinearity_params", {"shift": -1.0, "scale": 3.0}))
        else:
            nu_nonlinearity = torch.nn.Sigmoid()
        steer_nonlinearity = torch.nn.Identity()
        self.output_nonlinearities = SmoeActivations(
            (2*n_kernels, 1*n_kernels, 4*n_kernels),
            (xy_nonlinearity, nu_nonlinearity, steer_nonlinearity)
            )