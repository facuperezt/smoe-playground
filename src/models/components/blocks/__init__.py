"""This module contains all of the building blocks for the models, 
it's also responsible of importing them all and managing the name that's exposed to the outside world
"""
from .activation_functions import Swish, CustomizableSmoeLastLayerActivations as SmoeActivations, ShiftedSigmoid
from .vqgan_blocks import NonLocalBlock as NonLocalAttentionBlock, ResidualBlock as VqGanResidualBlock, DownSampleBlock as VqGanDownsampleBlock
from .conv_blocks import GeneralConvBlock
from .gan_blocks import PatchGanDiscriminator
from .linear_blocks import GeneralLinearBlock
from .normalization_blocks import GroupNorm
from .vq_blocks import Codebook as VqCodebook
from .residual_blocks import ResNetBlock as ResNetResidualBlock