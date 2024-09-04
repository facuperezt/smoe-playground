# These have a lot in common, so there's a lot of duplicated code that could be optimized away.
# For now it runs but they could use some more sensible class inheritance.
# They even use the same config files... 
from .CAE import GeneralAutoCAE as AutoCaeEncoder, GeneralManualCAE as ManualCaeEncoder
from .VAE import GeneralAutoVAE as AutoVaeEncoder, GeneralManualVAE as ManualVaeEncoder
from .ResNet import ResNetEncoder
from .VQGAN import Encoder as VqGanEncoder