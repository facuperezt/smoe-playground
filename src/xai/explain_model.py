#%%
from IPython import get_ipython
get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "2")
from typing import List, Optional, Tuple, Union
import torch
import numpy as np
import matplotlib.pyplot as plt

from zennit.composites import EpsilonGammaBox
from zennit.canonizers import SequentialMergeBatchNorm
from zennit.attribution import Gradient

from src.models.base_model import SmoeModel
from src.models import ResNet
from src.data import DataLoader

from src.utils import plot_kernel_centers, plot_kernels_chol, plot_kernels_inv, plot_kernel_centers_inv

class Explainer:
    def __init__(self, model: SmoeModel):
        self.canonizers = [SequentialMergeBatchNorm()]
        self.composite = EpsilonGammaBox(
            low=-3.0,
            high=3.0,
            canonizers=self.canonizers
        )
        self.model = model

    def forward(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _training = self.model.training
        self.model.eval()
        with Gradient(model=self.model, composite=self.composite) as attributor:
            out, relevance = attributor(data)
        self.model.train(_training)
        return out, relevance

    def plot(self, input: torch.Tensor, max_elems: int = 4):
        max_elems = min(input.shape[0], max_elems)
        out, relevance = self.forward(input[:max_elems])
        with torch.no_grad():
            all_kernels_parameters = self.model.encoder(input[:max_elems])
        if type(all_kernels_parameters) == Tuple:
            # The first one should always be the encoding
            all_kernels_parameters = all_kernels_parameters[0]

        input = input[:max_elems].detach().cpu()
        out = out[:max_elems].detach().cpu()
        relevance = relevance[:max_elems].detach().cpu()
        
        fig, axs = plt.subplots(max_elems, 4, figsize=(2*3*max_elems, 3*3))
        axs: np.ndarray = axs.reshape((-1, 4))
        for in_block, out_block, rel_block, kernel_parameters, ax_row in zip(input, out, relevance, all_kernels_parameters, axs):
            in_block = in_block.squeeze()
            out_block = out_block.squeeze()
            rel_block = rel_block.squeeze()
            ax_row: List[plt.Axes]
            self._plot(ax_row[0], in_block)
            self._plot(ax_row[1], out_block)
            self._plot(ax_row[2], out_block)
            self._plot(ax_row[2], kernel_parameters)
            self._plot(ax_row[3], rel_block)

    def _plot(self, ax: plt.Axes, content: torch.Tensor, vmin: Optional[float] = None, vmax: Optional[float] = None):
        ax.axis("off")
        # If it looks like image format plot it as image.
        if content.squeeze().ndim > 1:
            if vmin is None:
                vmin = min(0, content.min())
            if vmax is None:
                vmax = max(1, content.max())
            ax.imshow(content, cmap="gray", vmin=vmin, vmax=vmax)
            return
        # Otherwise assume it's kernel variables in SMoE space
        plot_kernels_inv(content, ax, block_size=self.model.block_size, padding=0, n_kernels=self.model.n_kernels)
        plot_kernel_centers_inv(content, ax, block_size=self.model.block_size, padding=0, n_kernels=self.model.n_kernels)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

model = ResNet("test_xai.json")
model.load_model("ResNetWeirdness_4_k_8_bs_real_ft_<latest>")
dataloader = DataLoader("synthetic", n_kernels=model.n_kernels, block_size=model.block_size, data_path="professional_photos", img_size=384, device="cuda")
exp = Explainer(model)
model.cuda()
data = dataloader.get(2)
exp.plot(data["input"], 2)
# %%
