from typing import Tuple
import torch
import numpy as  np

class VanillaSMoE(torch.nn.Module):
    """
    SMoE implemented in PyTorch, which allows for the gradients to be calculated with autograd
    """
    def __init__(self, n_kernels: int = 4, block_size: int = 8, device: torch.device = torch.device("cpu"), rescale_range: Tuple[int, int] = (None, None)):
        super().__init__()
        self.n_kernels = n_kernels
        self.block_size = block_size
        x = torch.linspace(0, 1, block_size, dtype=torch.float32)
        y = torch.linspace(0, 1, block_size, dtype=torch.float32)
        self.domain_init = torch.nn.Parameter(torch.tensor(np.array(np.meshgrid(x, y)).T.reshape([block_size ** 2, 2]), dtype=torch.float32, device=device, requires_grad=False))
        self.rescale_range = rescale_range
        pass

    def forward(self, x: torch.Tensor) -> np.ndarray:
        if x.ndim == 2:
            out = self.torch_smoe(x)
        else:
            out = torch.stack([self.torch_smoe(_x) for _x in x])
        if not any(range_limit is None for range_limit in self.rescale_range):
            assert self.rescale_range[1] - self.rescale_range[0] > 0, "Rescale Range in SMoE Decoder is invalid."
            out = (out*(self.rescale_range[1] - self.rescale_range[0]))+self.rescale_range[0]
        return out

    def to(self, device: torch.device):
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.domain_init.to(self.device)
        return self

    def cuda(self):
        self.device = torch.device("cuda")
        self.domain_init.to(self.device)
        return self

    def torch_smoe(self, arr):
        block_size = self.block_size
        kernel_num = self.n_kernels

        domain_init = self.domain_init.clone()
        center_x = arr[:, :kernel_num]
        center_y = arr[:, kernel_num:2 * kernel_num]
        A_NN = arr[:, 3 * kernel_num:].reshape([-1, kernel_num, 2, 2])
        A_NN = torch.tril(A_NN)
        nue_e = arr[:, 2 * kernel_num:3 * kernel_num]
        shape_x = center_x.shape
        reshape_x = center_x.view(shape_x[0], kernel_num, 1)
        reshape_y = center_y.view(shape_x[0], kernel_num, 1)
        centers = torch.cat([reshape_x, reshape_y], dim=2).view(shape_x[0], kernel_num, 2)

        musX = centers.unsqueeze(2)
        domain_exp = torch.unsqueeze(torch.unsqueeze(domain_init, dim=0), dim=0).expand(musX.shape[0], musX.shape[1], block_size * block_size, 2)
        x_sub_mu = (domain_exp - musX).unsqueeze(-1)
        n_exp = torch.exp(-0.5 * torch.einsum('abcli,ablm,abnm,abcnj->abc', x_sub_mu, A_NN, A_NN, x_sub_mu))

        n_w_norm = n_exp.sum(dim=1, keepdim=True)
        n_w_norm = torch.clamp(n_w_norm, min=1e-8)

        w_e_op = n_exp / n_w_norm

        res = (w_e_op * nue_e.unsqueeze(-1)).sum(dim=1)
        if any(range_limit is not None for range_limit in self.rescale_range):
            res = torch.clamp(res, min=self.rescale_range[0], max=self.rescale_range[1])
        res = res.view(-1, block_size, block_size)

        return res.reshape(-1, 1, block_size, block_size)