import torch
from typing import Optional

def match_kernels_per_block(ground_truth: torch.Tensor, prediction: torch.Tensor, n_kernels: Optional[int] = None):
    if n_kernels is None:
        n_kernels = int(ground_truth.shape[-1]//7)
    # Get centers
    gt_x = ground_truth[:, :n_kernels]
    gt_y = ground_truth[:, n_kernels:2*n_kernels]
    gt_centers = torch.stack([gt_x, gt_y], dim=-1)

    pred_x = prediction[:, :n_kernels]
    pred_y = prediction[:, n_kernels:2*n_kernels]
    pred_centers = torch.stack([pred_x, pred_y], dim=-1)

    # Get pairwise distances
    pairwise_distances = torch.abs(gt_centers - pred_centers).sum()
