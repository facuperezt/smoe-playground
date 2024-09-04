import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "NonLocalBlock",
]

class NonLocalBlock(nn.Module):
    def __init__(self, channels, n_groups: int = 32):
        super(NonLocalBlock, self).__init__()
        self.in_channels = channels

        self.gn = torch.nn.GroupNorm(num_channels=channels, num_groups=n_groups, eps=1e-6, affine=True)
        self.q = nn.Conv2d(channels, channels, 1, 1, 0)
        self.k = nn.Conv2d(channels, channels, 1, 1, 0)
        self.v = nn.Conv2d(channels, channels, 1, 1, 0)
        self.proj_out = nn.Conv2d(channels, channels, 1, 1, 0)

    def forward(self, x):
        h_ = self.gn(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape

        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)
        k = k.reshape(b, c, h*w)
        v = v.reshape(b, c, h*w)

        attn = torch.bmm(q, k)
        attn = attn * (int(c)**(-0.5))
        attn = F.softmax(attn, dim=2)
        attn = attn.permute(0, 2, 1)

        A = torch.bmm(v, attn)
        A = A.reshape(b, c, h, w)

        return x + A

