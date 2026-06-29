import torch.nn as nn
from torch import Tensor
from .layers.low_levels import ChannelLastConv1d, Modulation, modulate


class FinalBlock(nn.Module):

    def __init__(self, dim: int, out_dim: int):
        super(FinalBlock, self).__init__()
        self.adaLN_modulation = Modulation(dim, multiplier=2)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.conv = ChannelLastConv1d(dim, out_dim, kernel_size=7, padding=3)

    def forward(self, latent: Tensor, c: Tensor):
        """
        Args:
            latent (Tensor):  (batch_size, seq_len, hidden_dim)
            c (Tensor): global condition vector (batch_size, cond_dim)
        Returns:
            Tensor: (batch_size, seq_len, out_dim)
        """
        shift, scale = self.adaLN_modulation(c)
        latent = modulate(self.norm(latent), shift, scale)
        latent = self.conv(latent)
        return latent
