import math
import torch
from torch import nn
from torch.nn import functional as F


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


class Modulation(nn.Module):

    def __init__(self, dim: int, multiplier: int):
        super(Modulation, self).__init__()
        self.multiplier = multiplier
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * multiplier, bias=True)
        )

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        out = self.adaLN_modulation(c).unsqueeze(1).chunk(self.multiplier, dim=-1)
        return out


class ChannelLastConv1d(nn.Conv1d):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = super().forward(x)
        x = x.permute(0, 2, 1)
        return x


# https://github.com/Stability-AI/sd3-ref
class MLP(nn.Module):

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        multiple_of: int = 256,
    ):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            dropout (float): Dropout probability.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.

        Attributes:
            w1 (ColumnParallelLinear): Linear transformation for the first layer.
            w2 (RowParallelLinear): Linear transformation for the second layer.
            w3 (ColumnParallelLinear): Linear transformation for the third layer.

        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.dropout = nn.Dropout(dropout)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x)))


class ConvMLP(nn.Module):

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        multiple_of: int = 256,
        kernel_size: int = 3,
        padding: int = 1,
    ):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            dropout (float): Dropout probability.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.

        Attributes:
            w1 (ColumnParallelLinear): Linear transformation for the first layer.
            w2 (RowParallelLinear): Linear transformation for the second layer.
            w3 (ColumnParallelLinear): Linear transformation for the third layer.

        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.dropout = nn.Dropout(dropout)
        self.w1 = ChannelLastConv1d(dim,
                                    hidden_dim,
                                    bias=False,
                                    kernel_size=kernel_size,
                                    padding=padding)
        self.w2 = ChannelLastConv1d(hidden_dim,
                                    dim,
                                    bias=False,
                                    kernel_size=kernel_size,
                                    padding=padding)
        self.w3 = ChannelLastConv1d(dim,
                                    hidden_dim,
                                    bias=False,
                                    kernel_size=kernel_size,
                                    padding=padding)

    def forward(self, x):
        return self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x)))


class FusionProjector(nn.Module):
    def __init__(
        self,
        inp_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ):
        super(FusionProjector, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(inp_dim, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x):
        return self.mlp(x)


class CrossModalityProjector(nn.Module):
    def __init__(
        self,
        inp_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ):
        super(CrossModalityProjector, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(inp_dim, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
        )

    def forward(self, x):
        return self.proj(x)


class RMSNorm(nn.Module):
    def __init__(self, d: int, p: float = -1., eps: float = 1e-6, bias: bool = False):
        """
            Root Mean Square Layer Normalization
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-6
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed


class TimestepEmbedder(nn.Module):

    def __init__(self, dim: int, frequency_embed_size: int, max_period: int = 10000):
        super(TimestepEmbedder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embed_size, dim, bias=True),
            nn.SiLU(),
            nn.Linear(dim, dim, bias=True),
        )
        self.frequecy_embed_size = frequency_embed_size
        self.max_period = max_period

    def timestep_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """
        Create sinusoidal embeddings of timesteps.
        https://github.com/Stability-AI/sd3.5

        Args:
            t (torch.Tensor): Timesteps to embed with shape of (N,).
            max_period (int): Maximum period to consider.

        Returns:
            torch.Tensor: Sinusoidal embeddings of timesteps with shape of (N, dim).
        """
        half = self.frequecy_embed_size // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(t)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.frequecy_embed_size % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        if torch.is_floating_point(t):
            embedding = embedding.to(t)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t)
        t_emb = self.mlp(t_freq)
        return t_emb
