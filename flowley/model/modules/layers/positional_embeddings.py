import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Union
from einops import rearrange


"""
RoPE for 1D data (text)
"""


# https://github.com/hkchengrex/MMAudio
def compute_rope_rotations(length: int,
                           dim: int,
                           theta: int,
                           *,
                           freq_scaling: float = 1.0,
                           device: Union[torch.device, str] = 'cpu') -> Tensor:
    assert dim % 2 == 0

    with torch.amp.autocast(device_type='cuda', enabled=False):
        pos = torch.arange(length, dtype=torch.float32, device=device)
        scale = torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim
        freqs = 1.0 / (theta**scale)
        freqs *= freq_scaling

        rot = torch.einsum('..., f -> ... f', pos, freqs)
        rot = torch.stack([torch.cos(rot), -torch.sin(rot), torch.sin(rot), torch.cos(rot)], dim=-1)
        rot = rearrange(rot, 'n d (i j) -> 1 n d i j', i=2, j=2)
        return rot


def apply_rope(x: Tensor, rot: Tensor) -> tuple[Tensor, Tensor]:
    with torch.amp.autocast(device_type='cuda', enabled=False):
        _x = x.float()
        _x = _x.view(*_x.shape[:-1], -1, 1, 2)
        x_out = rot[..., 0].unsqueeze(1) * _x[..., 0] + rot[..., 1].unsqueeze(1) * _x[..., 1]
        return x_out.reshape(*x.shape).to(dtype=x.dtype)


class SinusoidalEmbedding(nn.Module):
    "Implement the PE function."

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.0,
        max_len: int = 5000,
        grid: int = 27,
        mode: str = 'seq'
    ):
        super(SinusoidalEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        assert mode in ['seq', 'vit']
        if mode == 'seq':
            assert max_len is not None
            pe = self.setup_1d_pe(max_len, d_model)
        else:
            assert grid is not None
            pe = self.setup_2d_pe(grid, d_model, cls_prepend=True)

        self.register_buffer("pe", pe)

    def setup_1d_pe(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def setup_2d_pe(self, n_grid, d_model, cls_prepend):
        height = width = n_grid
        pe = torch.zeros(height, width, d_model)
        # Each dimension use half of d_model
        d_model_half = d_model // 2
        div_term = torch.exp(
            torch.arange(0, d_model_half, 2) * -(math.log(10000.0) / d_model_half)
        )
        # Width (x) encodings go into the first half of the channels
        pos_w = torch.arange(0, width).unsqueeze(1)
        sin_w = torch.sin(pos_w * div_term)  # (width, d_model // 2)
        cos_w = torch.cos(pos_w * div_term)
        # sin_w, cos_w => shape (1, width, d_model_half//2) -> repeat across height
        sin_w = sin_w.T.unsqueeze(0).repeat(height, 1, 1).permute(0, 2, 1)  # (H, W, d_model // 2)
        cos_w = cos_w.T.unsqueeze(0).repeat(height, 1, 1).permute(0, 2, 1)  # (H, W, d_model // 2)

        # Similar to width, but for height (y) encodings
        pos_h = torch.arange(0, height).unsqueeze(1)

        sin_h = torch.sin(pos_h * div_term)  # (height, d_model_half//2)
        cos_h = torch.cos(pos_h * div_term)  # (height, d_model_half//2)

        # sin_h, cos_h => shape (height, 1, d_model_half//2) -> repeat across width
        sin_h = sin_h.unsqueeze(1).repeat(1, width, 1)  # (H, W, d_model_half//2)
        cos_h = cos_h.unsqueeze(1).repeat(1, width, 1)  # (H, W, d_model_half//2)

        pe[..., 0:d_model_half:2] = sin_w
        pe[..., 1:d_model_half:2] = cos_w
        pe[..., d_model_half::2] = sin_h
        pe[..., d_model_half + 1::2] = cos_h

        pe = pe.view(1, height * width, d_model)
        if cls_prepend:
            pe = torch.cat([torch.zeros(1, 1, d_model), pe], dim=1)
        else:
            pe = torch.cat([pe, torch.zeros(1, 1, d_model)], dim=1)

        return pe

    def forward(self, x: Tensor):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)
