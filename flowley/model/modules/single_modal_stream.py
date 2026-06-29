import torch
import torch.nn as nn
from typing import Tuple
from torch import Tensor
from .layers.attention import SelfAttention, CrossAttention
from .layers.low_levels import MLP, ConvMLP, Modulation, ChannelLastConv1d, modulate


class SingleModalStreamBlockDiT(nn.Module):

    def __init__(
        self,
        dim: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        attn_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
        qk_scale: float = None,
        kernel_size: int = 7,
        padding: int = 3,
    ):
        super(SingleModalStreamBlockDiT, self).__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = SelfAttention(
            dim, n_heads, qk_norm=True, qk_scale=qk_scale, rms=True, dropout=attn_dropout
        )
        if kernel_size == 1:
            self.linear1 = nn.Linear(dim, dim)
            self.ffn = MLP(dim, int(mlp_ratio * dim), dropout=mlp_dropout)
        else:
            self.linear1 = ChannelLastConv1d(dim, dim, kernel_size=kernel_size, padding=padding)
            self.ffn = ConvMLP(dim,
                               int(mlp_ratio * dim),
                               dropout=mlp_dropout,
                               kernel_size=kernel_size,
                               padding=padding)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.adaLN_modulation = Modulation(dim, multiplier=6)

    def pre_attention(self, x: Tensor, c: Tensor, rot: Tensor):
        modulation = self.adaLN_modulation(c)
        shift_msa, scale_msa, gate_msa, shift_ffn, scale_ffn, gate_ffn = modulation

        x = modulate(self.norm1(x), shift_msa, scale_msa)
        q, k, v = self.attn.pre_attention(x, rot)
        return (q, k, v), (gate_msa, shift_ffn, scale_ffn, gate_ffn)

    def post_attention(self, x: Tensor, attn_out: Tensor, c: Tuple[Tensor]):
        gate_msa, shift_ffn, scale_ffn, gate_ffn = c
        x = x + self.linear1(attn_out) * gate_msa
        r = modulate(self.norm2(x), shift_ffn, scale_ffn)
        return x + self.ffn(r) * gate_ffn

    def forward(self, x: Tensor, c: Tensor, rot: Tensor, mask: Tensor = None) -> Tensor:
        """Forward pass of the SingleModalStreamBlockDiT module.

        Args:
            x (Tensor): Audio latent tensor with shape of (batch_size, seq_len, dim).
            c (Tensor): Condition tensor with shape of (batch_size, dim).
            rot (Tensor): Rotation for Rotary PE with shape of (batch_size, seq_len, dim // 2, 2, 2).

        Returns:
            Tensor: Audio latent tensor with shape of (batch_size, seq_len, dim)
        """
        qkv, conds = self.pre_attention(x, c, rot)
        attn_out = self.attn.attention(qkv[0], qkv[1], qkv[2], mask=mask)
        output = self.post_attention(x, attn_out, conds)
        return output


class SingleModalStreamBlockDiTV2(nn.Module):

    def __init__(
        self,
        dim: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        attn_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
        qk_scale: float = None,
        kernel_size: int = 7,
        padding: int = 3,
        balance_hparam: float = None,
    ):
        super(SingleModalStreamBlockDiTV2, self).__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.self_attn = SelfAttention(
            dim, n_heads, qk_norm=True, qk_scale=qk_scale, rms=True, dropout=attn_dropout
        )
        self.vis_aud_cross_atn = CrossAttention(
            dim, n_heads, qk_norm=True, qk_scale=qk_scale, rms=True, dropout=attn_dropout
        )
        self.tex_aud_cross_atn = CrossAttention(
            dim, n_heads, qk_norm=True, qk_scale=qk_scale, rms=True, dropout=attn_dropout
        )
        if balance_hparam is not None:
            self.register_buffer("cross_weight", torch.tensor(balance_hparam))
        else:
            self.cross_weight = nn.Parameter(torch.tensor(0.5))
        if kernel_size == 1:
            self.linear1 = nn.Linear(dim, dim)
            self.ffn = MLP(dim, int(mlp_ratio * dim), dropout=mlp_dropout)
        else:
            self.linear1 = ChannelLastConv1d(dim, dim, kernel_size=kernel_size, padding=padding)
            self.ffn = ConvMLP(dim,
                               int(mlp_ratio * dim),
                               dropout=mlp_dropout,
                               kernel_size=kernel_size,
                               padding=padding)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.adaLN_modulation = Modulation(dim, multiplier=6)

    def forward(
        self,
        audio_feat: Tensor,
        visual_feat: Tensor,
        text_feat: Tensor,
        cond: Tensor,
        rot: Tensor,
        mask: Tensor = None
    ) -> Tensor:
        """Forward pass of the SingleModalStreamBlockDiT module.

        Args:
            audio_feat (Tensor): Audio latent tensor with shape of (bs, seq_len, dim).
            visual_feat (Tensor): Visual latent tensor with shape of (bs, num_frames, dim).
            text_feat (Tensor): Text latent tensor with shape of (bs, seq_len, dim).
            cond (Tensor): Condition tensor with shape of (bs, dim).
            rot (Tensor): Rotation for Rotary PE with shape of (bs, seq_len, dim // 2, 2, 2).
            mask (Tensor): Cross-attention mask with shape of (bs, seq_len, num_frames).

        Returns:
            Tensor: Audio latent tensor with shape of (bs, seq_len, dim)
        """
        modulation = self.adaLN_modulation(cond)
        shift_msa, scale_msa, gate_msa, shift_ffn, scale_ffn, gate_ffn = modulation

        # perform self-attention
        audio = modulate(self.norm1(audio_feat), shift_msa, scale_msa)
        self_attn_out = self.self_attn(audio, rot=rot)

        # perform cross-attention
        audio_feat = audio_feat + self.linear1(self_attn_out) * gate_msa
        vis_aud_cross_attn_out = self.vis_aud_cross_atn(audio_feat, visual_feat, mask=mask)
        tex_aud_cross_attn_out = self.tex_aud_cross_atn(audio_feat, text_feat, mask=None)
        audio_feat = audio_feat + \
            self.cross_weight * tex_aud_cross_attn_out + \
            (1 - self.cross_weight) * vis_aud_cross_attn_out

        # post attention
        output = modulate(self.norm2(audio_feat), shift_ffn, scale_ffn)
        output = audio_feat + self.ffn(output) * gate_ffn
        return output
