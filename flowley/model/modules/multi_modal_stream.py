import torch
import torch.nn as nn
from einops import repeat
from .single_modal_stream import SingleModalStreamBlockDiT
from .layers.attention import attention


class MultiModalStreamBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        attn_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
        qk_scale: float = None,
        pre_only: bool = False,
    ):
        super(MultiModalStreamBlock, self).__init__()
        self.n_heads = n_heads
        self.attn_dropout = attn_dropout
        self.qk_scale = qk_scale
        self.pre_only = pre_only
        self.audio_block = SingleModalStreamBlockDiT(
            dim,
            n_heads,
            mlp_ratio=mlp_ratio,
            attn_dropout=attn_dropout,
            mlp_dropout=mlp_dropout,
            qk_scale=qk_scale,
            kernel_size=3,
            padding=1,
        )
        self.visual_block = SingleModalStreamBlockDiT(
            dim,
            n_heads,
            mlp_ratio=mlp_ratio,
            attn_dropout=attn_dropout,
            mlp_dropout=mlp_dropout,
            qk_scale=qk_scale,
            kernel_size=3,
            padding=1,
        )
        self.text_block = SingleModalStreamBlockDiT(
            dim,
            n_heads,
            mlp_ratio=mlp_ratio,
            attn_dropout=attn_dropout,
            mlp_dropout=mlp_dropout,
            qk_scale=qk_scale,
            kernel_size=1,
        )

    def forward(
        self,
        audio: torch.Tensor,
        visual: torch.Tensor,
        text: torch.Tensor,
        c: torch.Tensor,
        audio_rot: torch.Tensor,
        visual_rot: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        x_qkv, x_mod = self.audio_block.pre_attention(audio, c, audio_rot)
        vis_qkv, vis_mod = self.visual_block.pre_attention(visual, c, visual_rot)
        text_qkv, text_mod = self.text_block.pre_attention(text, c, rot=None)

        x_len, vis_len, text_len = audio.size(1), visual.size(1), text.size(1)

        joint_qkv = [torch.cat([x_qkv[i], vis_qkv[i], text_qkv[i]], dim=2) for i in range(3)]

        if mask is not None and mask.ndim == 3:
            mask = repeat(mask, "b i j -> b h i j", h=self.n_heads)
        assert mask is None or mask.ndim == 4, "`mask` must be None or have 4 dimensions"

        attn_out = attention(
            q=joint_qkv[0],
            k=joint_qkv[1],
            v=joint_qkv[2],
            dropout=self.attn_dropout,
            qk_scale=self.qk_scale,
            mask=mask,
            training=self.training,
        )
        x_attn_out, vis_attn_out, text_attn_out = attn_out.split([x_len, vis_len, text_len], dim=1)

        x = self.audio_block.post_attention(audio, x_attn_out, x_mod)
        vis = self.visual_block.post_attention(visual, vis_attn_out, vis_mod)
        text = self.text_block.post_attention(text, text_attn_out, text_mod)

        return x, vis, text
