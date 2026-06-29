import math
import torch
import torch.nn as nn
from typing import Tuple
from torch import Tensor
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from .low_levels import RMSNorm
from .positional_embeddings import apply_rope


def attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    dropout: float = 0.0,
    qk_scale: float = None,
    mask: Tensor = None,
    training: bool = False,
) -> Tensor:
    """Wrapper around a basic attention operation
    Returns:
        Tensor: The output tensor with shape of (batch_size, q_len, hidden_dim)
    """
    dropout = dropout if training else 0.0
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    out = nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, dropout_p=dropout, is_causal=False, scale=qk_scale
    )
    out = rearrange(out, "b h n d -> b n (h d)").contiguous()
    return out


def compute_audio_visual_cross_attn_mask(
    audio_len: int,
    video_len: int,
    video_fps: float = 4,
    audio_fps: float = None,
    window_size: int = 1,
    fade: bool = False,
    fade_range: int = 1,
    fade_type: str = "linear",
    fade_scale: float = 1.0,
) -> Tensor:
    """Create a [audio_len, video_lean] attention mask for local cross-attention.
    Each audio index i can only attend to video frames in [j(i) - w, j(i) + w],
    where j(i) is derived from time alignment:
        t_i = i / audio_fps
        j(i) = round(t_i * video_fps)

    Then, for each audio token and video frame:
      - If |j - j_center| <= window_size: weight = 1 (fully attended).
      - If window_size < |j - j_center| <= window_size + fade_range:
          weight is computed using a fade profile on the offset (d - window_size),
          where d = |j - j_center|.
          - For "linear":   weight = 1 - ((d - window_size) / fade_range)
          - For "cosine":   weight = (cos(pi * (d - window_size) / fade_range) + 1) / 2
          - For "gaussian": weight = exp(-((d - window_size)^2) / (2 * sigma^2))
            with sigma defaulting to fade_range/2 if not provided.
      - If |j - j_center| > window_size + fade_range: weight = 0.

    Args:
        audio_len (int): The length of the audio sequence
        video_len (int): The length of the video sequence
        video_fps (float, optional): The frame rate of the video. Defaults to 4.
        audio_fps (float, optional): The frame rate of the audio. Defaults to 31.25.
        window_size (int): The size of the window for the local cross-attention
        fade (bool): Whether to apply a fade effect to the mask
        faed_range (int): The range of the fade effect
        fade_type (str): The type of the fade effect. Can be 'linear', 'cosine', or 'gaussian'

    Returns:
        Tensor: The attention mask tensor of shape (audio_len, video_len)
    """
    assert video_fps is not None, "video_fps must be provided"
    if audio_fps is None:
        audio_fps = audio_len / video_len * video_fps

    # Compute time for each audio token and determine its aligned video frame
    t_i = torch.arange(audio_len).float() / audio_fps
    j_center = torch.round(t_i * video_fps).long()
    j_center = torch.clamp(j_center, 0, video_len - 1)

    # Create a grid of video indices for each audio token
    j = torch.arange(video_len).unsqueeze(0).expand(audio_len, video_len).float()
    j_center = j_center.unsqueeze(1).expand(audio_len, video_len).float()

    # Compute the abs distance from the center for each pair (i, j)
    dist = torch.abs(j - j_center)

    if fade:
        # Full attention
        mask = torch.zeros_like(dist)
        mask[dist <= window_size] = 1.0

        # Fade region
        mask_fade = (dist > window_size) & (dist <= window_size + fade_range)
        if mask_fade.any():
            # Relative distance within the fade region
            delta = dist[mask_fade] - window_size
            if fade_type == "linear":
                fade_weights = 1 - delta / fade_range
            elif fade_type == "cosine":
                fade_weights = 0.5 * (1 + torch.cos(delta / fade_range * math.pi))
            elif fade_type == "gaussian":
                sigma = fade_range / 2
                fade_weights = torch.exp(-0.5 * (delta / sigma)**2)
            else:
                raise ValueError(f"Unknown supported type: {fade_type}.")

            fade_weights = fade_weights * fade_scale
            mask[mask_fade] = fade_weights  # Keep 0s for the rest
    else:
        mask = torch.where(dist <= window_size, torch.ones_like(dist), torch.zeros_like(dist))

    # Take log of the mask
    mask = (mask + 1e-6).log()

    return mask


def compute_multistream_mask(
    actual_text_lens: Tensor,
    audio_len: int,
    video_len: int,
    text_len: int,
):
    """Attention mask for multimodal stream block. Mask all <pad> tokens of text features.

    Args:
        audio_len (int): fixed length of the audio stream.
        video_len (int): fixed length of the video stream.
        text_len (int): padded length of the text stream.
        actual_text_lens (Tensor): actual lengths of each text sequence (B,)

    Returns:
        Tensor: attention mask of shape (B, T, T).
    """
    batch_size = actual_text_lens.size(0)
    seq_len = audio_len + video_len + text_len
    text_start = audio_len + video_len

    # Compute which positions are padded
    ids = torch.arange(seq_len, device=actual_text_lens.device)
    pad_mask = ids[None, :] >= (text_start + actual_text_lens[:, None])

    key_bias = torch.zeros(batch_size, seq_len, device=actual_text_lens.device)
    key_bias = key_bias.masked_fill(pad_mask, float("-inf"))

    return key_bias.unsqueeze(1).expand(-1, seq_len, -1)


class QKNorm(nn.Module):

    def __init__(self, dim: int, rms: bool = True):
        super(QKNorm, self).__init__()
        if rms:
            self.query_norm = RMSNorm(dim)
            self.key_norm = RMSNorm(dim)
        else:
            self.query_norm = nn.LayerNorm(dim)
            self.key_norm = nn.LayerNorm(dim)

    def forward(self, q: Tensor, k: Tensor) -> Tensor:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q, k


class CrossAttention(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        qk_norm: bool = True,
        qk_scale: float = None,
        rms: bool = True,
        dropout: float = 0.0,
    ):
        super(CrossAttention, self).__init__()
        self.dropout = dropout
        self.qk_scale = qk_scale
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        assert self.head_dim * n_heads == hidden_dim, "`hidden_dim` must be divisible by `n_heads`"

        self.q_prj = nn.Linear(hidden_dim, hidden_dim)
        self.kv_prj = nn.Linear(hidden_dim, hidden_dim * 2)
        self.split_q_into_heads = Rearrange(
            "b n (h d) -> b h n d",
            h=n_heads,
            d=self.head_dim,
        )
        self.split_kv_into_heads = Rearrange(
            "b n (h d j) -> b h n d j",
            h=n_heads,
            d=self.head_dim,
            j=2,
        )
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def pre_attention(self, input1: Tensor, input2: Tensor):
        """Prepare the inputs for the attention operation

        Args:
            input1 (Tensor): The first input tensor of shape (batch_size, seq1_len, input_dim)
            input2 (Tensor): The second input tensor of shape (batch_size, seq2_len, input_dim)

        Returns:
            Tuple[Tensor, Tensor, Tensor]: The query, key, and value tensors
        """
        q = self.q_prj(input1)
        q = self.split_q_into_heads(q)

        kv = self.kv_prj(input2)
        k, v = self.split_kv_into_heads(kv).chunk(2, dim=-1)
        k = k.squeeze(-1)
        v = v.squeeze(-1)

        q = self.q_norm(q)
        k = self.k_norm(k)

        return q, k, v

    def forward(
        self, input1: Tensor, input2: Tensor, mask: Tensor = None
    ) -> Tensor:
        """Forward pass of the cross-attention module

        Args:
            input1 (Tensor): The first input tensor of shape (batch_size, seq1_len, input_dim)
            input2 (Tensor): The second input tensor of shape (batch_size, seq2_len, input_dim)
            mask (Tensor, optional): The attention mask tensor. Defaults to None.

        Returns:
            Tensor: The output tensor of shape (batch_size, seq1_len, hidden_dim)
        """
        q, k, v = self.pre_attention(input1, input2)
        if mask is not None and mask.ndim == 2:
            mask = repeat(mask, "i j -> b i j", b=q.size(0))
        if mask is not None and mask.ndim == 3:
            mask = repeat(mask, "b i j -> b h i j", h=self.n_heads)
        assert mask is None or mask.ndim == 4, "`mask` must be None or have 4 dimensions"
        out = attention(
            q, k, v, self.dropout, self.qk_scale, mask=mask, training=self.training
        )
        return out


class SelfAttention(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        qk_norm: bool = True,
        qk_scale: float = None,
        rms: bool = True,
        dropout: float = 0.0,
    ):
        super(SelfAttention, self).__init__()
        self.dropout = dropout
        self.qk_scale = qk_scale
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        assert self.head_dim * n_heads == hidden_dim, "`hidden_dim` must be divisible by `n_heads`"

        self.qkv_prj = nn.Linear(hidden_dim, hidden_dim * 3)
        self.split_into_heads = Rearrange(
            "b n (h d j) -> b h n d j",
            h=n_heads,
            d=self.head_dim,
            j=3,
        )
        if qk_norm:
            self.qk_norm = QKNorm(self.head_dim, rms)
        else:
            self.qk_norm = nn.Identity()

    def pre_attention(
        self, x: Tensor, rot: Tensor | Tuple[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Prepare the inputs for the attention operation

        Args:
            x (Tensor): The input tensor of shape (batch_size, len, hidden_dim)

        Returns:
            Tuple[Tensor, Tensor, Tensor]: The query, key, and value tensors
        """
        qkv = self.qkv_prj(x)
        q, k, v = self.split_into_heads(qkv).chunk(3, dim=-1)
        q = q.squeeze(-1)
        k = k.squeeze(-1)
        v = v.squeeze(-1)

        q, k = self.qk_norm(q, k)

        # apply rotary embeddings
        if rot is not None:
            rot = (rot, rot) if not isinstance(rot, tuple) else rot
            q = apply_rope(q, rot[0])
            k = apply_rope(k, rot[1])

        return q, k, v

    def attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor = None) -> Tensor:
        return attention(
            q, k, v, self.dropout, self.qk_scale, mask=mask, training=self.training
        )

    def forward(self, x: Tensor, mask: Tensor = None, rot: Tensor = None) -> Tensor:
        """Forward pass of the cross-attention module

        Args:
            x (Tensor): The first input tensor of shape (batch_size, len, input_dim)
            mask (Tensor, optional): The attention mask tensor. Defaults to None.

        Returns:
            Tensor: The output tensor of shape (batch_size, len, hidden_dim)
        """
        q, k, v = self.pre_attention(x, rot)
        if mask is not None and mask.ndim == 3:
            mask = repeat(mask, "b i j -> b h i j", h=self.n_heads)
        assert mask is None or mask.ndim == 4, "`mask` must be None or have 4 dimensions"
        out = self.attention(q, k, v, mask=mask)
        return out
