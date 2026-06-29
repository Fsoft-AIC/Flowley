import torch
import torch.nn as nn
from functools import partial
from torch import Tensor
from accelerate.logging import get_logger
from .modules.layers.positional_embeddings import compute_rope_rotations
from .modules.layers.attention import compute_audio_visual_cross_attn_mask, compute_multistream_mask
from .modules.layers.low_levels import (
    ConvMLP,
    MLP,
    CrossModalityProjector,
    ChannelLastConv1d,
    TimestepEmbedder
)
from .modules import (
    MultiModalStreamBlock,
    SingleModalStreamBlockDiT,
    SingleModalStreamBlockDiTV2,
    FinalBlock
)

logger = get_logger(__name__)


class Flowley(nn.Module):

    def __init__(
        self,
        audio_dim: int,
        text_dim: int,
        video_dim: int,
        hidden_dim: int,
        audio_seq_len: int,
        text_seq_len: int,
        video_fps: int,
        visual_seq_len: int,
        latent_mean: Tensor = None,
        latent_std: Tensor = None,
        multi_n_layers: int = 4,
        multi_n_heads: int = 4,
        single_n_layers: int = 8,
        single_n_heads: int = 4,
        single_type: str = "DiT",
        balance_hparam: float = None,
        mask_window_size: int = 1,
        mask_fade: bool = False,
        mask_fade_range: int = 1,
        mask_fade_type: str = "linear",
        mlp_ratio: float = 4.0,
        qk_scale: float = None,
        attn_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
        adaln_init_gaussian: bool = False,
    ):
        super(Flowley, self).__init__()
        # Initialize parameters
        self.audio_seq_len = audio_seq_len
        self.text_seq_len = text_seq_len
        self.visual_seq_len = visual_seq_len
        self.video_fps = video_fps
        self.hidden_dim = hidden_dim
        self.multi_n_heads = multi_n_heads
        self.single_n_heads = single_n_heads
        self.single_type = single_type
        self.single_n_layers = single_n_layers
        self.mask_window_size = mask_window_size
        self.mask_fade = mask_fade
        self.mask_fade_range = mask_fade_range
        self.mask_fade_type = mask_fade_type
        self.adaln_init_gaussian = adaln_init_gaussian
        self.audio_fps = video_fps * audio_seq_len / visual_seq_len

        # Initialize input projectors
        self.text_projector = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.SiLU(),
            MLP(hidden_dim, hidden_dim * 4),
        )
        self.visual_projector = nn.Sequential(
            nn.Linear(video_dim, hidden_dim),
            nn.SiLU(),
            ConvMLP(hidden_dim, hidden_dim * 4, kernel_size=3, padding=1),
        )
        self.audio_projector = nn.Sequential(
            ChannelLastConv1d(audio_dim, hidden_dim, kernel_size=7, padding=3),
            nn.SiLU(),
            ConvMLP(hidden_dim, hidden_dim * 4, kernel_size=7, padding=3),
        )

        # Conditional projection
        self.text_cond_projector = nn.Linear(hidden_dim, hidden_dim)
        self.visual_cond_projector = nn.Linear(hidden_dim, hidden_dim)
        self.global_cond_projector = MLP(hidden_dim, hidden_dim * 4, dropout=mlp_dropout)
        # TODO: think about 10000
        self.timestep_embedder = TimestepEmbedder(hidden_dim, frequency_embed_size=256)

        # Blocks
        self.multimodal_mask_computer = partial(
            compute_multistream_mask,
            audio_len=audio_seq_len,
            video_len=visual_seq_len,
            text_len=text_seq_len
        )
        self.multimodal_stream = nn.ModuleList([
            MultiModalStreamBlock(
                hidden_dim,
                n_heads=multi_n_heads,
                mlp_ratio=mlp_ratio,
                attn_dropout=attn_dropout,
                mlp_dropout=mlp_dropout,
                qk_scale=qk_scale,
                pre_only=(i == multi_n_layers - 1),
            )
            for i in range(multi_n_layers)
        ])

        assert single_type in ["DiT", "DiTV2"]
        if single_type == "DiT":
            single_module = SingleModalStreamBlockDiT(
                hidden_dim,
                n_heads=single_n_heads,
                mlp_ratio=mlp_ratio,
                attn_dropout=attn_dropout,
                mlp_dropout=mlp_dropout,
                qk_scale=qk_scale,
                kernel_size=3,
                padding=1
            )
        else:
            self.cross_visual_proj = CrossModalityProjector(hidden_dim, hidden_dim, mlp_dropout)
            self.cross_textual_proj = CrossModalityProjector(hidden_dim, hidden_dim, mlp_dropout)
            single_module = SingleModalStreamBlockDiTV2(
                hidden_dim,
                n_heads=single_n_heads,
                mlp_ratio=mlp_ratio,
                attn_dropout=attn_dropout,
                mlp_dropout=mlp_dropout,
                qk_scale=qk_scale,
                kernel_size=3,
                padding=1,
                balance_hparam=balance_hparam,
            )
        self.single_modal_stream = nn.ModuleList([
            single_module for _ in range(single_n_layers)
        ])
        self.final_layer = FinalBlock(hidden_dim, audio_dim)

        # Check if latent_mean and latent_std are provided
        if latent_mean is None:
            assert latent_std is None
            logger.debug(
                "latent_mean and latent_std are not provided. They will be initialized as NaNs "
                "and are not meant to be used. If this is the mistake, call `compute_latent_stats` "
                "method of the dataset. Else, you should load them later from a checkpoint."
            )
            latent_mean = torch.ones(audio_dim).view(1, 1, -1).fill_(float("nan"))
            latent_std = torch.ones(audio_dim).view(1, 1, -1).fill_(float("nan"))
        else:
            assert latent_mean is not None and latent_std is not None, \
                f"latent_mean and latent_std must be provided. Got {latent_mean} and {latent_std}. " \
                "You're supposed to load them by calling `compute_latent_stats` method of the dataset."

        self.latent_mean = nn.Parameter(latent_mean.view(1, 1, -1), requires_grad=False)
        self.latent_std = nn.Parameter(latent_std.view(1, 1, -1), requires_grad=False)

        # Empty tensors
        self.empty_visual_feature = nn.Parameter(torch.zeros(1, video_dim), requires_grad=False)
        self.empty_text_feature = nn.Parameter(torch.zeros(1, text_dim), requires_grad=False)

        self.initialize_rotations()
        self.initialize_weights()
        self.initialize_attention_mask()

    @property
    def device(self):
        return next(self.parameters()).device

    def count_parameters(self, return_trainable: bool = True):
        total_params = 0
        trainable_params = 0
        for p in self.parameters():
            total_params += p.numel()
            trainable_params += p.numel() if p.requires_grad else 0

        if return_trainable:
            return total_params, trainable_params

        return total_params

    def initialize_attention_mask(self):

        if self.single_type == 'DiTV2' and self.mask_window_size >= 0:
            single_modal_mask = []
            depth = self.single_n_layers
            for i in range(depth):
                scale = 1.0 if depth == 1 else 1.0 - (i / (depth - 1))
                single_modal_mask.append(
                    compute_audio_visual_cross_attn_mask(
                        self.audio_seq_len,
                        self.visual_seq_len,
                        video_fps=self.video_fps,
                        audio_fps=self.audio_fps,
                        window_size=self.mask_window_size,
                        fade=self.mask_fade,
                        fade_range=self.mask_fade_range,
                        fade_type=self.mask_fade_type,
                        fade_scale=scale,
                    )
                )

            single_modal_mask = torch.stack(single_modal_mask, dim=0)
            self.register_buffer("single_modal_mask", single_modal_mask, persistent=False)
        else:
            self.single_modal_mask = [None] * self.single_n_layers

    def initialize_rotations(self):
        base_freq = 1.0
        multi_audio_rot = compute_rope_rotations(
            self.audio_seq_len,
            dim=self.hidden_dim // self.multi_n_heads,
            theta=10000,
            freq_scaling=base_freq,
            device=self.device
        )
        single_audio_rot = compute_rope_rotations(
            self.audio_seq_len,
            dim=self.hidden_dim // self.single_n_heads,
            theta=10000,
            freq_scaling=base_freq,
            device=self.device
        )
        multi_visual_rot = compute_rope_rotations(
            self.visual_seq_len,
            dim=self.hidden_dim // self.multi_n_heads,
            theta=10000,
            freq_scaling=base_freq * self.audio_seq_len / self.visual_seq_len,
            device=self.device
        )
        self.register_buffer("multi_audio_rot", multi_audio_rot, persistent=False)
        self.register_buffer("single_audio_rot", single_audio_rot, persistent=False)
        self.register_buffer("multi_visual_rot", multi_visual_rot, persistent=False)

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.timestep_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.timestep_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.multimodal_stream:
            nn.init.constant_(block.audio_block.adaLN_modulation.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.visual_block.adaLN_modulation.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.text_block.adaLN_modulation.adaLN_modulation[-1].bias, 0)
            if self.adaln_init_gaussian:
                nn.init.normal_(block.audio_block.adaLN_modulation.adaLN_modulation[-1].weight, std=0.001)
                nn.init.normal_(block.visual_block.adaLN_modulation.adaLN_modulation[-1].weight, std=0.001)
                nn.init.normal_(block.text_block.adaLN_modulation.adaLN_modulation[-1].weight, std=0.001)
            else:
                nn.init.constant_(block.audio_block.adaLN_modulation.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.visual_block.adaLN_modulation.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.text_block.adaLN_modulation.adaLN_modulation[-1].weight, 0)
        for block in self.single_modal_stream:
            if self.adaln_init_gaussian:
                nn.init.normal_(block.adaLN_modulation.adaLN_modulation[-1].weight, std=0.001)
            else:
                nn.init.constant_(block.adaLN_modulation.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.conv.weight, 0)
        nn.init.constant_(self.final_layer.conv.bias, 0)

        # empty string feat shall be initialized by a CLIP encoder
        nn.init.constant_(self.empty_text_feature, 0)
        nn.init.constant_(self.empty_visual_feature, 0)

    def normalize(self, x: Tensor) -> Tensor:
        return x.sub_(self.latent_mean).div_(self.latent_std)

    def denormalize(self, x: Tensor) -> Tensor:
        return x.mul_(self.latent_std).add_(self.latent_mean)

    def get_empty_conditions(self, bs: int):
        empty_visual_feat = self.empty_visual_feature.unsqueeze(0)
        empty_visual_feat = empty_visual_feat.expand(
            bs, self.visual_seq_len, -1
        )
        empty_text_feat = self.empty_text_feature.unsqueeze(0)
        empty_text_feat = empty_text_feat.expand(bs, self.text_seq_len, -1)

        return empty_visual_feat, empty_text_feat

    def preprocess_inputs(self, audio_latent: Tensor, text_feature: Tensor, visual_feature: Tensor):
        """Preprocess inputs

        Args:
            audio_latent (Tensor): The audio latent tensor of shape (batch_size, mel_len, audio_dim)
            text_feature (Tensor): The text feature tensor of shape (batch_size, seq_len, text_dim)
            visual_feature (Tensor): The visual feature tensor of shape
                                     (batch_size, frames, visual_dim)

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Preprocessed audio, text and visual features
                latent: (batch_size, mel_len, hidden_dim)
                text_feat: (batch_size, seq_len, hidden_dim)
                visual_feat: (batch_size, frames, hidden_dim)
        """
        # Projection
        latent = self.audio_projector(audio_latent)
        text_feat = self.text_projector(text_feature)
        visual_feat = self.visual_projector(visual_feature)
        return latent, text_feat, visual_feat

    def preprocess_conditions(self, text_feat: Tensor, visual_feat: Tensor, t: Tensor) -> Tensor:
        """Preprocess conditions

        Args:
            text_feat (Tensor): The text feature tensor of shape (batch_size, seq_len, hidden_dim)
            visual_feat (Tensor): The visual feature tensor of shape
                                 (batch_size, frames, hidden_dim)
            t (Tensor): The timestep tensor of shape (batch_size,)

        Returns:
            Tensor: The preprocessed conditions of shape (batch_size, hidden_dim)
        """
        # Conditional projection
        cond_text_feat = self.text_cond_projector(text_feat.mean(1))
        cond_visual_feat = self.visual_cond_projector(visual_feat.mean(1))
        cond_feat = self.global_cond_projector(cond_text_feat + cond_visual_feat)   # (B, D)

        # Timestep embedding
        cond_feat = cond_feat + self.timestep_embedder(t)

        return cond_feat

    def predict_flow(
        self,
        latent: Tensor,
        text_feat: Tensor,
        visual_feat: Tensor,
        cond_feat: Tensor,
        text_len: Tensor,
    ):
        multi_modal_mask = self.multimodal_mask_computer(actual_text_lens=text_len)
        for layer in self.multimodal_stream:
            latent, visual_feat, text_feat = layer(
                latent,
                visual_feat,
                text_feat,
                cond_feat,
                self.multi_audio_rot,
                self.multi_visual_rot,
                multi_modal_mask
            )

        if self.single_type == "DiTV2":
            visual_feat = self.cross_visual_proj(visual_feat)
            text_feat = self.cross_textual_proj(text_feat)
            for i, layer in enumerate(self.single_modal_stream):
                latent = layer(
                    latent,
                    visual_feat,
                    text_feat,
                    cond_feat,
                    self.single_audio_rot,
                    self.single_modal_mask[i]
                )
        else:
            for i, layer in enumerate(self.single_modal_stream):
                latent = layer(latent, cond_feat, self.single_audio_rot, self.single_modal_mask[i])

        flow = self.final_layer(latent, cond_feat)

        return flow

    def forward(
        self,
        audio_latent: Tensor,
        text_feature: Tensor,
        visual_feature: Tensor,
        text_len: Tensor,
        t: Tensor
    ):
        """Forward pass of Flowley

        Args:
            audio_latent (Tensor): The audio latent tensor of shape (batch_size, mel_len, audio_dim)
            text_feature (Tensor): The text feature tensor of shape (batch_size, seq_len, text_dim)
            visual_feature (Tensor): The visual feature tensor of shape
                                     (batch_size, frames, visual_dim)
            text_len (Tensor): Actual lengths of every text sequences of shape (batch_size,)
            t (Tensor): The timestep tensor of shape (batch_size,)

        Returns:
            Tensor: flow with shape of (batch_size, mel_len, audio_dim)
            Tuple[Tensor, Tensor]: multi_stream_outputs (latent, visual)
        """
        latent, text_feat, visual_feat = self.preprocess_inputs(audio_latent,
                                                                text_feature,
                                                                visual_feature)
        cond_feat = self.preprocess_conditions(text_feat, visual_feat, t)
        flow = self.predict_flow(latent, text_feat, visual_feat, cond_feat, text_len)
        return flow, visual_feat
