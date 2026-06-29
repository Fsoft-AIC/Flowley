import torch
import torch.nn as nn
from typing import Optional, Literal
from .constants import DATA_MEAN_80D, DATA_STD_80D, DATA_MEAN_128D, DATA_STD_128D
from .distributions import DiagonalGaussianDistribution
from .edm2_utils import MPConv1D
from .vae_modules import AttnBlock1D, Downsample1D, ResnetBlock1D, Upsample1D, nonlinearity


class AudioAutoEncoder(nn.Module):

    def __init__(self, *, vae_ckpt_path: str, mode: Literal['16k', '44k']):
        super(AudioAutoEncoder, self).__init__()

        assert mode in ['16k', '44k']
        self.vae = get_vae_model(mode)
        self.vae.load_state_dict(torch.load(vae_ckpt_path, weights_only=True, map_location='cpu'))
        self.vae.remove_weight_norm()

        self.freeze_vae()

    def freeze_vae(self):
        for param in self.vae.parameters():
            param.requires_grad = False

    @torch.inference_mode()
    def encode(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        return self.vae.encode(x)

    @torch.inference_mode()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(z)


class VAE(nn.Module):

    def __init__(
        self,
        *,
        data_dim: int,
        embed_dim: int,
        hidden_dim: int,
        ch_mult: tuple[int] = (1, 2, 4),
        num_res_blocks: int = 2,
        attn_layers: list[int] = [3],
        down_layers: list[int] = [0],
        kernel_size: int = 3,
    ):
        super(VAE, self).__init__()

        if data_dim == 80:
            self.register_buffer('data_mean', torch.tensor(DATA_MEAN_80D, dtype=torch.float32))
            self.register_buffer('data_std', torch.tensor(DATA_STD_80D, dtype=torch.float32))
        elif data_dim == 128:
            self.register_buffer('data_mean', torch.tensor(DATA_MEAN_128D, dtype=torch.float32))
            self.register_buffer('data_std', torch.tensor(DATA_STD_128D, dtype=torch.float32))

        self.data_mean = self.data_mean.view(1, -1, 1)
        self.data_std = self.data_std.view(1, -1, 1)

        self.encoder = Encoder1D(
            dim=hidden_dim,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            attn_layers=attn_layers,
            down_layers=down_layers,
            in_dim=data_dim,
            embed_dim=embed_dim,
            kernel_size=kernel_size
        )
        self.decoder = Decoder1D(
            dim=hidden_dim,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            attn_layers=attn_layers,
            down_layers=down_layers,
            in_dim=data_dim,
            out_dim=data_dim,
            embed_dim=embed_dim,
            kernel_size=kernel_size
        )

        self.embed_dim = embed_dim

    def encode(self, x: torch.Tensor, normalize: bool = True) -> DiagonalGaussianDistribution:
        if normalize:
            x = self.normalize(x)
        moments = self.encoder(x)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z: torch.Tensor, unnormalize: bool = True) -> torch.Tensor:
        dec = self.decoder(z)
        if unnormalize:
            dec = self.unnormalize(dec)
        return dec

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.data_mean) / self.data_std

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.data_std + self.data_mean

    @property
    def count_parameters(self) -> int:
        """Return the number of parameters in the model in millions."""
        return sum(p.numel() for p in self.parameters()) / 1e6

    def forward(
        self,
        x: torch.Tensor,
        sample_posterior: bool = True,
        rng: Optional[torch.Generator] = None,
        normalize: bool = True,
        unnormalize: bool = True,
    ) -> tuple[torch.Tensor, DiagonalGaussianDistribution]:

        posterior = self.encode(x, normalize=normalize)
        if sample_posterior:
            z = posterior.sample(rng)
        else:
            z = posterior.mode()
        dec = self.decode(z, unnormalize=unnormalize)
        return dec, posterior

    def load_weights(self, src_dict) -> None:
        self.load_state_dict(src_dict, strict=True)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def remove_weight_norm(self):
        for name, m in self.named_modules():
            if isinstance(m, MPConv1D):
                m.remove_weight_norm()
                print(f"Removed weight norm from {name}")
        return self


class Encoder1D(nn.Module):

    def __init__(self,
                 *,
                 dim: int,
                 ch_mult: tuple[int] = (1, 2, 4, 8),
                 num_res_blocks: int,
                 attn_layers: list[int] = [],
                 down_layers: list[int] = [],
                 resamp_with_conv: bool = True,
                 in_dim: int,
                 embed_dim: int,
                 double_z: bool = True,
                 kernel_size: int = 3,
                 clip_act: float = 256.0):
        super().__init__()
        self.dim = dim
        self.num_layers = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_dim
        self.clip_act = clip_act
        self.down_layers = down_layers
        self.attn_layers = attn_layers
        self.conv_in = MPConv1D(in_dim, self.dim, kernel_size=kernel_size)

        in_ch_mult = (1, ) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        # downsampling
        self.down = nn.ModuleList()
        for i_level in range(self.num_layers):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = dim * in_ch_mult[i_level]
            block_out = dim * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock1D(in_dim=block_in,
                                  out_dim=block_out,
                                  kernel_size=kernel_size,
                                  use_norm=True))
                block_in = block_out
                if i_level in attn_layers:
                    attn.append(AttnBlock1D(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level in down_layers:
                down.downsample = Downsample1D(block_in, resamp_with_conv)
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock1D(in_dim=block_in,
                                         out_dim=block_in,
                                         kernel_size=kernel_size,
                                         use_norm=True)
        self.mid.attn_1 = AttnBlock1D(block_in)
        self.mid.block_2 = ResnetBlock1D(in_dim=block_in,
                                         out_dim=block_in,
                                         kernel_size=kernel_size,
                                         use_norm=True)

        # end
        self.conv_out = MPConv1D(block_in,
                                 2 * embed_dim if double_z else embed_dim,
                                 kernel_size=kernel_size)

        self.learnable_gain = nn.Parameter(torch.zeros([]))

    def forward(self, x):

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_layers):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                h = h.clamp(-self.clip_act, self.clip_act)
                hs.append(h)
            if i_level in self.down_layers:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        h = h.clamp(-self.clip_act, self.clip_act)

        # end
        h = nonlinearity(h)
        h = self.conv_out(h, gain=(self.learnable_gain + 1))
        return h


class Decoder1D(nn.Module):

    def __init__(self,
                 *,
                 dim: int,
                 out_dim: int,
                 ch_mult: tuple[int] = (1, 2, 4, 8),
                 num_res_blocks: int,
                 attn_layers: list[int] = [],
                 down_layers: list[int] = [],
                 kernel_size: int = 3,
                 resamp_with_conv: bool = True,
                 in_dim: int,
                 embed_dim: int,
                 clip_act: float = 256.0):
        super().__init__()
        self.ch = dim
        self.num_layers = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_dim
        self.clip_act = clip_act
        self.down_layers = [i + 1 for i in down_layers]  # each downlayer add one

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = dim * ch_mult[self.num_layers - 1]

        # z to block_in
        self.conv_in = MPConv1D(embed_dim, block_in, kernel_size=kernel_size)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock1D(in_dim=block_in, out_dim=block_in, use_norm=True)
        self.mid.attn_1 = AttnBlock1D(block_in)
        self.mid.block_2 = ResnetBlock1D(in_dim=block_in, out_dim=block_in, use_norm=True)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_layers)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = dim * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock1D(in_dim=block_in, out_dim=block_out, use_norm=True))
                block_in = block_out
                if i_level in attn_layers:
                    attn.append(AttnBlock1D(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level in self.down_layers:
                up.upsample = Upsample1D(block_in, resamp_with_conv)
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.conv_out = MPConv1D(block_in, out_dim, kernel_size=kernel_size)
        self.learnable_gain = nn.Parameter(torch.zeros([]))

    def forward(self, z):
        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        h = h.clamp(-self.clip_act, self.clip_act)

        # upsampling
        for i_level in reversed(range(self.num_layers)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
                h = h.clamp(-self.clip_act, self.clip_act)
            if i_level in self.down_layers:
                h = self.up[i_level].upsample(h)

        h = nonlinearity(h)
        h = self.conv_out(h, gain=(self.learnable_gain + 1))
        return h


def VAE_16k(**kwargs) -> VAE:
    return VAE(data_dim=80, embed_dim=20, hidden_dim=384, **kwargs)


def VAE_44k(**kwargs) -> VAE:
    return VAE(data_dim=128, embed_dim=40, hidden_dim=512, **kwargs)


def get_vae_model(name: str, **kwargs) -> VAE:
    if name == '16k':
        return VAE_16k(**kwargs)
    elif name == '44k':
        return VAE_44k(**kwargs)
    else:
        raise ValueError(f"Unknown VAE model name: {name}")
