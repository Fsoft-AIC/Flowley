import torch
import torchaudio
import torch.nn as nn
from typing import Literal
from librosa.filters import mel as librosa_mel_fn


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5, norm_fn=torch.log10):
    return norm_fn(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes, norm_fn):
    output = dynamic_range_compression_torch(magnitudes, norm_fn=norm_fn)
    return output


def read_audio(path: str, sampling_rate: int = 16_000) -> torch.Tensor:
    """Read an audio file and return the waveform.

    Args:
        path (str): Path to the audio file.
        sampling_rate (int): Sampling rate.

    Returns:
        torch.Tensor: The waveform of shape (num_samples,).
    """
    waveform, sr = torchaudio.load(path)
    # Convert to mono if needed
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    # Resample if needed
    if sr != sampling_rate:
        resampler = torchaudio.transforms.Resample(
            sr,
            sampling_rate,
            resampling_method='sinc_interp_kaiser',
            lowpass_filter_width=64,
            rolloff=0.9475937167399596,
            beta=14.769656459379492
        )
        waveform = resampler(waveform)
    return waveform


class MelConverter(nn.Module):

    def __init__(
        self,
        *,
        sampling_rate: float = 16_000,
        n_fft: int = 1024,
        num_mels: int = 80,
        hop_size: int = 256,
        win_size: int = 1024,
        fmin: float = 0,
        fmax: float = 8_000,
        norm_fn=torch.log10,
    ):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.norm_fn = norm_fn

        mel = librosa_mel_fn(sr=self.sampling_rate,
                             n_fft=self.n_fft,
                             n_mels=self.num_mels,
                             fmin=self.fmin,
                             fmax=self.fmax)
        mel_basis = torch.from_numpy(mel).float()
        hann_window = torch.hann_window(self.win_size)

        self.register_buffer('mel_basis', mel_basis)
        self.register_buffer('hann_window', hann_window)

    @property
    def device(self):
        return self.mel_basis.device

    def forward(self, waveform: torch.Tensor, center: bool = False) -> torch.Tensor:
        waveform = waveform.clamp(min=-1., max=1.).to(self.device)

        waveform = torch.nn.functional.pad(
            waveform.unsqueeze(1),
            [int((self.n_fft - self.hop_size) / 2),
             int((self.n_fft - self.hop_size) / 2)],
            mode='reflect')
        waveform = waveform.squeeze(1)

        spec = torch.stft(waveform,
                          self.n_fft,
                          hop_length=self.hop_size,
                          win_length=self.win_size,
                          window=self.hann_window,
                          center=center,
                          pad_mode='reflect',
                          normalized=False,
                          onesided=True,
                          return_complex=True)

        spec = torch.view_as_real(spec)
        spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))
        spec = torch.matmul(self.mel_basis, spec)
        spec = spectral_normalize_torch(spec, self.norm_fn)

        return spec


def get_mel_converter(mode: Literal["16k", "44k"]) -> MelConverter:
    if mode == "16k":
        return MelConverter(
            sampling_rate=16_000,
            n_fft=1024,
            num_mels=80,
            hop_size=256,
            win_size=1024,
            fmin=0,
            fmax=8_000,
            norm_fn=torch.log10,
        )
    elif mode == "44k":
        return MelConverter(
            sampling_rate=44_100,
            n_fft=2048,
            num_mels=128,
            hop_size=512,
            win_size=2048,
            fmin=0,
            fmax=44100 / 2,
            norm_fn=torch.log,
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")
