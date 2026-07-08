import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from open_clip import create_model_from_pretrained, get_tokenizer
from transformers import AutoTokenizer, T5EncoderModel
from torchvision.transforms import Normalize
from flowley.model.preprocessors.vae import AudioAutoEncoder
from flowley.model.preprocessors.vae.distributions import DiagonalGaussianDistribution
from flowley.data.preprocessing.audio import get_mel_converter


def patch_clip(clip_model):
    # a hack to make it output last hidden states
    # https://github.com/mlfoundations/open_clip/blob/fc5a37b72d705f760ebbc7915b84729816ed471f/src/open_clip/model.py#L269
    def new_encode_text(self, text, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        return F.normalize(x, dim=-1) if normalize else x

    clip_model.encode_text = new_encode_text.__get__(clip_model)
    return clip_model


def pad_or_truncate(
    audio: Tensor,
    max_spec_t: int,
    pad_mode: str = "constant",
    pad_value: float = 0.0
):
    """Copied from Synchformer"""
    difference = max_spec_t - audio.shape[-1]  # safe for batched input
    # pad or truncate, depending on difference
    if difference > 0:
        # pad the last dim (time) -> (..., n_mels, 0+time+difference)  # safe for batched input
        pad_dims = (0, difference)
        audio = torch.nn.functional.pad(audio, pad_dims, pad_mode, pad_value)
    elif difference < 0:
        audio = audio[..., :max_spec_t]  # safe for batched input
    return audio


class FeaturesEncoder(nn.Module):

    def __init__(
        self,
        *,
        vae_ckpt_path: str = None,
        mode: str = "16k"
    ):
        super(FeaturesEncoder, self).__init__()

        clip_pretrained_path = "hf-hub:apple/DFN2B-CLIP-ViT-L-14"
        clip_tokenizer_path = "ViT-L-14"
        # Load CLIP model
        self.clip_model = create_model_from_pretrained(clip_pretrained_path,
                                                       return_transform=False)
        self.clip_video_process = Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                            std=[0.26862954, 0.26130258, 0.27577711])
        self.clip_model = patch_clip(self.clip_model)
        self.tokenizer = get_tokenizer(clip_tokenizer_path)

        # Load Flan T5 model
        self.flan_t5_model = T5EncoderModel.from_pretrained("google/flan-t5-large")
        self.flan_t5_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

        # Load VAE Model
        self.mel_converter = get_mel_converter(mode=mode)
        self.vae = AudioAutoEncoder(vae_ckpt_path=vae_ckpt_path, mode=mode)

    def compile(self):
        self.clip_model.encode_image = torch.compile(self.clip_model.encode_image)
        self.clip_model.encode_text = torch.compile(self.clip_model.encode_text)
        self.vae.encode = torch.compile(self.vae.encode)
        self.synchformer = torch.compile(self.synchformer)

    @torch.inference_mode()
    def encode_text_with_clip(self, text: str) -> Tensor:
        tokens = self.tokenizer(text).to(self.device)
        return self.clip_model.encode_text(tokens, normalize=True)

    @torch.inference_mode()
    def encode_text_with_flan_t5(
        self,
        text: str,
        max_length: int = 77,
        return_length: bool = True,
    ) -> Tensor:
        """Encode text using Flan-T5 model and return embeddings

        Args:
            text (str): Input text to encode
            max_length (int): Maximum length of the input text
            return_length (bool): Whether to return the length of the input text
            pooling (str): Pooling strategy - 'mean', 'first', or 'last'

        Returns:
            Tensor: Text embedding
        """
        # Tokenize input text
        inputs = self.flan_t5_tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",  # True: Pad to the longest sequence in the batch
            truncation=True,
            max_length=max_length
        ).to(self.device)

        # Get the original length of the input text
        original_length = torch.sum(inputs.attention_mask, dim=1)

        # Get encoder outputs from the model
        outputs = self.flan_t5_model(**inputs)

        # Get the last hidden states (embeddings)
        hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

        # Normalize the embedding
        embedding = F.normalize(hidden_states, p=2, dim=1)

        if return_length:
            return embedding, original_length
        else:
            return embedding

    @torch.inference_mode()
    def encode_audio_with_vae(self, x: Tensor) -> DiagonalGaussianDistribution:
        """Encode mel-spectrogram to latent space

        Args:
            x (Tensor): Audio waveform of shape `[B x L]`
        """
        mel = self.mel_fn(x)
        dist = self.vae.encode(mel)
        return dist

    @torch.inference_mode()
    def encode_video_with_clip(self, x: Tensor, batch_size: int = -1) -> Tensor:
        b, t, c, h, w = x.shape
        assert c == 3 and h == 224 and w == 224, f"Expected (b, t, 3, 224, 224), got {x.shape}"
        x = self.clip_video_process(x)
        x = rearrange(x, "b t c h w -> (b t) c h w")

        outputs = []
        if batch_size < 0:
            batch_size = b * t
        for i in range(0, b * t, batch_size):
            output = self.clip_model.encode_image(x[i:i + batch_size], normalize=True)
            outputs.append(output)
        x = torch.cat(outputs, dim=0)
        x = rearrange(x, "(b t) d -> b t d", b=b)
        return x

    @torch.inference_mode()
    def decode_audio(self, z: Tensor) -> Tensor:
        return self.vae.decode(z)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
