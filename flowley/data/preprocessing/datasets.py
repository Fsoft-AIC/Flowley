import torch
import pandas as pd
import torchaudio
from torch import Tensor
from torio.io import StreamingMediaDecoder
from torchvision.transforms import v2
from torch.utils.data import Dataset


class VATDataset(Dataset):

    def __init__(
        self,
        data_file: str,
        split: str = 'train',
        duration_sec: float = 8.0,
        clip_fps: int = 8,
        clip_size: int = 384,
        sampling_rate: int = 16_000,
        normalize_audio: bool = False,
    ):
        assert split in ['train', 'valid', 'test']

        data = pd.read_csv(data_file, sep='\t')
        self.data = data[data['split'] == split].reset_index(drop=True)

        self.normalize_audio = normalize_audio
        self.sr = sampling_rate
        self.clip_size = clip_size
        self.clip_fps = clip_fps
        self.duration_sec = duration_sec

        self.audio_length = int(sampling_rate * duration_sec)
        self.clip_length = int(clip_fps * duration_sec)
        self.clip_transform = v2.Compose([
            v2.Resize((clip_size, clip_size), interpolation=v2.InterpolationMode.BICUBIC),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])
        self.audio_resampler = {}

    def sample(self, idx: int) -> dict[str, Tensor]:
        row = self.data.loc[idx]
        video_path = row['video_path']
        text = row['text_description']
        label = row["label"]

        reader = StreamingMediaDecoder(video_path)
        reader.add_basic_video_stream(
            frames_per_chunk=int(self.clip_fps * self.duration_sec),
            frame_rate=self.clip_fps,
            format='rgb24',
        )
        reader.add_basic_audio_stream(frames_per_chunk=2**30)

        reader.fill_buffer()
        clip_chunk, audio_chunk = reader.pop_chunks()

        if clip_chunk is None:
            raise ValueError(f'No video frames found in {video_path}.')
        if clip_chunk.shape[0] < self.clip_length:
            raise ValueError(f'Video {video_path} is too short, expected {self.clip_length}, '
                             f'got {clip_chunk.shape[0]}.')

        # Process audio
        sample_rate = int(reader.get_out_stream_info(2).sample_rate)
        audio_chunk = self.process_audio(audio_chunk, sample_rate, video_path)

        # Process video
        clip_chunk = self.process_video(clip_chunk, video_path, chunk_type="clip")

        data_dict = {
            "path": video_path,
            "caption": text,
            "label": label,
            "audio": audio_chunk,
            "clip_video": clip_chunk,
        }
        return data_dict

    def process_audio(self, audio_chunk: Tensor, sample_rate: int, video_path: str) -> Tensor:
        audio_chunk = audio_chunk.transpose(0, 1)
        audio_chunk = audio_chunk.mean(dim=0)   # mono
        if self.normalize_audio:
            abs_max = audio_chunk.abs().max()
            audio_chunk = audio_chunk / abs_max * 0.95
            if abs_max <= 1e-6:
                raise RuntimeError(f'Audio in {video_path} is silent.')

        # Resample
        if sample_rate != self.sr:
            if sample_rate not in self.audio_resampler:
                self.audio_resampler[sample_rate] = torchaudio.transforms.Resample(
                    sample_rate,
                    self.sr,
                    lowpass_filter_width=64,
                    rolloff=0.9475937167399596,
                    resampling_method='sinc_interp_kaiser',
                    beta=14.769656459379492,
                )
            audio_chunk = self.audio_resampler[sample_rate](audio_chunk)

        if audio_chunk is None:
            raise ValueError(f'No audio found in {video_path}.')
        if audio_chunk.shape[0] < self.audio_length:
            raise ValueError(f'Audio {video_path} is too short, expected {self.audio_length}, '
                             f'got {audio_chunk.shape[0]}.')

        audio_chunk = audio_chunk[:self.audio_length]
        return audio_chunk

    def process_video(self, chunk: Tensor, video_path: str, chunk_type: str = "clip") -> Tensor:
        transform_fn = self.clip_transform if chunk_type == "clip" else self.sync_transform
        length = self.clip_length if chunk_type == "clip" else self.sync_length

        chunk = chunk[:length]
        if chunk.shape[0] != length:
            raise RuntimeError(f'CLIP video wrong length {video_path}, '
                               f'expected {length}, got {chunk.shape[0]}.')

        chunk = transform_fn(chunk)
        return chunk

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.sample(idx)
