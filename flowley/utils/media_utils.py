import torch
import torchaudio
import os
from torio.io import StreamingMediaDecoder, StreamingMediaEncoder


class VideoJoiner:

    def __init__(
        self,
        video_root: str,
        output_root: str,
        sample_rate: int = 16000,
        duration_secs: float = 8.,
    ):
        assert output_root != video_root, "output_root should be different from video_root."
        self.video_root = video_root
        self.output_root = output_root
        self.sample_rate = sample_rate
        self.duration_secs = duration_secs

        os.makedirs(self.output_root, exist_ok=True)

    def join(self, video_id: str, audio: torch.Tensor):
        video_path = os.path.join(self.video_root, f"{video_id}.mp4")
        output_path = os.path.join(self.output_root, f"{video_id}.mp4")
        add_audio_to_video(video_path, output_path, audio, self.sample_rate, self.duration_secs)


def add_audio_to_video(
    video_path: str,
    output_path: str,
    audio: torch.Tensor,
    sample_rate: int = 16000,
    duration_secs: float = 8.0
):
    frame_rate = 24
    reader = StreamingMediaDecoder(video_path)
    reader.add_basic_video_stream(
        frames_per_chunk=int(frame_rate * duration_secs),
        format="rgb24",
        frame_rate=frame_rate,
    )
    reader.fill_buffer()
    video_chunk = reader.pop_chunks()[0]
    _, _, h, w = video_chunk.shape

    writer = StreamingMediaEncoder(output_path)
    writer.add_audio_stream(
        sample_rate=sample_rate,
        num_channels=audio.shape[1],
        encoder="libmp3lame",
    )
    writer.add_video_stream(
        frame_rate=frame_rate,
        width=w,
        height=h,
        format="rgb24",
        encoder="libx264",
        encoder_format="yuv420p",
    )

    with writer.open():
        writer.write_audio_chunk(0, audio.float())
        writer.write_video_chunk(1, video_chunk)


def save_audio(audio: torch.Tensor, filename: str, sr: int):
    assert audio.ndim == 2 and audio.size(0) == 1, f"Audio must be 2D tensor but got {audio.shape}."
    torchaudio.save(f"{filename}.wav", audio.cpu().float(), sample_rate=sr)
