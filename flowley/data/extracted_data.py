import os
import torch
import pandas as pd
from typing import Tuple
from tensordict import TensorDict
from torch.utils.data import Dataset
from accelerate.logging import get_logger


logger = get_logger(__name__)


class ExtractedVATDataset(Dataset):

    def __init__(
        self,
        tsv_file: str,
        memmap_dir: str,
        generator: torch.Generator,
        debug: bool = False,
        n_samples_for_debug: int = 100,
        output_dir: str = None,
    ):
        super(ExtractedVATDataset, self).__init__()
        self.split = os.path.basename(memmap_dir)
        assert self.split in ["train", "valid", "test"]

        self.df = pd.read_csv(tsv_file, sep='\t')
        if self.split == "train" and debug:
            n_samples_for_debug = min(n_samples_for_debug, len(self.df))
            selected_ids = torch.multinomial(
                torch.ones(len(self.df), device=generator.device),
                n_samples_for_debug,
                replacement=False,
                generator=generator,
            ).cpu().tolist()
            # Get selected training samples
            self.df = self.df.iloc[selected_ids]
            logger.info(f"Debug mode: {n_samples_for_debug} samples are selected.")
            # Save training set
            save_path = os.path.join(output_dir, "train_debug.tsv")
            self.df.to_csv(save_path, sep="\t", index=False)
            logger.info(f"Debugged samples are saved at {save_path}.")

        logger.info(f"Loading precomputed features from {memmap_dir}.")
        td = TensorDict.load_memmap(memmap_dir)
        self.mean = td["mean"]
        self.std = td["std"]
        self.video_features = td["video_features"]
        self.text_features = td["text_features"]
        self.text_len = td["text_len"]
        self.label = td["label"]
        if self.split == "train" and debug:
            self.mean = self.mean[selected_ids]
            self.std = self.std[selected_ids]
            self.video_features = self.video_features[selected_ids]
            self.text_features = self.text_features[selected_ids]
            self.text_len = self.text_len[selected_ids]
            self.label = self.label[selected_ids]

    def compute_latent_stats(self) -> Tuple[torch.Tensor, torch.Tensor]:
        latents = self.mean
        return latents.mean(dim=(0, 1)), latents.std(dim=(0, 1))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        id = os.path.splitext(os.path.basename(row["path"]))[0]
        data = {
            "path": row["path"],
            "id": id,
            "caption": row["caption"],
            "audio_mean": self.mean[idx],
            "audio_std": self.std[idx],
            "video_features": self.video_features[idx],
            "text_features": self.text_features[idx],
            "text_len": self.text_len[idx],
            "label": self.label[idx],
        }
        return data
