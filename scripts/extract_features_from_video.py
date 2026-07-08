import os
import pandas as pd
import tensordict as td
import torch
import torch.distributed as distributed
from omegaconf import DictConfig
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from flowley.data.preprocessing import VATDataset
from flowley.model.preprocessors.feature_encoder import FeaturesEncoder
from flowley.utils.configs import parse_config_with_modified_args, write_config
from flowley.utils.dist_utils import local_rank, world_size
from flowley.utils.logger_utils import get_logger


logger = get_logger("./logs/extract_features_from_video.log")


def setup_distributed():
    distributed.init_process_group(backend='nccl')
    logger.info(f'Initialized: local_rank={local_rank}, world_size={world_size}')
    return local_rank, world_size


def setup_dataset(config: DictConfig, split: str):
    sr = 16000 if config.mode == '16k' else 44100
    dataset = VATDataset(
        data_file=config.tsv_file,
        split=split,
        duration_sec=config.duration_sec,
        clip_fps=config.clip_fps,
        clip_size=config.clip_size,
        sampling_rate=sr,
        normalize_audio=config.normalize_audio,
    )
    sampler = DistributedSampler(dataset, rank=local_rank, shuffle=False)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        sampler=sampler,
        pin_memory=True,
        drop_last=False,
    )
    return dataset, loader


@torch.inference_mode()
def extract(config: DictConfig, features_encoder: FeaturesEncoder, split: str):
    print(f"Starting split {split}")
    dataset, loader = setup_dataset(config, split)
    logger.info(f"Extracting features for the {split} split.")
    logger.info(f"Number of samples: {len(dataset)}.")
    logger.info(f"Number of batches: {len(loader)}.")

    tmp_dir = os.path.join(config.tmp_dir, split)

    for curr_iter, data in enumerate(tqdm(loader, total=len(loader), desc="Encoding")):
        output = {
            "path": data["path"],
            "caption": data["caption"],
        }
        # process audio with vae
        audio = data["audio"].to(local_rank)
        dist = features_encoder.encode_audio_with_vae(audio)
        output["mean"] = dist.mean.detach().cpu().transpose(1, 2)
        output["std"] = dist.std.detach().cpu().transpose(1, 2)

        # process video with clip
        clip_video = data["clip_video"].to(local_rank)
        clip_video_features = features_encoder.encode_video_with_clip(clip_video)
        output["video_features"] = clip_video_features.detach().cpu()

        # process text
        text = data["caption"]
        text_features, text_len = features_encoder.encode_text_with_flan_t5(
            text, max_length=config.max_len
        )
        output["text_features"] = text_features.detach().cpu()
        output["text_len"] = text_len.detach().cpu()

        torch.save(output, os.path.join(tmp_dir, f"rank={local_rank}-{curr_iter}.pth"))

    distributed.barrier()

    # Combine all outputs
    if local_rank == 0:
        logger.info(f"Combining outputs for the {split} split.")
        list_of_ids_and_labels = []
        output_data = {
            "mean": [],
            "std": [],
            "video_features": [],
            "text_features": [],
            "text_len": [],
        }

        for t in tqdm(sorted(os.listdir(tmp_dir))):
            data = torch.load(os.path.join(tmp_dir, t), weights_only=True)

            list_of_ids_and_labels.extend([
                {"path": path, "caption": caption} for path, caption in zip(data["path"], data["caption"])
            ])
            output_data["mean"].append(data["mean"])
            output_data["std"].append(data["std"])
            output_data["video_features"].append(data["video_features"])
            output_data["text_features"].append(data["text_features"])
            output_data["text_len"].append(data["text_len"])

        output_df = pd.DataFrame(list_of_ids_and_labels)
        output_df.to_csv(os.path.join(config.output_dir, f"{split}.tsv"), sep="\t", index=False)

        output_data = {k: torch.concat(v, dim=0) for k, v in output_data.items()}

        logger.info("Proceed to save TensorDict...")
        td.TensorDict(
            output_data,
            batch_size=len(list_of_ids_and_labels)
        ).memmap_(os.path.join(config.output_dir, f"{split}"))
        logger.info(f"Finish saving TensorDict for {split} split.")

        logger.info("Removing tmp_dir...")
        os.system(f"rm -rf {tmp_dir}")
        logger.info("Done!")


if __name__ == "__main__":
    # setup distributed
    setup_distributed()
    # torch.backends.cuda.matmul.allow_tf32 = True
    # torch.backends.cudnn.allow_tf32 = True

    # Init configuration
    config = parse_config_with_modified_args()
    os.makedirs(config.output_dir, exist_ok=True)
    write_config(config, config.output_dir)
    logger.info(f"Write config to {os.path.join(config.output_dir, 'config.yaml')}.")

    # Load modules
    features_encoder = FeaturesEncoder(
        vae_ckpt_path=config.vae_ckpt_path,
        mode=config.mode,
    ).eval().to(local_rank)

    os.makedirs(os.path.join(config.tmp_dir, config.split), exist_ok=True)
    extract(config, features_encoder, split=config.split)

    distributed.destroy_process_group()
