import os
import hydra
import rootutils
import torch
import multiprocessing as mp
from functools import partial
from omegaconf import DictConfig
from accelerate.logging import get_logger
from flowley.runner import Runner
from flowley.utils.configs import read_config, override_nested
from flowley.utils.ema import synthesize_ema
from flowley.utils.media_utils import VideoJoiner, save_audio

logger = get_logger(__name__)
joiner: VideoJoiner

rootutils.setup_root(__file__, indicator=".project_root", pythonpath=True)


def mkdir(cfg, save_type):
    run_name = os.path.basename(cfg.run_dir)
    save_root_dir = cfg.paths.output_video_dir if save_type == "video" \
        else cfg.paths.output_audio_dir
    if isinstance(cfg.cfg_strength, list):
        cfg_strength = f"{cfg.cfg_strength[0]}-{cfg.cfg_strength[1]}"
    else:
        cfg_strength = cfg.cfg_strength
    output_root_dir = os.path.join(
        save_root_dir,
        f"{run_name}_emaiter={cfg.min_step_ema}-{cfg.max_step_ema}_fmstep={cfg.flow_matching.num_steps}_cfg={cfg_strength}"
    )
    os.makedirs(output_root_dir, exist_ok=True)
    return output_root_dir


def init_worker(cfg):
    """
    This runs once in each worker process when it starts.
    It sets up the global `joiner` there, so each task can reuse it.
    """
    output_root_dir = mkdir(cfg, save_type="video")

    global joiner
    joiner = VideoJoiner(
        video_root=cfg.video_root_dir,
        output_root=output_root_dir,
        sample_rate=cfg.sample_rate,
        duration_secs=cfg.duration
    )


def join_worker(args):
    """
    Each args is just (video_id, audio_tensor).
    We rely on the already-initialized global `joiner`.
    """
    video_id, audio = args
    joiner.join(video_id, audio)


@hydra.main(version_base="1.3.2", config_path="../configs", config_name="sample.yaml")
def sample(cfg: DictConfig):
    cfg = override_nested(
        cfg, read_config(cfg.train_cfg_path), exclude_keys=["test_dataset", "num_steps"]
    )
    if cfg.max_step_ema is None:
        maxiter = max(int(fname.split('.')[1]) for fname in os.listdir(cfg.ema.checkpoint_folder))
        cfg.max_step_ema = maxiter
    else:
        maxiter = cfg.max_step_ema
    if cfg.min_step_ema is None:
        miniter = 0
        cfg.min_step_ema = 0
    else:
        miniter = cfg.min_step_ema

    runner = Runner(cfg, training=False)

    save_path = os.path.join(
        cfg.run_dir, f"ema_final_{miniter}-{maxiter}_step={cfg.ema.checkpoint_every}.pth"
    )
    if runner.is_main:
        if not os.path.exists(save_path) or cfg.force_synthesize:
            logger.info(f"Synthesizing EMA with sigma={cfg.ema.default_output_sigma}")
            ema_sigma = cfg.ema.default_output_sigma
            state_dict = synthesize_ema(cfg, ema_sigma)
            torch.save(state_dict, save_path)
            logger.info(f"Synthesized EMA saved to {save_path}")
        else:
            logger.info(f"Path {save_path} is existed. Skip synthesizing new EMA.")
    runner.accelerator.wait_for_everyone()

    runner.load_weights(save_path)
    logger.info("EMA weights loaded. Begin to sample...")
    ids, audios = runner.inference(vocode_gt=cfg.vocode_gt)

    # Save generated sounds
    if cfg.export_with_video:
        # Join audio to video and write to disk
        args = list(zip(ids, audios))
        with mp.Pool(processes=cfg.num_workers, initializer=init_worker, initargs=(cfg,)) as pool:
            pool.map(join_worker, args)
    else:
        # Write only audio to disk
        output_root_dir = mkdir(cfg, save_type="audio")
        filenames = [os.path.join(output_root_dir, ind) for ind in ids]

        save_fn = partial(save_audio, sr=cfg.sample_rate)
        args = list(zip(audios.permute(0, 2, 1), filenames))
        with mp.Pool(processes=cfg.num_workers) as pool:
            pool.starmap(save_fn, args)

    logger.info("Complete!!!")


if __name__ == "__main__":
    sample()
