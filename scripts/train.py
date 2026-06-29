import hydra
import rootutils
from pathlib import Path
from omegaconf import DictConfig
from accelerate.logging import get_logger
from flowley.runner import Runner
from flowley.utils.configs import read_config

logger = get_logger(__name__)

rootutils.setup_root(__file__, indicator=".project_root", pythonpath=True)


@hydra.main(version_base="1.3.2", config_path="../configs", config_name="train.yaml")
def train(cfg: DictConfig):

    if cfg.resume_from_ckpt is not None:
        print("Pretrained ckpt detected. Restore old config.")
        ckpt_path = cfg.resume_from_ckpt
        prev_cfg_file = Path(ckpt_path).parent.parent / "train_cfg.yaml"
        cfg = read_config(str(prev_cfg_file))
        cfg.resume_from_ckpt = ckpt_path

    runner = Runner(cfg)
    runner.train()


if __name__ == "__main__":
    train()
