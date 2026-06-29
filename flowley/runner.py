import os
import re
import math
import torch
import wandb
import torch.optim as optim
from datetime import timedelta
from typing import List
from torch.utils.data import DataLoader
from accelerate import Accelerator, DistributedDataParallelKwargs, InitProcessGroupKwargs
from accelerate.utils import tqdm
from accelerate.logging import get_logger
from omegaconf import DictConfig, OmegaConf
from .utils.ema import PostHocEMA
from .utils.logger_utils import mels_to_image
from .model.losses import LossBreakdown
from .model.flow_matching import RectifiedFlowMatching
from .model.model import Flowley
from .data.extracted_data import ExtractedVATDataset
from .vocoder import BigVGAN
from .model.preprocessors.vae import AudioAutoEncoder


logger = get_logger(__name__)


class Runner:

    def __init__(self, cfg: DictConfig, training: bool = True):

        self.training = training
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        process_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))
        self.accelerator = Accelerator(
            kwargs_handlers=[ddp_kwargs, process_kwargs],
            **cfg.accelerator
        )
        self.rng = torch.Generator(self.accelerator.device).manual_seed(cfg.seed)

        if cfg.disable_wandb:
            if not self.training:
                assert cfg.wandb.id is not None, "WandB ID must be provided for inference."
            self.accelerator.init_trackers(
                project_name="random",  # flowley
                config=OmegaConf.to_container(cfg, resolve=True),
                init_kwargs={
                    "wandb": {
                        "id": cfg.wandb.id,
                        "resume": "allow",
                        "dir": cfg.wandb.dir,
                        "group": cfg.wandb.group,
                        "name": cfg.wandb.name,
                        "settings": wandb.Settings(_disable_stats=True, _disable_meta=True)
                    }
                }
            )
            cfg.wandb.id = self.define_wandb_metrics()

        # Save config to yaml
        self.save_config(cfg)

        # Setup dataloader
        self.dataloader = self.setup_dataloader(cfg)
        if self.training:
            num_iters = math.ceil(len(self.dataloader[0]) // self.accelerator.num_processes)
            logger.info(f"Number of iterations per epoch: {num_iters}")

        # Initialize models
        self.latent_mean, self.latent_std = self.compute_latent_stats()
        self.vocoder = BigVGAN(**cfg.vocoder)
        self.vae = AudioAutoEncoder(**cfg.vae)
        model = Flowley(latent_mean=self.latent_mean, latent_std=self.latent_std, **cfg.model)
        self.rfm = RectifiedFlowMatching(model=model, **cfg.flow_matching)

        # Calculate total number of parameters
        total_params, trainable_params = model.count_parameters(return_trainable=True)
        logger.info(f"Total parameters: {total_params / 1e6:.2f}M")
        logger.info(f"Total trainable parameters: {trainable_params / 1e6:.2f}M")

        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.rfm.parameters(),
            lr=cfg.optim.lr,
            weight_decay=cfg.optim.weight_decay,
            betas=cfg.optim.betas,
        )

        # Initialize scheduler
        self.scheduler = self.setup_scheduler(cfg.scheduler)

        # Initialize EMA (if not using consistency FM)
        self.use_ema = cfg.use_ema
        logger.info(f"use_ema: {self.use_ema}")

        if self.is_main and self.use_ema:
            self.ema_model = PostHocEMA(
                self.rfm,
                sigma_rels=cfg.ema.sigma_rels,
                update_every=cfg.ema.update_every,
                checkpoint_every_num_steps=cfg.ema.checkpoint_every,
                checkpoint_folder=cfg.ema.checkpoint_folder,
                step_size_correction=True,
                allow_different_devices=True,
            )
            self.ema_start = cfg.ema.start

        # state
        self.best_valid_loss = float("inf")
        self.train_global_it = 0
        self.valid_global_it = 0
        self.epoch = 0
        self.valid_gen = []
        self.train_gen = []
        self.test_gen = []

        if cfg.resume_from_ckpt:
            assert isinstance(cfg.resume_from_ckpt, str), "resume_from_ckpt must be a string."
            assert os.path.exists(cfg.resume_from_ckpt), f"ckpt {cfg.resume_from_ckpt} not found."
            self.load_state_dict(cfg.resume_from_ckpt)

        # Prepare instances
        self.prepare()
        # Unwrap self.rfm for easier calling
        self.unwrap_rfm: RectifiedFlowMatching = self.accelerator.unwrap_model(self.rfm)

        # Initialize params
        self.sample_rate = cfg.sample_rate
        self.checkpoint_dir = cfg.checkpoint_dir
        self.null_condition_prob = cfg.null_condition_prob
        self.grad_accum_steps = cfg.accelerator.gradient_accumulation_steps
        self.clip_grad_norm = cfg.clip_grad_norm
        self.cfg_strength = cfg.cfg_strength
        self.train_log_extra_interval = cfg.train_log_extra_interval
        self.valid_log_extra_interval = cfg.valid_log_extra_interval
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.num_train_log_data = cfg.num_train_log_data \
            if cfg.num_train_log_data < cfg.batch_size \
            else cfg.batch_size
        self.num_valid_log_data = cfg.num_valid_log_data \
            if cfg.num_valid_log_data < cfg.batch_size \
            else cfg.batch_size

        self.n_epochs = cfg.n_epochs
        self.save_ckpt_per_epoch = cfg.save_ckpt_per_epoch
        self.save_ckpt_every_num_steps = cfg.save_ckpt_every_num_steps

        # Prepare for inference
        if not self.training:
            assert cfg.run_dir is not None, "run_dir must be provided for inference."

            self.num_test_log_data = cfg.num_test_log_data \
                if cfg.num_test_log_data < cfg.batch_size \
                else cfg.batch_size

    def define_wandb_metrics(self):
        wandb_tracker = self.accelerator.get_tracker("wandb", unwrap=True)
        if self.is_main:
            # Define custom x-axis metric
            wandb_tracker.define_metric("train_step")
            wandb_tracker.define_metric("valid_step")
            wandb_tracker.define_metric("epoch")

            # Define which metrics to plot against that x-axis
            fields = LossBreakdown._fields
            for field in fields:
                wandb_tracker.define_metric(f"train/{field}", step_metric="train_step")
                wandb_tracker.define_metric(f"valid/{field}", step_metric="valid_step")
            wandb_tracker.define_metric("train/lr", step_metric="train_step")
            wandb_tracker.define_metric("train/loss_epoch", step_metric="epoch")
            wandb_tracker.define_metric("valid/loss_epoch", step_metric="epoch")

            return wandb_tracker.id

    def save_config(self, cfg):
        if self.is_main:
            filename = "train_cfg.yaml" if self.training else "infer_cfg.yaml"
            with open(os.path.join(cfg.run_dir, filename), "w") as f:
                OmegaConf.save(cfg, f, resolve=True)

    def prepare(self):
        if self.training:
            train_loader, valid_loader = self.dataloader
            self.rfm, self.ema_model, self.vae, self.vocoder, self.optimizer, self.scheduler, self.train_loader, self.valid_loader = \
                self.accelerator.prepare(
                    self.rfm, self.ema_model, self.vae, self.vocoder, self.optimizer, self.scheduler, train_loader, valid_loader
                )
            self.test_loader = None
        else:
            test_loader = self.dataloader[0]
            self.rfm, self.ema_model, self.vae, self.vocoder, self.optimizer, self.scheduler, self.test_loader = \
                self.accelerator.prepare(
                    self.rfm, self.ema_model, self.vae, self.vocoder, self.optimizer, self.scheduler, test_loader
                )
            self.train_loader = self.valid_loader = None

    def setup_scheduler(self, cfg: DictConfig) -> optim.lr_scheduler.LRScheduler:

        warmup_steps = cfg.warmup_steps

        def warmup(current_step: int):
            return (current_step + 1) / (warmup_steps + 1)

        warmup_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, warmup)

        if cfg.type == "constant":
            next_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lambda _: 1)
        elif cfg.type == "poly":
            total_num_iter = cfg.iterations
            next_scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer, lambda x: (1 - x / total_num_iter)**cfg.power
            )
        elif cfg.type == "step":
            next_scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer, cfg.milestones, cfg.gamma
            )
        elif cfg.type == "cosine":
            next_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, cfg.T_max, cfg.eta_min
            )
        else:
            raise ValueError(f"Invalid scheduler type: {cfg.type}")

        scheduler = optim.lr_scheduler.SequentialLR(
            self.optimizer, [warmup_scheduler, next_scheduler], [warmup_steps]
        )

        return scheduler

    def setup_dataloader(self, cfg: DictConfig) -> List[DataLoader]:
        if self.training:
            logger.info("Load training dataset.")
            train_dataset = ExtractedVATDataset(
                tsv_file=cfg.data.train_dataset.tsv_file,
                memmap_dir=cfg.data.train_dataset.memmap_dir,
                generator=self.rng,
                debug=cfg.debug,
                n_samples_for_debug=cfg.n_samples_for_debug,
                output_dir=cfg.run_dir,
            )
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=cfg.num_workers,
                pin_memory=True,
            )
            logger.info("Load validation dataset.")
            valid_dataset = ExtractedVATDataset(
                tsv_file=cfg.data.valid_dataset.tsv_file,
                memmap_dir=cfg.data.valid_dataset.memmap_dir,
                generator=self.rng,
                debug=cfg.debug,
                n_samples_for_debug=cfg.n_samples_for_debug // 4,
            )
            valid_dataloader = DataLoader(
                valid_dataset,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                pin_memory=True,
            )
            dataloader = [train_dataloader, valid_dataloader]
        else:
            logger.info("Load test dataset.")
            test_dataset = ExtractedVATDataset(
                tsv_file=cfg.data.test_dataset.tsv_file,
                memmap_dir=cfg.data.test_dataset.memmap_dir,
                generator=self.rng,
                debug=cfg.debug,
                n_samples_for_debug=cfg.n_samples_for_debug,
                output_dir=cfg.run_dir,
            )
            test_dataloader = DataLoader(
                test_dataset,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                pin_memory=True,
            )
            dataloader = [test_dataloader]

        return dataloader

    def compute_latent_stats(self):
        for loader in self.dataloader:
            if loader.dataset.split == "train":
                return loader.dataset.compute_latent_stats()
        return None, None

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def save_last_k_models(self, k: int = 2):
        """
        Save the most recent k checkpoints (by epoch).
        Filenames: epoch={epoch}.pth
        Keeps only the latest k checkpoints.
        """

        # Save current checkpoint
        ckpt_name = f"epoch={str(self.epoch).zfill(3)}.pth"
        self.save_state_dict(ckpt_name)

        # List all checkpoints and extract epochs
        pattern = re.compile(r"epoch=(\d+)\.pth")
        ckpts = []
        for fname in os.listdir(self.checkpoint_dir):
            match = pattern.fullmatch(fname)
            if match:
                ep = int(match.group(1))
                ckpts.append((ep, fname))

        # If more than k, remove the oldest (smallest epoch)
        if self.is_main and len(ckpts) > k:
            oldest = min(ckpts, key=lambda x: x[0])
            os.remove(os.path.join(self.checkpoint_dir, oldest[1]))

    def save_state_dict(self, filename: str):
        self.accelerator.wait_for_everyone()

        if self.is_main:
            ema = self.accelerator.unwrap_model(self.ema_model) if self.use_ema else None
            rfm = self.accelerator.unwrap_model(self.rfm)

            state_dict = {
                "best_valid_loss": self.best_valid_loss,
                "train_global_it": self.train_global_it,
                "valid_global_it": self.valid_global_it,
                "epoch": self.epoch,
                "ema": ema.state_dict() if ema is not None else None,
                "rfm": rfm.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict()
            }

            filepath = os.path.join(self.checkpoint_dir, filename)

            torch.save(state_dict, filepath)

            logger.info(f"Save checkpoint to {filepath} at train_step={self.train_global_it} "
                        f"with best_valid_loss={self.best_valid_loss:.3f}")

    def load_state_dict(self, path: str):
        with self.accelerator.main_process_first():
            state_dict = torch.load(path, map_location=self.accelerator.device, weights_only=False)
            if self.ema_model:
                self.ema_model.load_state_dict(state_dict["ema"])
                logger.info(f"EMA states loaded from step {self.ema_model.step}")
            self.rfm.load_state_dict(state_dict["rfm"])
            logger.info("Load optimizer and scheduler states.")
            self.optimizer.load_state_dict(state_dict["optimizer"])
            self.scheduler.load_state_dict(state_dict["scheduler"])

            self.train_global_it = state_dict["train_global_it"]
            self.valid_global_it = state_dict["valid_global_it"]
            self.best_valid_loss = state_dict["best_valid_loss"]
            self.epoch = state_dict["epoch"]

            logger.info(f"Global iter {self.train_global_it}/Epoch {self.epoch} progress loaded.")
            logger.info(f"Current best valid loss: {self.best_valid_loss:.3f}")
            logger.info("Network weights, optimizer states, and scheduler states are loaded.")

            return self.train_global_it

    def load_weights(self, path: str):
        """This method only loads the RFM's weight and should be used to load a pretrained model"""
        with self.accelerator.main_process_first():
            src_dict = torch.load(path, map_location=self.accelerator.device, weights_only=True)
            prefix = "" if self.accelerator.num_processes == 1 else "module."

            src_dict = {f"{prefix}{k}": v for k, v in src_dict.items()}

            if f"{prefix}model.multi_audio_rot" in src_dict:
                del src_dict[f"{prefix}model.multi_audio_rot"]
            if f"{prefix}model.single_audio_rot" in src_dict:
                del src_dict[f"{prefix}model.single_audio_rot"]
            if f"{prefix}model.multi_visual_rot" in src_dict:
                del src_dict[f"{prefix}model.multi_visual_rot"]

            self.rfm.load_state_dict(src_dict, strict=True)

    def log(self, *args, **kwargs):
        return self.accelerator.log(*args, **kwargs)

    def print(self, *args, **kwargs):
        return self.accelerator.print(*args, **kwargs)

    def log_data(self, split, gt_mels, gt_audios, pred_mels, pred_audios, id, caption, iter):
        columns = ["Iter", "ID", "Caption", "Mels", "GT_Audio", "Pred_Audio"]
        data = []

        # Convert mel spectrograms to wandb Image objects
        mel_imgs = [wandb.Image(mels_to_image(mel_gt, mel_synth), mode='RGB')
                    for mel_gt, mel_synth in zip(gt_mels, pred_mels)]

        # Convert audio to wandb Audio objects
        gt_audio_objs = [wandb.Audio(audio, sample_rate=self.sample_rate) for audio in gt_audios]
        pred_audio_objs = [wandb.Audio(audio, sample_rate=self.sample_rate) for audio in pred_audios]

        # Create table data
        for i in range(len(id)):
            data.append([
                iter,
                id[i],
                caption[i],
                mel_imgs[i],
                gt_audio_objs[i],
                pred_audio_objs[i]
            ])

        if split == "train":
            self.train_gen.extend(data)
            table = wandb.Table(columns=columns, data=self.train_gen)
        elif split == "valid":
            self.valid_gen.extend(data)
            table = wandb.Table(columns=columns, data=self.valid_gen)
        else:
            self.test_gen.extend(data)
            table = wandb.Table(columns=columns, data=self.test_gen)

        self.log({f"{split}_generations": table})

    def training_forward(self, batch):
        # A single forward pass
        latent_mean = batch["audio_mean"]
        latent_std = batch["audio_std"]
        visual_feat = batch["video_features"]
        text_feat = batch["text_features"]
        text_len = batch["text_len"]
        batch_size = latent_mean.size(0)

        # Sample latent
        latent_noise = torch.empty_like(
            latent_mean, device=latent_mean.device).normal_(generator=self.rng)
        x1 = latent_mean + latent_std * latent_noise

        # Classifier-free training
        visual_mask_ids = torch.rand(batch_size, device=x1.device, generator=self.rng)
        text_mask_ids = torch.rand(batch_size, device=x1.device, generator=self.rng)
        # Mask ids
        null_video = (visual_mask_ids < self.null_condition_prob)
        null_text = (text_mask_ids < self.null_condition_prob)
        # Mask conditions
        visual_feat[null_video] = self.unwrap_rfm.model.empty_visual_feature
        text_feat[null_text] = self.unwrap_rfm.model.empty_text_feature

        # Forward pass
        loss, loss_breakdown = self.rfm(x1, visual_feat, text_feat, text_len, noise=None)

        return x1, loss, loss_breakdown

    def validation_forward(self, batch):
        # A single forward pass
        latent_mean = batch["audio_mean"]
        latent_std = batch["audio_std"]
        visual_feat = batch["video_features"]
        text_feat = batch["text_features"]
        text_len = batch["text_len"]
        batch_size = latent_mean.size(0)

        # Sample latent
        latent_noise = torch.empty_like(latent_mean).normal_(generator=self.rng)
        x1 = latent_mean + latent_std * latent_noise

        # Classifier-free training
        visual_mask_ids = torch.rand(batch_size, device=x1.device, generator=self.rng)
        text_mask_ids = torch.rand(batch_size, device=x1.device, generator=self.rng)
        # Mask ids
        null_video = (visual_mask_ids < self.null_condition_prob)
        null_text = (text_mask_ids < self.null_condition_prob)
        # Mask conditions
        visual_feat[null_video] = self.unwrap_rfm.model.empty_visual_feature
        text_feat[null_text] = self.unwrap_rfm.model.empty_text_feature

        # Forward pass
        loss, loss_breakdown = self.rfm(x1, visual_feat, text_feat, text_len, noise=None)

        return x1, loss, loss_breakdown

    def training_step(self, batch):
        if not self.training:
            raise ValueError("train() should not be called when training is False.")

        # self.rfm.train()
        with self.accelerator.accumulate(self.rfm):
            x1, loss, loss_breakdown = self.training_forward(batch)
            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.rfm.parameters(), self.clip_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        if self.ema_model and self.train_global_it >= self.ema_start:
            self.ema_model.update()

        log_stats = {f"train/{k}": v for k, v in loss_breakdown._asdict().items()}
        log_stats.update({
            "train/lr": self.scheduler.get_last_lr()[0],
            "train_step": self.train_global_it,
        })
        self.log(log_stats)

        # Log extra information
        if self.train_global_it % self.train_log_extra_interval == 0:
            n_log = min(self.num_train_log_data, batch["text_features"].size(0))
            if n_log > 0:
                x1 = self.unwrap_rfm.model.denormalize(x1[:n_log])
                gt_mel = self.vae.decode(x1.transpose(1, 2))
                gt_audio = self.vocoder(gt_mel)

                # Sample latent for prediction
                text_feat = batch["text_features"][:n_log]
                visual_feat = batch["video_features"][:n_log]
                text_len = batch["text_len"][:n_log]
                pred_x1 = self.unwrap_rfm.sample(
                    text_feat,
                    visual_feat,
                    text_len,
                    data_shape=x1.shape[1:],
                    batch_size=n_log,
                    noise=None,
                )
                pred_mel = self.vae.decode(pred_x1.transpose(1, 2))
                pred_audio = self.vocoder(pred_mel)

                id = batch["id"][:n_log]  # list of str
                caption = batch["caption"][:n_log]  # list of str
                self.log_data(
                    "train",
                    gt_mel.cpu().numpy(),
                    gt_audio.squeeze(1).cpu().numpy(),
                    pred_mel.cpu().numpy(),
                    pred_audio.squeeze(1).cpu().numpy(),
                    id,
                    caption,
                    self.train_global_it
                )

        return loss.detach().cpu().item()

    @torch.inference_mode()
    def validation_step(self, batch):
        # self.rfm.eval()
        x1, loss, loss_breakdown = self.validation_forward(batch)
        log_stats = {f"valid/{k}": v for k, v in loss_breakdown._asdict().items()}
        log_stats.update({"valid_step": self.valid_global_it})
        self.log(log_stats)

        if self.valid_global_it % self.valid_log_extra_interval == 0:
            n_log = min(self.num_valid_log_data, batch["text_features"].size(0))
            if n_log > 0:
                x1 = self.unwrap_rfm.model.denormalize(x1[:n_log])
                gt_mel = self.vae.decode(x1.transpose(1, 2))
                gt_audio = self.vocoder(gt_mel)

                # Sample latent for prediction
                text_feat = batch["text_features"][:n_log]
                visual_feat = batch["video_features"][:n_log]
                text_len = batch["text_len"][:n_log]
                pred_x1 = self.unwrap_rfm.sample(
                    text_feat,
                    visual_feat,
                    text_len,
                    data_shape=x1.shape[1:],
                    batch_size=n_log,
                    noise=None,
                )

                pred_mel = self.vae.decode(pred_x1.transpose(1, 2))
                pred_audio = self.vocoder(pred_mel)

                id = batch["id"][:n_log]  # list of str
                caption = batch["caption"][:n_log]  # list of str
                self.log_data(
                    "valid",
                    gt_mel.cpu().numpy(),
                    gt_audio.squeeze(1).cpu().numpy(),
                    pred_mel.cpu().numpy(),
                    pred_audio.squeeze(1).cpu().numpy(),
                    id, caption, self.valid_global_it
                )

        return loss.detach().cpu().item()

    @torch.inference_mode()
    def inference_step(self, batch, it: int, vocode_gt: bool = False):
        latent_mean = batch["audio_mean"]
        text_feat = batch["text_features"]
        visual_feat = batch["video_features"]
        text_len = batch["text_len"]

        n_log = min(self.num_test_log_data, latent_mean.size(0))

        # Sample
        pred_x1 = self.unwrap_rfm.sample(
            text_feat,
            visual_feat,
            text_len,
            data_shape=latent_mean.shape[1:],
            batch_size=latent_mean.size(0),
            noise=None,
            inference=True,
            cfg_strength=self.cfg_strength
        )
        pred_mel = self.vae.decode(pred_x1.transpose(1, 2))
        pred_audio = self.vocoder(pred_mel)
        id = batch["id"]
        if n_log > 0:
            if vocode_gt:
                latent_std = batch["audio_std"]
                latent_noise = torch.empty_like(latent_mean).normal_(generator=self.rng)
                x1 = latent_mean + latent_std * latent_noise
                gt_mel = self.vae.decode(x1.transpose(1, 2))
                gt_audio = self.vocoder(gt_mel)
            else:
                gt_mel, gt_audio = None, None

            caption = batch["caption"]
            self.log_data(
                "test",
                gt_mel[:n_log].cpu().numpy() if gt_mel is not None else None,
                gt_audio[:n_log].squeeze(1).cpu().numpy() if gt_audio is not None else None,
                pred_mel[:n_log].cpu().numpy(),
                pred_audio[:n_log].squeeze(1).cpu().numpy(),
                id[:n_log],
                caption[:n_log],
                it
            )
        return pred_audio, id

    def train(self):

        for epoch in tqdm(range(self.n_epochs), desc="Training"):
            self.rfm.train()
            train_loss = []
            for batch in self.train_loader:
                loss = self.training_step(batch)
                train_loss.append(loss)
                self.train_global_it += 1

            train_loss = sum(train_loss) / len(train_loss)

            # Validation
            self.rfm.eval()
            valid_loss = []
            for batch in self.valid_loader:
                loss = self.validation_step(batch)
                self.valid_global_it += 1
                valid_loss.append(loss)
            valid_loss = sum(valid_loss) / len(valid_loss)
            self.log({
                "epoch": self.epoch,
                "train/loss_epoch": train_loss,
                "valid/loss_epoch": valid_loss,
            })

            if self.save_ckpt_per_epoch:
                self.save_last_k_models(k=2)

            if valid_loss < self.best_valid_loss:
                logger.info(
                    f"Best valid loss: {self.best_valid_loss} -> {valid_loss} at epoch {epoch}."
                )
                self.best_valid_loss = valid_loss
            self.epoch += 1

        self.accelerator.end_training()

    def inference(self, vocode_gt: bool = True):
        self.rfm.eval()

        synthesized_audio = []
        ids = []

        for i, batch in tqdm(enumerate(self.test_loader), total=len(self.test_loader)):
            audio, id = self.inference_step(batch, i, vocode_gt=vocode_gt)
            synthesized_audio.append(audio)
            ids.extend(id)
        synthesized_audio = torch.cat(synthesized_audio, dim=0).transpose(1, 2).cpu()

        # end accelerate
        self.accelerator.end_training()

        return ids, synthesized_audio
