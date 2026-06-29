import torch
import torch.nn as nn
import math
from functools import partial
from torch import Tensor
from torchdiffeq import odeint
from scipy.optimize import linear_sum_assignment
from .model import Flowley
from .losses import (
    PseudoHuberLoss,
    VelocityDirectionLoss,
    LossBreakdown,
)
from accelerate.logging import get_logger


logger = get_logger(__name__)


def cosmap(bs: int) -> Tensor:
    u = torch.rand(bs)
    return 1 - (1 / (torch.tan(math.pi / 2 * u) + 1))


def logit_normal_sample(bs: int, m: float = 0., s: float = 1.) -> Tensor:
    # Logit-normal sampling introduced in SD3 paper
    s = torch.randn(bs) * s + m
    return torch.sigmoid(s)


# Adapted from https://github.com/lucidrains/rectified-flow-pytorch
# and https://github.com/hkchengrex/MMAudio
class RectifiedFlowMatching(nn.Module):

    def __init__(
        self,
        model: Flowley,
        loss_type: str = "mse",
        ode_solver: str = "euler",
        noise_scheduler: str = "cosmap",
        lns_mean: float = 0.0,
        lns_scale: float = 0.1,
        num_steps: int = 25,
        direction_loss_weight: float = 1.0,
        immiscible: bool = False,
        **kwargs
    ):

        super(RectifiedFlowMatching, self).__init__()

        self.model = model
        if loss_type == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_type == "pseudo_huber":
            self.loss_fn = PseudoHuberLoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        self.direction_loss_fn = VelocityDirectionLoss()

        if noise_scheduler == "cosmap":
            self.noise_scheduler = partial(cosmap)
        elif noise_scheduler == "logit_normal":
            self.noise_scheduler = partial(
                logit_normal_sample, m=lns_mean, s=lns_scale
            )
        elif not noise_scheduler:
            self.noise_scheduler = nn.Identity()
        else:
            raise ValueError(f"Unknown noise scheduler: {noise_scheduler}")

        self.ode_solver = ode_solver
        self.num_steps = num_steps
        # Auxilliary losses
        self.direction_loss_weight = direction_loss_weight

        self.immiscible = immiscible

    @property
    def device(self):
        return next(self.model.parameters()).device

    def predict_flow(
        self,
        model: nn.Module,
        latent: Tensor,
        text_feat: Tensor,
        visual_feat: Tensor,
        text_len: Tensor,
        t: Tensor
    ):
        if t.ndim == 3:
            t = t.squeeze(dim=(1, 2))
        elif t.ndim == 0:
            t = t.unsqueeze(0)
        assert t.ndim == 1, f"t should have the ndim=1 but {t.ndim}"
        flow, proj_visual_feat = model(latent, text_feat, visual_feat, text_len, t=t)
        return flow, proj_visual_feat

    def inference(
        self,
        visual_feat: Tensor,
        text_feat: Tensor,
        text_len: Tensor,
        cfg_strength: float,
        t: Tensor,
        latent: Tensor,
    ):
        bs = latent.size(0)
        empty_visual_feat, empty_text_feat = self.model.get_empty_conditions(bs)

        if cfg_strength < 1.0:
            return self.predict_flow(self.model, latent, text_feat, visual_feat, text_len, t)[0]
        else:
            return (
                cfg_strength *
                self.predict_flow(self.model, latent, text_feat, visual_feat, text_len, t)[0] +
                (1 - cfg_strength) *
                self.predict_flow(
                    self.model, latent, empty_text_feat, empty_visual_feat, text_len, t)[0]
            )

    def get_noises_and_flows(
        self,
        model: nn.Module,
        latent: Tensor,
        visual_feat: Tensor,
        text_feat: Tensor,
        text_len: Tensor,
        noise: Tensor,
        t: Tensor
    ):

        # Get intermediate latent
        xt = t * latent + (1. - t) * noise

        # Flows
        tgt_flow = latent - noise
        pred_flow, proj_visual_feat = self.predict_flow(
            model, xt, text_feat, visual_feat, text_len, t
        )

        # Predicted latent
        pred_latent = xt + (1. - t) * pred_flow

        return xt, pred_latent, pred_flow, tgt_flow, proj_visual_feat

    @torch.no_grad()
    def sample(
        self,
        text_feat: Tensor,
        visual_feat: Tensor,
        text_len: Tensor,
        data_shape: list[int],
        batch_size: int = 1,
        noise: Tensor = None,
        inference: bool = False,
        cfg_strength: float = None,
    ):
        was_training = self.training
        self.eval()

        def ode_fn(t, x):
            flow, _ = self.predict_flow(self.model, x, text_feat, visual_feat, text_len, t)
            return flow

        if inference:
            assert cfg_strength is not None, "cfg_strength should be provided for inference"
            fn = partial(self.inference, visual_feat, text_feat, text_len, cfg_strength)
        else:
            fn = ode_fn

        noise = torch.randn(batch_size, *data_shape, device=self.device) if noise is None else noise

        t = torch.linspace(0., 1., self.num_steps, device=self.device)

        if self.ode_solver == "euler":
            sampled_latent = noise
            steps = torch.linspace(0., 1., self.num_steps + 1, device=self.device)
            for ti, t in enumerate(steps[:-1]):
                flow = fn(t, sampled_latent)
                next_t = steps[ti + 1]
                dt = next_t - t
                sampled_latent = sampled_latent + dt * flow
        else:
            trajectory = odeint(fn, noise, t, method=self.ode_solver)
            sampled_latent = trajectory[-1]

        self.train(was_training)

        return self.model.denormalize(sampled_latent)

    def forward(
        self,
        latent: Tensor,
        visual_feat: Tensor,
        text_feat: Tensor,
        text_len: Tensor,
        noise: Tensor = None,
    ):

        noise = torch.randn_like(latent) if noise is None else noise
        latent = self.model.normalize(latent)

        if self.immiscible:
            cost = torch.cdist(latent.flatten(1), noise.flatten(1))
            _, reorder_indices = linear_sum_assignment(cost.cpu())
            noise = noise[torch.from_numpy(reorder_indices).to(cost.device)]

        # Get timestep
        t = self.noise_scheduler(bs=latent.size(0)).to(latent.device)
        t = t[:, None, None]

        # Get noises and flows
        _, _, pred_flow, tgt_flow, _ = self.get_noises_and_flows(
            self.model, latent, visual_feat, text_feat, text_len, noise, t
        )

        # Compute losses
        main_loss = self.loss_fn(pred_flow, tgt_flow)
        direction_loss = self.direction_loss_fn(pred_flow, tgt_flow)

        loss = main_loss + direction_loss * self.direction_loss_weight

        return loss, LossBreakdown(
            total=loss,
            main=main_loss,
            direction_loss=direction_loss,
        )
