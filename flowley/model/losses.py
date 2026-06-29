import math
import random
import torch
import torch.nn.functional as F
from collections import namedtuple
from torch import Tensor
from torch.nn.modules.loss import _Loss


LossBreakdown = namedtuple("LossBreakdown", ["total", "main", "direction_loss"])


class PseudoHuberLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(PseudoHuberLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        data_dim = pred.shape[-1]
        c = 0.00054 * data_dim
        loss = (F.mse_loss(pred, target, reduction=self.reduction) + c * c).sqrt() - c

        return loss


class VelocityDirectionLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(VelocityDirectionLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        loss = 1 - F.cosine_similarity(pred, target, dim=-1)
        if self.reduction == "mean":
            loss = loss.mean()
        else:
            raise NotImplementedError(f"Reduction method {self.reduction} is not implemented")

        return loss


class InfoNCELoss(_Loss):
    def __init__(
        self,
        temperature: float = 0.07,
        size_average=None,
        reduce=None,
        reduction: str = 'mean'
    ) -> None:
        super(InfoNCELoss, self).__init__(size_average, reduce, reduction)
        self.temperature = temperature

    def forward(self, vfeat: Tensor, afeat: Tensor) -> Tensor:
        """InfoNCELoss

        Args:
            vfeat (Tensor): [B, D]
            afeat (Tensor): [B, D]

        Returns:
            Tensor: InfoNCE loss
        """
        # L2 normalize features for cosine similarity
        afeat = F.normalize(afeat, dim=-1)
        vfeat = F.normalize(vfeat, dim=-1)
        bs = afeat.size(0)

        # Similarity matrix: [bs, bs] (row i: a_i with all v_j)
        logits_a2v = (afeat @ vfeat.T) / self.temperature  # [bs, bs]
        logits_v2a = (vfeat @ afeat.T) / self.temperature  # [bs, bs]

        # Targets: positives are diagonal elements [0, 1, ..., bs-1]
        labels = torch.arange(bs, dtype=torch.long, device=afeat.device)

        # Cross-entropy loss (efficient, numerically stable)
        loss_a2v = F.cross_entropy(logits_a2v, labels, reduction=self.reduction)
        loss_v2a = F.cross_entropy(logits_v2a, labels, reduction=self.reduction)

        # Symmetric loss
        loss = (loss_a2v + loss_v2a) / 2
        return loss


def make_segments(
    video: Tensor,
    audio: Tensor,
    video_segment_sz: float,
    segment_stride: float,
    n_segments: int = None,
    video_fps: float = 8.,
    audio_fps: float = 31.25,
    random_start: bool = False,
):
    """Split video and audio (latent) representations to multiple segments

    Args:
        video (Tensor): video tensor of shape [B, vframes, D]
        audio (Tensor): audio tensor of shape [B, aframes, D]
        video_segment_sz (float): length of a video segment (in #frames)
        segment_stride (float): number of segment a stride has
        n_segments (int): number of segments in a sample. If None, it defaults to possible max.
        video_fps (float): video fps
        audio_fps (float): audio fps
        random_start (bool): if False, we extract the center of the sample.

    Returns:
        Tuple(Tensor, Tensor): video and audio segments of shape [B, n_segments, vid_seg_size, D]
                               and [B, n_segments, aud_seg_size, D]
    """
    audio_segment_sz = int(video_segment_sz / video_fps * audio_fps)
    vframes_stride = int(video_segment_sz * segment_stride)
    aframes_stride = int(audio_segment_sz * segment_stride)

    video_len = video.size(1)
    audio_len = audio.size(1)

    # Calculate number of segment
    n_segments_max_v = math.floor((video_len - video_segment_sz) / vframes_stride) + 1
    n_segments_max_a = math.floor((audio_len - audio_segment_sz) / aframes_stride) + 1
    n_segments_max = min(n_segments_max_a, n_segments_max_v)
    n_segments = n_segments_max if n_segments is None else n_segments
    assert n_segments <= n_segments_max

    # Calculate length of sequence of segments (in frames)
    segment_len = n_segments * segment_stride - segment_stride + 1
    vframes_segment_len = int(segment_len * video_segment_sz)
    aframes_segment_len = int(segment_len * audio_segment_sz)

    max_v_start = video_len - vframes_segment_len
    v_start = random.randint(0, max_v_start) if random_start else max_v_start // 2
    a_start = int(v_start / video_fps * audio_fps)
    # Make segments starts
    v_starts = torch.tensor(
        [v_start + i * vframes_stride for i in range(n_segments)], dtype=torch.int32
    )
    a_starts = torch.tensor(
        [a_start + i * aframes_stride for i in range(n_segments)], dtype=torch.int32
    )
    v_ends = v_starts + video_segment_sz
    a_ends = a_starts + audio_segment_sz
    video_segments = torch.stack([video[:, s:e, :] for s, e in zip(v_starts, v_ends)], dim=1)
    audio_segments = torch.stack([audio[:, s:e, :] for s, e in zip(a_starts, a_ends)], dim=1)
    return video_segments, audio_segments


def sample_contrastive_pairs(
    data: tuple[Tensor, Tensor],
    num_pos_pairs: int = 4,
    delta_t: float = 1,
    video_fps: float = 4.0,
    audio_fps: float = 31.25,
    resolution: int = 100,
):
    """Sample contrastive pairs

    Args:
        data (tuple[Tensor, Tensor]): Audio and video features (B, T1, D) and (B, T2, D)
        num_pos_pairs (int, optional): Number of positive pairs in a single feature.
        num_neg_samples_per_pos (int, optional): Number of negative samples per positive pair.
        delta_t (float, optional): Duration (in seconds) of the positive pairs.
        video_fps (float, optional): Video frames per second. Defaults to 4.0.
        audio_fps (float, optional): Audio frames per second. Defaults to 31.25.

    Returns:
        List[Tensor]: Audio positive, audio negative, video positive, video negative
    """
    # Unpack data
    audio_feat, video_feat = data
    bs, num_video_frames = video_feat.shape[:2]
    actual_duration = num_video_frames / video_fps

    max_start_time = actual_duration - delta_t

    candidate_times = torch.linspace(0, max_start_time, steps=resolution, device=video_feat.device)
    candidate_probs = torch.ones(bs, resolution, device=video_feat.device)
    pos_indices = torch.multinomial(candidate_probs, num_pos_pairs, replacement=False)
    pos_start_times = candidate_times[pos_indices]

    # Compute positive video indices
    pos_video_start = torch.round(pos_start_times * video_fps).long()
    pos_video_end = torch.round((pos_start_times + delta_t) * video_fps).long()
    pos_audio_start = torch.round(pos_start_times * audio_fps).long()
    pos_audio_end = torch.round((pos_start_times + delta_t) * audio_fps).long()

    # For negatives, each clip use the other positive start times as negatives
    pos_expanded = pos_start_times.unsqueeze(1).expand(bs, num_pos_pairs, num_pos_pairs)
    # Remove diagonal elements (each sample should not be its own negative)
    diag_mask = ~torch.eye(num_pos_pairs, dtype=torch.bool, device=pos_expanded.device)
    diag_mask = diag_mask.unsqueeze(0).expand(bs, num_pos_pairs, num_pos_pairs)
    neg_start_times = pos_expanded.masked_select(diag_mask).view(bs, num_pos_pairs, num_pos_pairs - 1)

    # Compute negative video indices
    neg_video_start = torch.round(neg_start_times * video_fps).long()
    neg_video_end = torch.round((neg_start_times + delta_t) * video_fps).long()
    neg_audio_start = torch.round(neg_start_times * audio_fps).long()
    neg_audio_end = torch.round((neg_start_times + delta_t) * audio_fps).long()

    audio_positive, audio_negative = extract_contrastive_features(
        audio_feat, pos_audio_start, pos_audio_end, neg_audio_start, neg_audio_end
    )
    video_positive, video_negative = extract_contrastive_features(
        video_feat, pos_video_start, pos_video_end, neg_video_start, neg_video_end
    )
    return audio_positive, audio_negative, video_positive, video_negative


def extract_contrastive_features(
    feature_map: Tensor,
    pos_start: Tensor,
    pos_end: Tensor,
    neg_start: Tensor,
    neg_end: Tensor
):
    """
    Extracts contrastive features by aggregating (mean pooling) along time from a feature map.

    Args:
        feature_map: Tensor of shape (B, T, D) representing features over time.
        pos_start: Tensor of shape (B, M) containing positive sub-range start indices.
        pos_end:   Tensor of shape (B, M) containing positive sub-range end indices.
        neg_start: Tensor of shape (B, M, M-1) containing negative sub-range start indices.
        neg_end:   Tensor of shape (B, M, M-1) containing negative sub-range end indices.

    Returns:
        Tuple[Tensor, Tensor]: positive and negative features extracted from the feature map.
          - "positive": Tensor of shape (B, M, D)
          - "negative": Tensor of shape (B, M, M-1, D)
    """
    T = feature_map.size(1)

    # Create a time index vector of shape (T,) and then expand it.
    time_idx = torch.arange(T, device=feature_map.device).view(1, 1, T)  # shape (1, 1, T)

    # ---- Positive features extraction ----
    # Expand pos_start and pos_end to shape (B, M, 1) to compare with time indices.
    pos_start_exp = pos_start.unsqueeze(-1)  # (B, M, 1)
    pos_end_exp = pos_end.unsqueeze(-1)    # (B, M, 1)

    # Create a mask for each positive sub-range: True if time index is in [start, end)
    pos_mask = (time_idx >= pos_start_exp) & (time_idx <= pos_end_exp)  # (B, M, T)
    pos_mask_float = pos_mask.float()  # convert to float for multiplication

    # Expand feature_map to (B, 1, T, D) so we can broadcast over M.
    feat_exp = feature_map.unsqueeze(1)  # (B, 1, T, D)

    # Multiply features by mask and sum over T.
    pos_features_sum = (feat_exp * pos_mask_float.unsqueeze(-1)).sum(dim=2)  # (B, M, D)

    # Count the number of time steps for each sub-range.
    pos_counts = pos_mask_float.sum(dim=2).unsqueeze(-1)  # (B, M, 1)

    pos_features = pos_features_sum / pos_counts  # (B, M, D)

    # ---- Negative features extraction ----
    # neg_start and neg_end: shape (B, M, M-1)
    # We need to create a mask over time for each negative sub-range.
    # Expand them to shape (B, M, M-1, 1) for comparison.
    neg_start_exp = neg_start.unsqueeze(-1)  # (B, M, M-1, 1)
    neg_end_exp = neg_end.unsqueeze(-1)    # (B, M, M-1, 1)

    # Create a time index tensor of shape (1, 1, 1, T)
    time_idx_neg = torch.arange(T, device=feature_map.device).view(1, 1, 1, T)  # (1, 1, 1, T)

    # Create the mask: True if time index is in [start, end)
    neg_mask = (time_idx_neg >= neg_start_exp) & (time_idx_neg <= neg_end_exp)  # (B, M, M-1, T)
    neg_mask_float = neg_mask.float()

    # Expand feature_map to shape (B, 1, 1, T, D)
    feat_exp_neg = feature_map.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T, D)

    # Multiply and sum over T.
    neg_features_sum = (feat_exp_neg * neg_mask_float.unsqueeze(-1)).sum(dim=3)  # (B, M, M-1, D)

    # Count the time steps in each negative sub-range.
    neg_counts = neg_mask_float.sum(dim=3).unsqueeze(-1)  # (B, M, M-1, 1)

    neg_features = neg_features_sum / neg_counts  # (B, M, M-1, D)

    return pos_features, neg_features
