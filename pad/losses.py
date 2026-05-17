"""Loss functions used by PAD: ID / Triplet / I2T (SupCon) / KD."""

from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- cross-entropy with label smoothing ----------------------------------
class CrossEntropyLabelSmooth(nn.Module):
    r"""y_smooth = (1 - eps) * y + eps / K."""

    def __init__(self, num_classes: int, epsilon: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(inputs, dim=1)
        with torch.no_grad():
            y = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
            y = (1 - self.epsilon) * y + self.epsilon / self.num_classes
        return (-y * log_probs).mean(0).sum()


# ---- hard-mining triplet loss --------------------------------------------
def _euclidean_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, n = x.size(0), y.size(0)
    xx = x.pow(2).sum(1, keepdim=True).expand(m, n)
    yy = y.pow(2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy - 2 * torch.matmul(x, y.t())
    return dist.clamp(min=1e-12).sqrt()


class TripletLoss:
    """Triplet loss with batch-hard mining."""

    def __init__(self, margin: float = None):
        self.margin = margin
        self.ranking_loss = (
            nn.MarginRankingLoss(margin=margin) if margin is not None else nn.SoftMarginLoss()
        )

    def __call__(self, feat: torch.Tensor, labels: torch.Tensor):
        dist_mat = _euclidean_dist(feat, feat)
        N = dist_mat.size(0)
        is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
        is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())
        dist_ap = dist_mat[is_pos].contiguous().view(N, -1).max(dim=1)[0]
        dist_an = dist_mat[is_neg].contiguous().view(N, -1).min(dim=1)[0]
        y = dist_an.new_ones(dist_an.size())
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss, dist_ap, dist_an


# ---- supervised contrastive (used for SupCon image-text alignment) -------
class SupConLoss(nn.Module):
    """Image-to-text supervised contrastive loss (Khosla et al., NeurIPS 2020)."""

    def __init__(self, device: str = "cuda", temperature: float = 1.0):
        super().__init__()
        self.device = device
        self.temperature = temperature

    def forward(self, text_features: torch.Tensor, image_features: torch.Tensor,
                t_label: torch.Tensor, i_label: torch.Tensor) -> torch.Tensor:
        B_t, B_i = text_features.shape[0], image_features.shape[0]
        mask = torch.eq(
            t_label.unsqueeze(1).expand(B_t, B_i),
            i_label.unsqueeze(0).expand(B_t, B_i),
        ).float().to(self.device)

        logits = text_features @ image_features.t() / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        log_prob = logits - torch.log(logits.exp().sum(dim=1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return -mean_log_prob_pos.mean()


# ---- knowledge-distillation (VISKD) --------------------------------------
class KDLosses(nn.Module):
    """Feature-level MSE + logit-level temperature-smoothed KL (PAD Eqs. 7-8)."""

    def __init__(self, lambda_feat: float = 0.5, lambda_logit: float = 0.5,
                 temperature: float = 4.0, feat_layers: Optional[Iterable[int]] = None):
        super().__init__()
        self.lambda_feat = float(lambda_feat)
        self.lambda_logit = float(lambda_logit)
        self.temperature = float(temperature)
        self.feat_layers = list(feat_layers) if feat_layers else None

    def _feat_loss(self, s, t) -> torch.Tensor:
        if not isinstance(s, (list, tuple)):
            s, t = [s], [t]
        idxs = self.feat_layers if self.feat_layers else range(min(len(s), len(t)))
        losses = [
            F.mse_loss(s[i], t[i])
            for i in idxs
            if s[i] is not None and t[i] is not None and s[i].shape == t[i].shape
        ]
        if not losses:
            return torch.zeros((), device=s[0].device)
        return torch.stack(losses).mean()

    def kl(self, s_logits: torch.Tensor, t_logits: torch.Tensor) -> torch.Tensor:
        if s_logits is None or t_logits is None or s_logits.shape != t_logits.shape:
            ref = s_logits if isinstance(s_logits, torch.Tensor) else t_logits
            return torch.zeros((), device=(ref.device if ref is not None else "cpu"))
        T = self.temperature
        log_p_s = F.log_softmax(s_logits / T, dim=1)
        p_t = F.softmax(t_logits.detach() / T, dim=1)
        return F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)

    def forward(self, student_feats, teacher_feats, student_logits, teacher_logits):
        device = None
        for ref in (student_logits, teacher_logits):
            if isinstance(ref, torch.Tensor):
                device = ref.device
                break
        if device is None:
            for lst in (student_feats, teacher_feats):
                if isinstance(lst, (list, tuple)) and len(lst):
                    for t in lst:
                        if isinstance(t, torch.Tensor):
                            device = t.device
                            break
                if device is not None:
                    break
        device = device or torch.device("cpu")

        feat_loss = (
            self._feat_loss(student_feats, teacher_feats)
            if self.lambda_feat > 0.0 else torch.zeros((), device=device)
        )
        kl_loss = (
            self.kl(student_logits, teacher_logits)
            if self.lambda_logit > 0.0 else torch.zeros((), device=device)
        )
        total = self.lambda_feat * feat_loss + self.lambda_logit * kl_loss
        return total, feat_loss, kl_loss

    __call__ = forward


# ---- Stage-2 objective builder -------------------------------------------
def make_loss(cfg, num_classes: int):
    """Return a ``(score, feat, target, i2t_score) -> loss`` callable."""
    if cfg.DATALOADER.SAMPLER != "softmax_triplet":
        raise ValueError(f"Unsupported sampler: {cfg.DATALOADER.SAMPLER}")

    triplet = TripletLoss(cfg.SOLVER.MARGIN)
    xent = CrossEntropyLabelSmooth(num_classes) if cfg.MODEL.LABEL_SMOOTH else None

    def _id_loss(score, target):
        if isinstance(score, list):
            return sum(
                xent(s, target) if xent is not None else F.cross_entropy(s, target)
                for s in score
            )
        return xent(score, target) if xent is not None else F.cross_entropy(score, target)

    def _tri_loss(feat, target):
        if isinstance(feat, list):
            return sum(triplet(f, target)[0] for f in feat)
        return triplet(feat, target)[0]

    def loss_fn(score, feat, target, i2t_score=None):
        loss = cfg.MODEL.ID_LOSS_WEIGHT * _id_loss(score, target) \
               + cfg.MODEL.TRIPLET_LOSS_WEIGHT * _tri_loss(feat, target)
        if i2t_score is not None:
            i2t = xent(i2t_score, target) if xent is not None else F.cross_entropy(i2t_score, target)
            loss = loss + cfg.MODEL.I2T_LOSS_WEIGHT * i2t
        return loss

    return loss_fn
