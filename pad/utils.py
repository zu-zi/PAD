"""Logger, running-average meter, EMA teacher and R1/mAP evaluator."""

import copy
import logging
import os
import sys
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn


# ---- logger ---------------------------------------------------------------
def setup_logger(name: str = "pad", save_dir: str = "", if_train: bool = True) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if logger.hasHandlers():
        logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fh = logging.FileHandler(
            os.path.join(save_dir, "train_log.txt" if if_train else "test_log.txt"), mode="w",
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# ---- meter ----------------------------------------------------------------
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


# ---- EMA teacher (PAD Eq. 6) ---------------------------------------------
class EMATeacher(nn.Module):
    r"""Momentum teacher: ``theta_tea = m * theta_tea + (1 - m) * theta_stu``."""

    def __init__(self, student: nn.Module, momentum: float = 0.997,
                 device: Optional[torch.device] = None):
        super().__init__()
        self.momentum = float(momentum)
        self.teacher = copy.deepcopy(student)
        if device is None:
            try:
                device = next(student.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        self.teacher.to(device)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()

    @torch.no_grad()
    def update(self, student: nn.Module, momentum: Optional[float] = None) -> None:
        m = self.momentum if momentum is None else float(momentum)
        cur_params: Dict[str, torch.Tensor] = dict(student.named_parameters())
        cur_buffers: Dict[str, torch.Tensor] = dict(student.named_buffers())
        for name, p in self.teacher.named_parameters():
            if name not in cur_params:
                continue
            p_cur = cur_params[name]
            if p.shape != p_cur.shape or not torch.is_floating_point(p):
                continue
            p.mul_(m).add_(p_cur.detach().to(device=p.device, dtype=p.dtype), alpha=(1.0 - m))
        for name, b in self.teacher.named_buffers():
            if name not in cur_buffers:
                continue
            b_cur = cur_buffers[name]
            if b.shape != b_cur.shape or not torch.is_floating_point(b):
                continue
            b.mul_(m).add_(b_cur.detach().to(device=b.device, dtype=b.dtype), alpha=(1.0 - m))

    def forward(self, *args, **kwargs):
        return self.teacher(*args, **kwargs)


# ---- R1 / mAP evaluator ---------------------------------------------------
def _euclidean_distance(qf: torch.Tensor, gf: torch.Tensor) -> np.ndarray:
    m, n = qf.shape[0], gf.shape[0]
    dist = (
        qf.pow(2).sum(dim=1, keepdim=True).expand(m, n)
        + gf.pow(2).sum(dim=1, keepdim=True).expand(n, m).t()
    )
    dist.addmm_(1, -2, qf, gf.t())
    return dist.cpu().numpy()


def _eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank: int = 50):
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    all_cmc, all_AP = [], []
    num_valid_q = 0.0
    for q_idx in range(num_q):
        q_pid, q_camid = q_pids[q_idx], q_camids[q_idx]
        order = indices[q_idx]
        keep = np.invert((g_pids[order] == q_pid) & (g_camids[order] == q_camid))
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            continue
        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.0
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum() / np.arange(1, orig_cmc.shape[0] + 1)
        all_AP.append((tmp_cmc * orig_cmc).sum() / num_rel)

    assert num_valid_q > 0, "All query identities are missing from the gallery."
    cmc = np.asarray(all_cmc).astype(np.float32).sum(0) / num_valid_q
    return cmc, float(np.mean(all_AP))


class R1_mAP_eval:
    """Accumulates query/gallery features and computes CMC + mAP."""

    def __init__(self, num_query: int, max_rank: int = 50, feat_norm: str = "yes"):
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reset()

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat, pid, camid = output
        self.feats.append(feat.cpu())
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == "yes":
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
        qf, gf = feats[: self.num_query], feats[self.num_query:]
        q_pids = np.asarray(self.pids[: self.num_query])
        q_camids = np.asarray(self.camids[: self.num_query])
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        distmat = _euclidean_distance(qf, gf)
        cmc, mAP = _eval_func(distmat, q_pids, g_pids, q_camids, g_camids, self.max_rank)
        return cmc, mAP
