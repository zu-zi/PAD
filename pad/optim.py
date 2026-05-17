"""Optimiser and LR scheduler factories."""

import logging
from bisect import bisect_right
from typing import List, Tuple

import torch
from timm.scheduler.cosine_lr import CosineLRScheduler

logger = logging.getLogger("pad.optim")


# ---- warmup + multistep (used by Stage 2) --------------------------------
class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    """Linear/constant warm-up followed by multi-step decay. Called per epoch."""

    def __init__(self, optimizer, milestones, gamma: float = 0.1,
                 warmup_factor: float = 1.0 / 3, warmup_iters: int = 500,
                 warmup_method: str = "linear", last_epoch: int = -1):
        if list(milestones) != sorted(milestones):
            raise ValueError(f"milestones must be increasing, got {milestones}")
        if warmup_method not in ("constant", "linear"):
            raise ValueError(f"warmup_method must be constant or linear, got {warmup_method}")
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup = 1.0
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup = self.warmup_factor
            else:
                alpha = self.last_epoch / max(self.warmup_iters, 1)
                warmup = self.warmup_factor * (1 - alpha) + alpha
        decay = self.gamma ** bisect_right(self.milestones, self.last_epoch)
        return [base_lr * warmup * decay for base_lr in self.base_lrs]


def make_cosine_scheduler(optimizer, num_epochs: int, lr_min: float,
                          warmup_lr_init: float, warmup_t: int):
    """Cosine scheduler used by Stage 1 (thin wrapper around timm)."""
    return CosineLRScheduler(
        optimizer,
        t_initial=num_epochs,
        lr_min=lr_min,
        warmup_lr_init=warmup_lr_init,
        warmup_t=warmup_t,
        cycle_limit=1,
        t_in_epochs=True,
    )


# ---- param grouping + optimiser construction -----------------------------
_PROMPT_HINTS: Tuple[str, ...] = (
    "va_prompt", "g_prompt", "e_prompt", "e_key", "prompt_learner",
)
_HEAD_HINTS: Tuple[str, ...] = (
    "classifier", "classifier_proj", "bottleneck", "bottleneck_proj",
)


def _match(name: str, hints: Tuple[str, ...]) -> bool:
    lowered = name.lower()
    return any(h in lowered for h in hints)


def _split_param_groups(model: torch.nn.Module, stage: str):
    prompts, heads, backbones = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lowered = n.lower()
        if "text_encoder" in lowered:
            continue
        if stage == "STAGE1" and "va_prompt" in lowered:
            continue
        if stage == "STAGE2" and "prompt_learner" in lowered:
            continue
        if _match(lowered, _PROMPT_HINTS):
            prompts.append(p)
        elif _match(lowered, _HEAD_HINTS):
            heads.append(p)
        else:
            backbones.append(p)
    return prompts, heads, backbones


def make_optimizer(cfg, model: torch.nn.Module, stage: str = "STAGE1") -> torch.optim.Optimizer:
    stage_cfg = getattr(cfg.SOLVER, stage)
    base_lr = float(stage_cfg.BASE_LR)
    base_wd = float(getattr(stage_cfg, "WEIGHT_DECAY", 5e-4))
    optim_name = str(getattr(stage_cfg, "OPTIMIZER", "Adam")).lower()

    prompt_lr = float(cfg.OPT.PROMPT_LR) if cfg.OPT.PROMPT_LR > 0 else base_lr
    prompt_wd = float(cfg.OPT.PROMPT_WD)

    prompts, heads, backbones = _split_param_groups(model, stage.upper())

    def _count(ps: List[torch.nn.Parameter]) -> int:
        return sum(p.numel() for p in ps)

    logger.info(
        "[OPT] stage=%s base_lr=%.3g wd=%.3g | prompt_lr=%.3g wd=%.3g | "
        "prompt=%d head=%d backbone=%d",
        stage, base_lr, base_wd, prompt_lr, prompt_wd,
        _count(prompts), _count(heads), _count(backbones),
    )

    groups = []
    if backbones:
        groups.append({"params": backbones, "lr": base_lr, "weight_decay": base_wd, "name": "backbone"})
    if heads:
        groups.append({"params": heads, "lr": base_lr, "weight_decay": base_wd, "name": "head"})
    if prompts:
        groups.append({"params": prompts, "lr": prompt_lr, "weight_decay": prompt_wd, "name": "prompt"})
    if not groups:
        groups.append({
            "params": [p for p in model.parameters() if p.requires_grad],
            "lr": base_lr, "weight_decay": base_wd, "name": "all",
        })

    if optim_name == "sgd":
        return torch.optim.SGD(groups, momentum=0.9, nesterov=True)
    if optim_name == "adamw":
        return torch.optim.AdamW(groups)
    return torch.optim.Adam(groups)
