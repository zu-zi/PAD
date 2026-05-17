"""Per-domain lifelong trainer for PAD."""

import argparse
import copy
import os
import random

import numpy as np
import torch

from pad import (
    WarmupMultiStepLR, cfg, do_train_stage1, do_train_stage2,
    load_cfg_from_yaml, make_cosine_scheduler, make_dataloader,
    make_loss, make_model, make_optimizer, setup_logger,
)


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _set_all_requires_grad(model, flag: bool):
    for p in model.parameters():
        p.requires_grad_(flag)


def _set_by_name(model, substrings, flag: bool):
    if isinstance(substrings, str):
        substrings = (substrings,)
    keys = tuple(s.lower() for s in substrings)
    for n, p in model.named_parameters():
        if any(k in n.lower() for k in keys):
            p.requires_grad_(flag)


def _unfreeze_last_blocks(model, n_last: int):
    if n_last <= 0:
        return
    blocks = list(model.image_encoder.transformer.resblocks)
    start = max(0, len(blocks) - int(n_last))
    for i, blk in enumerate(blocks):
        if i >= start:
            for p in blk.parameters():
                p.requires_grad_(True)


def _log_params(model, logger, tag: str):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"[{tag}] trainable params: {trainable}/{total} ({trainable/total:.2%})")


def _filter_resume_state_dict(sd):
    """Keep only the visual-branch weights (backbone + VA-Prompt + BN necks)."""
    def _strip(k):
        while k.startswith("module."):
            k = k[len("module."):]
        return k
    sd = {_strip(k): v for k, v in sd.items()}
    keep = {}
    for k, v in sd.items():
        lk = k.lower()
        if lk.startswith("image_encoder"):
            keep[k] = v
        elif "bottleneck" in lk:
            keep[k] = v
        elif (("ln_post" in lk) and ("text" not in lk)) or (lk.endswith(".proj") and "text" not in lk):
            keep[k] = v
    return keep


def _register_va_grad_mask(model):
    """Zero-out gradients on VA-Prompt slots owned by previous domains."""
    va = getattr(model.image_encoder, "va_prompt", None)
    if va is None:
        return

    def _make_hook(fmask):
        def hook(grad):
            if fmask.ndim == 1:
                expand = fmask.view(-1, 1, 1)
            elif fmask.ndim == 2:
                expand = fmask.unsqueeze(-1)
            else:
                expand = fmask
            grad = grad.clone()
            grad[expand.expand_as(grad)] = 0
            return grad
        return hook

    for li in range(va.num_layers):
        fmask = va.frozen_masks[li].detach().cpu()
        va.e_prompt[li].register_hook(_make_hook(fmask))
        if va.key_learnable:
            va.e_key[li].register_hook(_make_hook(fmask))


def main():
    parser = argparse.ArgumentParser(description="PAD lifelong trainer (per-domain)")
    parser.add_argument("--config_file", default="configs/pad.yml", type=str)
    parser.add_argument("--domain_idx", type=int, default=None,
                        help="index into DOMAINS list inside the config")
    parser.add_argument("--domain_name", type=str, default=None,
                        help="name of a DOMAINS entry (alternative to --domain_idx)")
    parser.add_argument("--resume_ckpt", type=str, default="",
                        help="previous-domain checkpoint; empty on the first domain")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.domain_idx is None and args.domain_name is None:
        parser.error("--domain_idx or --domain_name is required")

    load_cfg_from_yaml(cfg, args.config_file,
                       domain_idx=args.domain_idx, domain_name=args.domain_name)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()

    _set_seed(cfg.SOLVER.SEED)

    output_dir = cfg.OUTPUT_DIR or "./logs_lifelong"
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logger("pad", output_dir, if_train=True)
    logger.info(f"[Lifelong] dataset={cfg.DATASETS.NAMES}  output_dir={output_dir}")
    logger.info(f"[Switches] TEXTKD={cfg.TEXTKD.ENABLE} VISKD={cfg.DISTILL.ENABLE}")

    train_loader_stage2, train_loader_stage1, val_loader, num_query, num_classes = make_dataloader(cfg)
    model = make_model(cfg, num_class=num_classes).cuda()

    prev_ckpt = args.resume_ckpt if args.resume_ckpt and os.path.isfile(args.resume_ckpt) else None
    if prev_ckpt:
        raw = torch.load(prev_ckpt, map_location="cpu")
        raw = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
        filt = _filter_resume_state_dict(raw)
        ret = model.load_state_dict(filt, strict=False)
        logger.info(
            f"[Resume] from {os.path.basename(prev_ckpt)}: loaded={len(filt)} "
            f"missing={len(ret.missing_keys)} unexpected={len(ret.unexpected_keys)}"
        )
    else:
        logger.info("[Resume] first domain -- starting from CLIP pretrain")

    # Safety net: if a later-domain config accidentally forgets to turn on
    # the KD switches, skip KD on the first domain anyway (no teacher).
    if not prev_ckpt:
        cfg.defrost()
        cfg.DISTILL.ENABLE = False
        cfg.TEXTKD.ENABLE = False
        cfg.freeze()

    if hasattr(model.image_encoder, "va_prompt"):
        slots = int(cfg.VA_PROMPT.NEW_SLOTS_PER_DOMAIN)
        model.image_encoder.va_prompt.allocate_new_domain_slots(slots)
        _register_va_grad_mask(model)
        logger.info(f"[VA-Prompt] allocated {slots} new expert slots per layer")

    model_old = copy.deepcopy(model).cpu()
    teacher_for_textkd = model_old if (prev_ckpt and cfg.TEXTKD.ENABLE) else None

    loss_func = make_loss(cfg, num_classes=num_classes)

    # Stage 1: TA-Prompt warm-up.
    _set_all_requires_grad(model, False)
    _set_by_name(model, "prompt_learner", True)
    _log_params(model, logger, "STAGE1")

    optimizer_1 = make_optimizer(cfg, model, stage="STAGE1")
    sched_1 = make_cosine_scheduler(
        optimizer_1,
        num_epochs=cfg.SOLVER.STAGE1.MAX_EPOCHS,
        lr_min=cfg.SOLVER.STAGE1.LR_MIN,
        warmup_lr_init=cfg.SOLVER.STAGE1.WARMUP_LR_INIT,
        warmup_t=cfg.SOLVER.STAGE1.WARMUP_EPOCHS,
    )
    do_train_stage1(
        cfg=cfg, model=model, train_loader_stage1=train_loader_stage1,
        optimizer=optimizer_1, scheduler=sched_1,
        local_rank=0, teacher_model=teacher_for_textkd,
    )

    # Stage 2: visual branch + VISKD.
    if not prev_ckpt:
        _set_all_requires_grad(model, True)
        _set_by_name(model, "prompt_learner", False)
        _log_params(model, logger, "STAGE2 (first domain)")
    else:
        _set_all_requires_grad(model, False)
        _set_by_name(model, "prompt_learner", False)
        _set_by_name(model, ("classifier", "bottleneck"), True)
        if cfg.VA_PROMPT.ENABLE:
            _set_by_name(model, ("va_prompt", "g_prompt", "e_prompt", "e_key"), True)
        _unfreeze_last_blocks(model, int(cfg.UNFREEZE.LAST_BLOCKS))
        _log_params(model, logger, "STAGE2")

    optimizer_2 = make_optimizer(cfg, model, stage="STAGE2")
    sched_2 = WarmupMultiStepLR(
        optimizer_2,
        cfg.SOLVER.STAGE2.STEPS,
        cfg.SOLVER.STAGE2.GAMMA,
        cfg.SOLVER.STAGE2.WARMUP_FACTOR,
        cfg.SOLVER.STAGE2.WARMUP_ITERS,
        cfg.SOLVER.STAGE2.WARMUP_METHOD,
    )

    do_train_stage2(
        cfg, model, train_loader_stage2, val_loader,
        optimizer_2, sched_2, loss_func, num_query,
        local_rank=0,
        teacher_model=(model_old if prev_ckpt else None),
    )

    final_ckpt = os.path.join(output_dir, f"{cfg.MODEL.NAME}_stage2.pth")
    torch.save(model.state_dict(), final_ckpt)
    logger.info(f"[Save] {final_ckpt}")
    logger.info("[Lifelong] finished.")


if __name__ == "__main__":
    main()
