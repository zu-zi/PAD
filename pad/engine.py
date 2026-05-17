"""
Training / inference engine for PAD.

* ``do_train_stage1`` -- TA-Prompt warm-up on cached image features
                         (SupCon + optional TEXKD, PAD Sec. 3.3).
* ``do_train_stage2`` -- joint training of the visual branch with VISKD
                         under an EMA teacher (PAD Sec. 3.4).
* ``do_inference``    -- R1 / mAP evaluation on a val loader.
"""

import logging
import os
import time
from datetime import timedelta

import torch
import torch.nn as nn
from torch.cuda import amp

from .losses import KDLosses, SupConLoss
from .utils import AverageMeter, EMATeacher, R1_mAP_eval


# ===========================================================================
# Stage 1: TA-Prompt warm-up
# ===========================================================================
def do_train_stage1(cfg, model, train_loader_stage1, optimizer, scheduler,
                    local_rank: int, teacher_model=None):
    logger = logging.getLogger("pad.train")
    logger.info("[Stage1] start")

    device = "cuda"
    epochs = cfg.SOLVER.STAGE1.MAX_EPOCHS
    log_period = cfg.SOLVER.STAGE1.LOG_PERIOD
    textkd_on = bool(cfg.TEXTKD.ENABLE) and (teacher_model is not None)

    if textkd_on:
        kd_text = KDLosses(
            lambda_feat=0.0,
            lambda_logit=float(cfg.TEXTKD.LAMBDA_LOGIT),
            temperature=float(cfg.TEXTKD.TEMP),
        )
        logit_scale = float(cfg.TEXTKD.LOGITS_SCALE)
    else:
        kd_text, logit_scale = None, 1.0

    model.to(local_rank)
    if torch.cuda.device_count() > 1:
        logger.info(f"[Stage1] using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    student_ref = model.module if hasattr(model, "module") else model

    total = sum(p.numel() for p in student_ref.parameters())
    trainable = sum(p.numel() for p in student_ref.parameters() if p.requires_grad)
    logger.info(f"[Stage1] trainable params: {trainable}/{total} ({trainable/max(total,1):.2%})")

    loss_meter = AverageMeter()
    scaler = amp.GradScaler()
    supcon = SupConLoss(device)

    t_start = time.monotonic()

    # Cache image features once (encoder is frozen during Stage 1).
    logger.info("[Stage1] caching image features ...")
    was_training = model.training
    model.eval()
    feats_all, labels_all = [], []
    with torch.no_grad():
        for img, vid, _, _ in train_loader_stage1:
            img = img.to(device, non_blocking=True)
            vid = vid.to(device, non_blocking=True)
            with amp.autocast(enabled=True):
                feats_all.append(model(img, vid, get_image=True).detach().float().cpu())
            labels_all.append(vid.detach().cpu())
    feats_all = torch.cat(feats_all, dim=0).cuda(non_blocking=True).float()
    labels_all = torch.cat(labels_all, dim=0).cuda(non_blocking=True)
    N = labels_all.size(0)
    if was_training:
        model.train()

    batch = cfg.SOLVER.STAGE1.IMS_PER_BATCH
    iters_per_epoch = (N + batch - 1) // batch
    logger.info(f"[Stage1] cached {N} samples, {iters_per_epoch} iters/epoch")

    # Frozen text-anchor bank for TEXKD.
    Te_all = None
    if textkd_on:
        teacher = teacher_model.to(device).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        C = int(student_ref.prompt_learner.num_class)
        with torch.no_grad(), amp.autocast(enabled=True):
            Te_all = teacher(
                label=torch.arange(C, device=device, dtype=torch.long),
                get_text=True,
            ).detach().float()
        Te_all = Te_all / Te_all.norm(p=2, dim=1, keepdim=True).clamp_min(1e-6)

    neg_K = int(cfg.TEXTKD.NEG_PER_BATCH)

    for epoch in range(1, epochs + 1):
        loss_meter.reset()
        scheduler.step(epoch)
        model.train()
        kd_running = None

        idx_all = torch.randperm(N, device=device)
        for it in range(iters_per_epoch):
            s, e = it * batch, min((it + 1) * batch, N)
            if s >= e:
                continue
            idx = idx_all[s:e]
            cur_feats = feats_all[idx]
            cur_labels = labels_all[idx]

            optimizer.zero_grad(set_to_none=True)
            with amp.autocast(enabled=True):
                text_feats = model(label=cur_labels, get_text=True).float()
                loss = (
                    supcon(cur_feats, text_feats, cur_labels, cur_labels)
                    + supcon(text_feats, cur_feats, cur_labels, cur_labels)
                )

            if textkd_on:
                with torch.no_grad():
                    uniq_pos, _ = torch.unique(cur_labels, sorted=False, return_inverse=True)
                    C = Te_all.size(0)
                    mask = torch.ones(C, dtype=torch.bool, device=Te_all.device)
                    mask[uniq_pos] = False
                    neg_pool = torch.nonzero(mask, as_tuple=False).squeeze(1)
                    if neg_K > 0 and neg_pool.numel() > 0:
                        if neg_pool.numel() <= neg_K:
                            neg_idx = neg_pool
                        else:
                            perm = torch.randperm(neg_pool.numel(), device=neg_pool.device)
                            neg_idx = neg_pool[perm[:neg_K]]
                        cls_idx = torch.cat([uniq_pos, neg_idx], dim=0)
                    else:
                        cls_idx = uniq_pos

                with amp.autocast(enabled=True):
                    Ts_sub = model(label=cls_idx, get_text=True)
                Ts_sub = Ts_sub.float()
                Ts_sub = Ts_sub / Ts_sub.norm(p=2, dim=1, keepdim=True).clamp_min(1e-6)
                Te_sub = Te_all.index_select(0, cls_idx)
                S = cur_feats / cur_feats.norm(p=2, dim=1, keepdim=True).clamp_min(1e-6)

                logits_s = (S @ Ts_sub.t()) * logit_scale
                logits_t = (S @ Te_sub.t()) * logit_scale
                kl = kd_text.kl(logits_s, logits_t)
                loss = loss + kd_text.lambda_logit * kl
                with torch.no_grad():
                    v = float(kl.item())
                    kd_running = v if kd_running is None else 0.9 * kd_running + 0.1 * v

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), cur_feats.size(0))
            if (it + 1) % log_period == 0:
                lr = optimizer.param_groups[0]["lr"]
                msg = (f"Stage1 Epoch[{epoch}] Iter[{it+1}/{iters_per_epoch}] "
                       f"Loss {loss_meter.avg:.3f} LR {lr:.2e}")
                if textkd_on and kd_running is not None:
                    msg += f" TEXKD {kd_running:.3f}"
                logger.info(msg)

    save_path = os.path.join(cfg.OUTPUT_DIR, f"{cfg.MODEL.NAME}_stage1.pth")
    torch.save(model.state_dict(), save_path)
    logger.info(f"[Stage1] saved -> {save_path}")
    logger.info(f"[Stage1] total time {timedelta(seconds=time.monotonic() - t_start)}")


# ===========================================================================
# Stage 2: visual branch training with EMA-based VISKD
# ===========================================================================
def do_train_stage2(cfg, model, train_loader_stage2, val_loader,
                    optimizer, scheduler, loss_fn, num_query,
                    local_rank: int, teacher_model=None):
    logger = logging.getLogger("pad.train")
    logger.info("[Stage2] start")

    device = "cuda"
    epochs = cfg.SOLVER.STAGE2.MAX_EPOCHS
    log_period = cfg.SOLVER.STAGE2.LOG_PERIOD

    model.to(local_rank)
    if torch.cuda.device_count() > 1:
        logger.info(f"[Stage2] using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
        student_ref = model.module
    else:
        student_ref = model
    num_classes = student_ref.num_classes

    distill_on = bool(cfg.DISTILL.ENABLE) and (teacher_model is not None)
    ema_teacher = None
    kd_criterion = None
    if distill_on:
        ema_teacher = EMATeacher(
            teacher_model.to(device),
            momentum=float(cfg.DISTILL.EMA_MOMENTUM),
            device=torch.device(device),
        )
        kd_criterion = KDLosses(
            lambda_feat=float(cfg.DISTILL.LAMBDA_FEAT),
            lambda_logit=float(cfg.DISTILL.LAMBDA_LOGIT),
            temperature=float(cfg.DISTILL.TEMP),
        )

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    scaler = amp.GradScaler()

    t_start = time.monotonic()

    # Pre-compute the per-class text bank once with the trained TA-Prompt.
    batch = cfg.SOLVER.STAGE2.IMS_PER_BATCH
    text_bank = []
    with torch.no_grad():
        for s in range(0, num_classes, batch):
            e = min(s + batch, num_classes)
            with amp.autocast(enabled=True):
                text_bank.append(
                    model(label=torch.arange(s, e, device=device), get_text=True).cpu()
                )
    text_bank = torch.cat(text_bank, dim=0).to(device).float()

    for epoch in range(1, epochs + 1):
        t_epoch = time.time()
        loss_meter.reset()
        acc_meter.reset()
        scheduler.step()
        model.train()

        for n_iter, (img, vid, _, _) in enumerate(train_loader_stage2):
            img = img.to(device, non_blocking=True)
            target = vid.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with amp.autocast(enabled=True):
                score, feat, image_features = model(x=img, label=target)
                logits = image_features @ text_bank.t()
                loss = loss_fn(score, feat, target, logits)

                if distill_on:
                    with torch.no_grad():
                        _, t_feat, t_image_features = ema_teacher(
                            x=img, label=target, return_train_outputs=True,
                        )
                        t_logits = t_image_features @ text_bank.t()
                    kd_total, kd_feat_val, kd_kl_val = kd_criterion(
                        student_feats=feat, teacher_feats=t_feat,
                        student_logits=logits, teacher_logits=t_logits,
                    )
                    loss = loss + kd_total
                else:
                    kd_feat_val = kd_kl_val = None

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if distill_on:
                ema_teacher.update(student_ref)

            acc = (logits.max(1)[1] == target).float().mean().item()
            loss_meter.update(loss.item(), img.size(0))
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                lr = optimizer.param_groups[0]["lr"]
                msg = (f"Stage2 Epoch[{epoch}] Iter[{n_iter+1}/{len(train_loader_stage2)}] "
                       f"Loss {loss_meter.avg:.3f} Acc {acc_meter.avg:.3f} LR {lr:.2e}")
                if distill_on and kd_feat_val is not None:
                    msg += f" KD_feat {kd_feat_val.item():.3f} KD_kl {kd_kl_val.item():.3f}"
                logger.info(msg)

        dt = (time.time() - t_epoch) / max(n_iter + 1, 1)
        logger.info(
            f"[Stage2] epoch {epoch} done, time/batch {dt:.3f}s, "
            f"speed {train_loader_stage2.batch_size / max(dt, 1e-6):.1f} samples/s"
        )

    save_path = os.path.join(cfg.OUTPUT_DIR, f"{cfg.MODEL.NAME}_stage2.pth")
    torch.save(model.state_dict(), save_path)
    logger.info(f"[Stage2] saved -> {save_path}")
    logger.info(f"[Stage2] total time {timedelta(seconds=time.monotonic() - t_start)}")


# ===========================================================================
# inference
# ===========================================================================
def do_inference(cfg, model, val_loader, num_query: int):
    logger = logging.getLogger("pad.test")
    device = "cuda"
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator.reset()

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device).eval()

    for img, pid, camid, _, _, _ in val_loader:
        with torch.no_grad():
            evaluator.update((model(img.to(device)), pid, camid))

    cmc, mAP = evaluator.compute()
    logger.info(f"[Eval] mAP={mAP:.1%}")
    for r in (1, 5, 10):
        logger.info(f"[Eval] Rank-{r}: {cmc[r-1]:.1%}")
    return cmc[0], cmc[4], mAP
