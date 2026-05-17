"""Evaluate a PAD checkpoint on one or more seen / unseen datasets."""

import argparse
import csv
import os

import torch

from pad import (
    cfg, do_inference, load_cfg_from_yaml, make_dataloader,
    make_model, setup_logger,
)


def _load_visual_state_dict(model, ckpt_path: str, logger=None):
    """Load only the visual branch + VA-Prompt + BN necks from ``ckpt_path``.

    The previous-domain classifier / TA-Prompt are skipped because the
    identity space differs across domains.
    """
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}

    kept = {}
    for k, v in sd.items():
        lk = k.lower()
        if lk.startswith("image_encoder"):
            kept[k] = v
        elif "bottleneck" in lk:
            kept[k] = v
        elif (("ln_post" in lk) and ("text" not in lk)) or (lk.endswith(".proj") and "text" not in lk):
            kept[k] = v

    # Guard against shape drift on the lifelong bookkeeping buffers.
    buffers = dict(model.named_buffers())
    pruned = {}
    for k, v in kept.items():
        if any(t in k for t in ("va_prompt.active_sizes",
                                "va_prompt.trainable_masks",
                                "va_prompt.frozen_masks")):
            cur = buffers.get(k)
            if cur is not None and tuple(cur.shape) != tuple(v.shape):
                continue
        pruned[k] = v

    ret = model.load_state_dict(pruned, strict=False)
    if logger is not None:
        logger.info(
            f"[Load] {os.path.basename(ckpt_path)}: kept={len(pruned)} "
            f"missing={len(ret.missing_keys)} unexpected={len(ret.unexpected_keys)}"
        )

    # Reconstruct VA-Prompt slot usage from expert weights if the
    # bookkeeping buffers were filtered out above.
    if hasattr(model.image_encoder, "va_prompt"):
        va = model.image_encoder.va_prompt
        with torch.no_grad():
            for li in range(va.num_layers):
                used = int((va.e_prompt[li].detach().abs().sum(dim=(1, 2)) > 1e-6).sum().item())
                va.active_sizes[li, 0] = min(used, va.pool_size)


def _eval_one_domain(domain_name: str, ckpt: str, trained_domain: str, logger):
    cfg.defrost()
    cfg.DATASETS.NAMES = (domain_name,)
    cfg.freeze()

    _, _, val_loader, num_query, num_classes = make_dataloader(cfg)
    model = make_model(cfg, num_class=num_classes)

    if ckpt and os.path.isfile(ckpt):
        _load_visual_state_dict(model, ckpt, logger)
    else:
        logger.warning("[test] no valid checkpoint; using randomly initialised weights (debug)")

    rank1, rank5, mAP = do_inference(cfg, model, val_loader, num_query)
    logger.info(
        f"[Eval] trained={trained_domain} eval={domain_name} "
        f"Rank-1={rank1:.4f} Rank-5={rank5:.4f} mAP={mAP:.4f}"
    )
    return {
        "trained_domain": trained_domain or "",
        "eval_domain": domain_name,
        "rank1": f"{rank1:.4f}",
        "rank5": f"{rank5:.4f}",
        "map":   f"{mAP:.4f}",
        "ckpt":  os.path.basename(ckpt) if ckpt else "",
    }


def main():
    parser = argparse.ArgumentParser(description="PAD evaluation")
    parser.add_argument("--config_file", default="configs/pad.yml", type=str)
    parser.add_argument("--domain_idx", type=int, default=None,
                        help="load shared+this domain's config from DOMAINS")
    parser.add_argument("--domain_name", type=str, default=None)
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--eval_domains", type=str, required=True,
                        help="comma-separated domain names to evaluate")
    parser.add_argument("--trained_domain", type=str, default="")
    parser.add_argument("--csv_out", type=str, default="")
    parser.add_argument("--outdir", type=str, default="")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    load_cfg_from_yaml(cfg, args.config_file,
                       domain_idx=args.domain_idx, domain_name=args.domain_name)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = args.outdir or cfg.OUTPUT_DIR
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    logger = setup_logger("pad", output_dir, if_train=False)

    domains = [d.strip() for d in args.eval_domains.split(",") if d.strip()]
    rows = [_eval_one_domain(d, args.ckpt, args.trained_domain, logger) for d in domains]

    if args.csv_out:
        write_header = not os.path.exists(args.csv_out)
        with open(args.csv_out, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["trained_domain", "eval_domain",
                                               "rank1", "rank5", "map", "ckpt"])
            if write_header:
                w.writeheader()
            for r in rows:
                w.writerow(r)
        logger.info(f"[test] wrote results to {args.csv_out}")


if __name__ == "__main__":
    main()
