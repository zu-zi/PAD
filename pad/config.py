"""
Default config + YAML loader for PAD.

The single ``configs/pad.yml`` file stores both the shared
hyper-parameters (at the top level) and a ``DOMAINS`` list of per-domain
overrides.  ``load_cfg_from_yaml(cfg, path, domain_idx)`` merges the
shared part first, then applies the i-th domain's overrides.
"""

import os
from typing import Any, Dict, Optional

import yaml
from yacs.config import CfgNode as CN


# -------------------- default config ----------------------------------------
_C = CN()

# Model.
_C.MODEL = CN()
_C.MODEL.NAME = "ViT-B-16"
_C.MODEL.STRIDE_SIZE = [16, 16]
_C.MODEL.ID_LOSS_WEIGHT = 0.25
_C.MODEL.TRIPLET_LOSS_WEIGHT = 1.0
_C.MODEL.I2T_LOSS_WEIGHT = 1.0
_C.MODEL.LABEL_SMOOTH = True
_C.MODEL.DIST_TRAIN = False

# Input pipeline.
_C.INPUT = CN()
_C.INPUT.SIZE_TRAIN = [256, 128]
_C.INPUT.SIZE_TEST = [256, 128]
_C.INPUT.PROB = 0.5
_C.INPUT.RE_PROB = 0.5
_C.INPUT.PADDING = 10
_C.INPUT.PIXEL_MEAN = [0.5, 0.5, 0.5]
_C.INPUT.PIXEL_STD = [0.5, 0.5, 0.5]

# Datasets / dataloader.
_C.DATASETS = CN()
_C.DATASETS.NAMES = ("market1501",)
_C.DATASETS.ROOT_DIR = "data"

_C.DATALOADER = CN()
_C.DATALOADER.SAMPLER = "softmax_triplet"
_C.DATALOADER.NUM_INSTANCE = 4
_C.DATALOADER.NUM_WORKERS = 8

# TA-Prompt (PAD Sec. 3.3).
_C.TA_PROMPT = CN()
_C.TA_PROMPT.N_CTX = 4
_C.TA_PROMPT.N_CLS_CTX = 4

# VA-Prompt pool (PAD Sec. 3.4.2).
_C.VA_PROMPT = CN()
_C.VA_PROMPT.ENABLE = True
_C.VA_PROMPT.G_LEN = 6
_C.VA_PROMPT.E_LEN = 6
_C.VA_PROMPT.POOL_SIZE = 36
_C.VA_PROMPT.TOP_K = 4
_C.VA_PROMPT.EMBEDDING_KEY = "mean"
_C.VA_PROMPT.KEY_LEARNABLE = True
_C.VA_PROMPT.STRIP_AFTER_BLOCK = True
_C.VA_PROMPT.NEW_SLOTS_PER_DOMAIN = 8

# Selective layer unfreezing (PAD Sec. 3.4.3).
_C.UNFREEZE = CN()
_C.UNFREEZE.LAST_BLOCKS = 4

# TEXKD (PAD Sec. 3.3.2).
_C.TEXTKD = CN()
_C.TEXTKD.ENABLE = False
_C.TEXTKD.LAMBDA_LOGIT = 0.5
_C.TEXTKD.TEMP = 0.07
_C.TEXTKD.LOGITS_SCALE = 7.0
_C.TEXTKD.NEG_PER_BATCH = 256

# VISKD (PAD Sec. 3.4.1) with EMA teacher.
_C.DISTILL = CN()
_C.DISTILL.ENABLE = False
_C.DISTILL.LAMBDA_FEAT = 0.5
_C.DISTILL.LAMBDA_LOGIT = 0.5
_C.DISTILL.TEMP = 4.0
_C.DISTILL.EMA_MOMENTUM = 0.997

# Solver.
_C.SOLVER = CN()
_C.SOLVER.SEED = 42
_C.SOLVER.MARGIN = 0.3

_C.SOLVER.STAGE1 = CN()
_C.SOLVER.STAGE1.IMS_PER_BATCH = 64
_C.SOLVER.STAGE1.OPTIMIZER = "Adam"
_C.SOLVER.STAGE1.BASE_LR = 3.5e-4
_C.SOLVER.STAGE1.WEIGHT_DECAY = 1e-4
_C.SOLVER.STAGE1.MAX_EPOCHS = 12
_C.SOLVER.STAGE1.LOG_PERIOD = 50
_C.SOLVER.STAGE1.WARMUP_EPOCHS = 3
_C.SOLVER.STAGE1.WARMUP_LR_INIT = 1e-5
_C.SOLVER.STAGE1.LR_MIN = 1e-6

_C.SOLVER.STAGE2 = CN()
_C.SOLVER.STAGE2.IMS_PER_BATCH = 64
_C.SOLVER.STAGE2.OPTIMIZER = "Adam"
_C.SOLVER.STAGE2.BASE_LR = 5e-6
_C.SOLVER.STAGE2.WEIGHT_DECAY = 1e-4
_C.SOLVER.STAGE2.MAX_EPOCHS = 25
_C.SOLVER.STAGE2.LOG_PERIOD = 50
_C.SOLVER.STAGE2.STEPS = [14, 22]
_C.SOLVER.STAGE2.GAMMA = 0.1
_C.SOLVER.STAGE2.WARMUP_ITERS = 10
_C.SOLVER.STAGE2.WARMUP_FACTOR = 0.1
_C.SOLVER.STAGE2.WARMUP_METHOD = "linear"

# Prompt-specific optimiser overrides.
_C.OPT = CN()
_C.OPT.PROMPT_LR = 5e-3
_C.OPT.PROMPT_WD = 0.0

# Test.
_C.TEST = CN()
_C.TEST.IMS_PER_BATCH = 128
_C.TEST.FEAT_NORM = "yes"
_C.TEST.NECK_FEAT = "before"

# Unseen-domain evaluation protocol (see pad/data/unseen.py). A bare
# call uses split 0; scripts/eval_unseen.sh loops over the splits used
# to report Table 1's Unseen-Avg and averages the results.
_C.EVAL = CN()
_C.EVAL.SPLIT_ID = 0
_C.EVAL.SEED = 123
_C.EVAL.DIRECTION = "auto"   # "auto" | "A2B" | "B2A"

_C.OUTPUT_DIR = ""

cfg = _C  # public alias


# -------------------- YAML loader -------------------------------------------
def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def list_domains(path: str):
    """Return the ordered list of domain names declared under ``DOMAINS``."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    domains = raw.get("DOMAINS") or []
    return [d["name"] for d in domains]


def load_cfg_from_yaml(cfg: CN, path: str, domain_idx: Optional[int] = None,
                       domain_name: Optional[str] = None) -> CN:
    """Merge the shared block and (optionally) one domain's overrides into ``cfg``.

    Usage::

        from pad.config import cfg, load_cfg_from_yaml
        cfg.defrost()
        load_cfg_from_yaml(cfg, "configs/pad.yml", domain_idx=0)
        cfg.freeze()
    """
    if not path:
        return cfg
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    domains = raw.pop("DOMAINS", []) or []

    # Start with the shared block.
    merged = dict(raw)

    # Select the requested domain (by index or by name).
    if domain_idx is not None or domain_name is not None:
        if domain_idx is not None:
            if not (0 <= domain_idx < len(domains)):
                raise IndexError(
                    f"domain_idx={domain_idx} out of range [0, {len(domains)-1}]"
                )
            domain_cfg = dict(domains[domain_idx])
        else:
            match = [d for d in domains if d.get("name") == domain_name]
            if not match:
                raise ValueError(f"domain '{domain_name}' not found in {path}")
            domain_cfg = dict(match[0])

        domain_cfg.pop("name", None)
        ds_name = domain_cfg.pop("dataset", None)
        if ds_name is not None:
            domain_cfg.setdefault("DATASETS", {})["NAMES"] = (ds_name,)
        merged = _deep_merge(merged, domain_cfg)

    cfg.merge_from_other_cfg(CN(merged))
    return cfg
