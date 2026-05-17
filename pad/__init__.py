"""PAD: Prompt-Anchored Vision-Text Distillation for Lifelong Person Re-id."""

from .config import cfg, load_cfg_from_yaml, list_domains
from .data import make_dataloader
from .engine import do_inference, do_train_stage1, do_train_stage2
from .losses import KDLosses, SupConLoss, TripletLoss, make_loss
from .model import PADModel, TAPromptLearner, VAPromptPool, make_model
from .optim import WarmupMultiStepLR, make_cosine_scheduler, make_optimizer
from .utils import AverageMeter, EMATeacher, R1_mAP_eval, setup_logger

__all__ = [
    # config
    "cfg", "load_cfg_from_yaml", "list_domains",
    # data
    "make_dataloader",
    # engine
    "do_train_stage1", "do_train_stage2", "do_inference",
    # losses
    "KDLosses", "SupConLoss", "TripletLoss", "make_loss",
    # model
    "PADModel", "TAPromptLearner", "VAPromptPool", "make_model",
    # optim
    "WarmupMultiStepLR", "make_cosine_scheduler", "make_optimizer",
    # utils
    "AverageMeter", "EMATeacher", "R1_mAP_eval", "setup_logger",
]
