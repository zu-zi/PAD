"""DataLoader factory for PAD."""

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from timm.data.random_erasing import RandomErasing

from .bases import ImageDataset
from .sampler import RandomIdentitySampler
from .seen import CUHK03, CUHKSYSU, DukeMTMCreID, LPW_S2, MSMT17, Market1501
from .unseen import CUHK01, CUHK02, GRID, ILIDS, PRID, SenseReID, VIPeR


_FACTORY = {
    # seen-domain AKA benchmark
    "market1501":   Market1501,
    "dukemtmcreid": DukeMTMCreID,
    "msmt17":       MSMT17,
    "cuhksysu":     CUHKSYSU,
    "cuhk03":       CUHK03,
    "lpw_s2":       LPW_S2,
    # unseen-domain generalisation
    "cuhk01":       CUHK01,
    "cuhk02":       CUHK02,
    "grid":         GRID,
    "ilids":        ILIDS,
    "prid":         PRID,
    "sensereid":    SenseReID,
    "viper":        VIPeR,
}


def _train_collate(batch):
    imgs, pids, camids, viewids, _ = zip(*batch)
    return (
        torch.stack(imgs, dim=0),
        torch.tensor(pids, dtype=torch.int64),
        torch.tensor(camids, dtype=torch.int64),
        torch.tensor(viewids, dtype=torch.int64),
    )


def _val_collate(batch):
    imgs, pids, camids, viewids, img_paths = zip(*batch)
    return (
        torch.stack(imgs, dim=0),
        pids,
        camids,
        torch.tensor(camids, dtype=torch.int64),
        torch.tensor(viewids, dtype=torch.int64),
        img_paths,
    )


def make_dataloader(cfg):
    train_tf = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
        T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
        T.Pad(cfg.INPUT.PADDING),
        T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        RandomErasing(probability=cfg.INPUT.RE_PROB, mode="pixel", max_count=1, device="cpu"),
    ])
    val_tf = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
    ])

    names = cfg.DATASETS.NAMES
    name = names[0] if isinstance(names, (list, tuple)) else names
    eval_kwargs = {}
    if hasattr(cfg, "EVAL"):
        eval_kwargs = dict(
            split_id=int(cfg.EVAL.SPLIT_ID),
            seed=int(cfg.EVAL.SEED),
            direction=str(cfg.EVAL.DIRECTION),
        )
    dataset = _FACTORY[name](root=cfg.DATASETS.ROOT_DIR, **eval_kwargs)

    num_workers = cfg.DATALOADER.NUM_WORKERS
    num_classes = dataset.num_train_pids
    train_is_empty = len(dataset.train) == 0

    train_set = ImageDataset(dataset.train, train_tf)
    train_set_normal = ImageDataset(dataset.train, val_tf)
    val_set = ImageDataset(dataset.query + dataset.gallery, val_tf)

    if train_is_empty:
        train_loader_stage2 = DataLoader(
            train_set, batch_size=1, shuffle=False,
            num_workers=num_workers, collate_fn=_train_collate,
        )
    elif cfg.DATALOADER.SAMPLER == "softmax_triplet":
        train_loader_stage2 = DataLoader(
            train_set, batch_size=cfg.SOLVER.STAGE2.IMS_PER_BATCH,
            sampler=RandomIdentitySampler(
                dataset.train, cfg.SOLVER.STAGE2.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE,
            ),
            num_workers=num_workers, collate_fn=_train_collate,
        )
    else:
        train_loader_stage2 = DataLoader(
            train_set, batch_size=cfg.SOLVER.STAGE2.IMS_PER_BATCH, shuffle=True,
            num_workers=num_workers, collate_fn=_train_collate,
        )

    if train_is_empty:
        train_loader_stage1 = DataLoader(
            train_set_normal, batch_size=1, shuffle=False,
            num_workers=num_workers, collate_fn=_train_collate,
        )
    else:
        train_loader_stage1 = DataLoader(
            train_set_normal, batch_size=cfg.SOLVER.STAGE1.IMS_PER_BATCH, shuffle=True,
            num_workers=num_workers, collate_fn=_train_collate,
        )

    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False,
        num_workers=num_workers, collate_fn=_val_collate,
    )

    return train_loader_stage2, train_loader_stage1, val_loader, len(dataset.query), num_classes
