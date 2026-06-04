"""Dataset loaders, identity sampler and dataloader factory."""

from .bases import BaseImageDataset, ImageDataset, read_image
from .loader import make_dataloader
from .sampler import RandomIdentitySampler
from .seen import CUHK03, CUHKSYSU, DukeMTMCreID, LPW_S2, MSMT17, Market1501
from .unseen import CUHK01, CUHK02, GRID, ILIDS, PRID, SenseReID, VIPeR

__all__ = [
    "make_dataloader",
    "ImageDataset", "BaseImageDataset", "read_image",
    "RandomIdentitySampler",
    # seen
    "Market1501", "DukeMTMCreID", "MSMT17", "CUHKSYSU", "CUHK03", "LPW_S2",
    # unseen
    "CUHK01", "CUHK02", "GRID", "ILIDS", "PRID", "SenseReID", "VIPeR",
]
