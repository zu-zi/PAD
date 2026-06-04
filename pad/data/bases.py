"""Shared base for image-based ReID datasets."""

import os.path as osp

from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_image(img_path: str) -> Image.Image:
    if not osp.exists(img_path):
        raise IOError(f"{img_path} does not exist")
    while True:
        try:
            return Image.open(img_path).convert("RGB")
        except IOError:
            print(f"[read_image] retrying: {img_path}")


class BaseImageDataset:
    """Common statistics helpers for image-based ReID datasets."""

    def get_imagedata_info(self, data):
        pids, cams, tracks = set(), set(), set()
        for _, pid, camid, trackid in data:
            pids.add(pid); cams.add(camid); tracks.add(trackid)
        return len(pids), len(data), len(cams), len(tracks)

    def print_dataset_statistics(self, train, query, gallery):
        n_tp, n_ti, n_tc, _ = self.get_imagedata_info(train)
        n_qp, n_qi, n_qc, _ = self.get_imagedata_info(query)
        n_gp, n_gi, n_gc, _ = self.get_imagedata_info(gallery)
        print("  ------------------------------------------")
        print("  subset  | # ids | # images | # cameras")
        print("  ------------------------------------------")
        print(f"  train   | {n_tp:5d} | {n_ti:8d} | {n_tc:9d}")
        print(f"  query   | {n_qp:5d} | {n_qi:8d} | {n_qc:9d}")
        print(f"  gallery | {n_gp:5d} | {n_gi:8d} | {n_gc:9d}")
        print("  ------------------------------------------")

    def _finalize(self, train, query, gallery, tag: str, verbose: bool = True):
        self.train = train
        self.query = query
        self.gallery = gallery
        if verbose:
            print(f"=> {tag} loaded")
            self.print_dataset_statistics(train, query, gallery)
        (self.num_train_pids, self.num_train_imgs,
         self.num_train_cams, self.num_train_vids) = self.get_imagedata_info(train)
        self.num_query_pids, _, _, _ = self.get_imagedata_info(query)
        self.num_gallery_pids, _, _, _ = self.get_imagedata_info(gallery)
        # Unseen loaders have no training split; clamp so PADModel can still
        # be constructed safely at inference time.
        if self.num_train_pids <= 0:
            self.num_train_pids = 1


class ImageDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path, pid, camid, trackid = self.data[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, pid, camid, trackid, img_path.split("/")[-1]
