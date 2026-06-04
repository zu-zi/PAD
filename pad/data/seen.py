"""Seen-domain datasets for the AKA benchmark."""

import glob
import os.path as osp
import re

from .bases import BaseImageDataset


class Market1501(BaseImageDataset):
    """Standard Market-1501 (bounding_box_train / query / bounding_box_test)."""

    dataset_dir = "market1501"

    def __init__(self, root: str = "", verbose: bool = True, pid_begin: int = 0, **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, "bounding_box_train")
        self.query_dir = osp.join(self.dataset_dir, "query")
        self.gallery_dir = osp.join(self.dataset_dir, "bounding_box_test")
        self.pid_begin = pid_begin
        for d in (self.dataset_dir, self.train_dir, self.query_dir, self.gallery_dir):
            if not osp.exists(d):
                raise RuntimeError(f"'{d}' is not available")

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)
        self._finalize(train, query, gallery, "Market-1501", verbose)

    def _process_dir(self, dir_path: str, relabel: bool = False):
        img_paths = sorted(glob.glob(osp.join(dir_path, "*.jpg")))
        pattern = re.compile(r"([-\d]+)_c(\d)")
        pids = {int(pattern.search(p).group(1)) for p in img_paths
                if int(pattern.search(p).group(1)) != -1}
        pid2label = {pid: lab for lab, pid in enumerate(sorted(pids))}
        data = []
        for p in img_paths:
            pid, camid = map(int, pattern.search(p).groups())
            if pid == -1:
                continue
            assert 0 <= pid <= 1501 and 1 <= camid <= 6
            camid -= 1
            if relabel:
                pid = pid2label[pid]
            data.append((p, self.pid_begin + pid, camid, 0))
        return data


class DukeMTMCreID(BaseImageDataset):
    """Standard Duke layout (bounding_box_train / query / bounding_box_test)."""

    dataset_dir = "dukemtmcreid"

    def __init__(self, root: str = "", verbose: bool = True, pid_begin: int = 0, **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, "bounding_box_train")
        self.query_dir = osp.join(self.dataset_dir, "query")
        self.gallery_dir = osp.join(self.dataset_dir, "bounding_box_test")
        self.pid_begin = pid_begin
        for d in (self.dataset_dir, self.train_dir, self.query_dir, self.gallery_dir):
            if not osp.exists(d):
                raise RuntimeError(f"'{d}' is not available")

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)
        self._finalize(train, query, gallery, "DukeMTMC-reID", verbose)

    def _process_dir(self, dir_path: str, relabel: bool = False):
        img_paths = sorted(glob.glob(osp.join(dir_path, "*.jpg")))
        pattern = re.compile(r"([-\d]+)_c(\d)")
        pids = {int(pattern.search(p).group(1)) for p in img_paths}
        pid2label = {pid: lab for lab, pid in enumerate(sorted(pids))}
        data = []
        for p in img_paths:
            pid, camid = map(int, pattern.search(p).groups())
            assert 1 <= camid <= 8
            camid -= 1
            if relabel:
                pid = pid2label[pid]
            data.append((p, self.pid_begin + pid, camid, 0))
        return data


class MSMT17(BaseImageDataset):
    """MSMT17 (train / test folders + list_*.txt annotation files)."""

    dataset_dir = "msmt17"

    def __init__(self, root: str = "", verbose: bool = True, pid_begin: int = 0, **kwargs):
        self.pid_begin = pid_begin
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, "train")
        self.test_dir = osp.join(self.dataset_dir, "test")
        lt = osp.join(self.dataset_dir, "list_train.txt")
        lv = osp.join(self.dataset_dir, "list_val.txt")
        lq = osp.join(self.dataset_dir, "list_query.txt")
        lg = osp.join(self.dataset_dir, "list_gallery.txt")
        for d in (self.dataset_dir, self.train_dir, self.test_dir):
            if not osp.exists(d):
                raise RuntimeError(f"'{d}' is not available")

        train = self._process_dir(self.train_dir, lt) + self._process_dir(self.train_dir, lv)
        query = self._process_dir(self.test_dir, lq)
        gallery = self._process_dir(self.test_dir, lg)
        self._finalize(train, query, gallery, "MSMT17", verbose)

    def _process_dir(self, dir_path: str, list_path: str):
        data = []
        with open(list_path, "r") as f:
            for line in f:
                img_name, pid = line.strip().split(" ")
                pid = int(pid)
                camid = int(img_name.split("_")[2]) - 1
                data.append((osp.join(dir_path, img_name), self.pid_begin + pid, camid, 0))
        return data


class CUHKSYSU(BaseImageDataset):
    """Filenames follow ``{pid:04d}_c{camid}_{idx:05d}.jpg`` (ReID-style conversion)."""

    dataset_dir = "cuhksysu"

    def __init__(self, root: str = "", verbose: bool = True, pid_begin: int = 0, **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, "bounding_box_train")
        self.query_dir = osp.join(self.dataset_dir, "query")
        self.gallery_dir = osp.join(self.dataset_dir, "bounding_box_test")
        self.pid_begin = pid_begin
        for d in (self.dataset_dir, self.train_dir, self.query_dir, self.gallery_dir):
            if not osp.exists(d):
                raise RuntimeError(f"'{d}' is not available")

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)
        self._finalize(train, query, gallery, "CUHK-SYSU", verbose)

    def _process_dir(self, dir_path: str, relabel: bool = False):
        img_paths = sorted(glob.glob(osp.join(dir_path, "*.jpg")))
        pattern = re.compile(r"([-\d]+)_c(\d+)")
        pids = set()
        for p in img_paths:
            m = pattern.search(osp.basename(p))
            if m is not None and int(m.group(1)) != -1:
                pids.add(int(m.group(1)))
        pid2label = {pid: lab for lab, pid in enumerate(sorted(pids))}
        data = []
        for p in img_paths:
            m = pattern.search(osp.basename(p))
            if m is None:
                continue
            pid, camid = map(int, m.groups())
            if pid == -1:
                continue
            assert camid >= 1
            camid -= 1
            if relabel:
                pid = pid2label[pid]
            data.append((p, self.pid_begin + pid, camid, 0))
        return data


class CUHK03(BaseImageDataset):
    """CUHK03 new-protocol (detected) layout; camid in [1, 10]."""

    dataset_dir = "cuhk03"

    def __init__(self, root: str = "", verbose: bool = True, pid_begin: int = 0, **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, "bounding_box_train")
        self.query_dir = osp.join(self.dataset_dir, "query")
        self.gallery_dir = osp.join(self.dataset_dir, "bounding_box_test")
        self.pid_begin = pid_begin
        for d in (self.dataset_dir, self.train_dir, self.query_dir, self.gallery_dir):
            if not osp.exists(d):
                raise RuntimeError(f"'{d}' is not available")

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)
        self._finalize(train, query, gallery, "CUHK03", verbose)

    def _process_dir(self, dir_path: str, relabel: bool = False):
        img_paths = sorted(glob.glob(osp.join(dir_path, "*.jpg")))
        pattern = re.compile(r"([-\d]+)_c(\d+)_")
        pids = set()
        for p in img_paths:
            m = pattern.search(osp.basename(p))
            if m is not None:
                pids.add(int(m.group(1)))
        pid2label = {pid: lab for lab, pid in enumerate(sorted(pids))}
        data = []
        for p in img_paths:
            m = pattern.search(osp.basename(p))
            if m is None:
                continue
            pid, camid = map(int, m.groups())
            assert 1 <= camid <= 10, f"unexpected camid {camid} in {p}"
            camid -= 1
            if relabel:
                pid = pid2label[pid]
            data.append((p, self.pid_begin + pid, camid, 0))
        return data


class LPW_S2(BaseImageDataset):
    """LPW-s2 (pre-converted to Market-style ``<pid>_c<camid>s<sid>_*.jpg``)."""

    dataset_dir = "LPW_s2"

    def __init__(self, root: str = "", verbose: bool = True, pid_begin: int = 0, **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, "bounding_box_train")
        self.query_dir = osp.join(self.dataset_dir, "query")
        self.gallery_dir = osp.join(self.dataset_dir, "bounding_box_test")
        self.pid_begin = pid_begin
        for d in (self.dataset_dir, self.train_dir, self.query_dir, self.gallery_dir):
            if not osp.exists(d):
                raise RuntimeError(f"'{d}' is not available")

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)
        self._finalize(train, query, gallery, "LPW-s2", verbose)

    def _process_dir(self, dir_path: str, relabel: bool = False):
        img_paths = sorted(glob.glob(osp.join(dir_path, "*.jpg")))
        pattern = re.compile(r"([-\d]+)_c(\d+)s(\d+)")
        pids = set()
        for p in img_paths:
            m = pattern.search(osp.basename(p))
            if m is not None and int(m.group(1)) != -1:
                pids.add(int(m.group(1)))
        pid2label = {pid: lab for lab, pid in enumerate(sorted(pids))}
        data = []
        for p in img_paths:
            m = pattern.search(osp.basename(p))
            if m is None:
                continue
            pid, camid, _ = map(int, m.groups())
            if pid == -1:
                continue
            camid -= 1
            if relabel:
                pid = pid2label[pid]
            data.append((p, self.pid_begin + pid, camid, 0))
        return data
