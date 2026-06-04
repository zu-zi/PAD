"""Unseen-domain loaders used for cross-dataset generalisation evaluation.

All loaders follow the multi-sample protocol commonly used in recent
lifelong-ReID papers (LSTKC / DKP / DKUA): cam-A images form the query
set and cam-B images form the gallery.
"""

import glob
import os.path as osp
from collections import defaultdict

from .bases import BaseImageDataset


class CUHK01(BaseImageDataset):
    """``<root>/cuhk01/campus/*.png`` with ``PPPPCCCC.png`` names."""

    dataset_dir = "cuhk01"

    def __init__(self, root: str = "", verbose: bool = True, **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.campus_dir = osp.join(self.dataset_dir, "campus")
        for d in (self.dataset_dir, self.campus_dir):
            if not osp.exists(d):
                raise RuntimeError(f"'{d}' is not available")

        paths = sorted(glob.glob(osp.join(self.campus_dir, "*.png")))
        if not paths:
            raise RuntimeError(f"no images under {self.campus_dir}")

        query, gallery = [], []
        for p in paths:
            fname = osp.basename(p)
            try:
                pid = int(fname[:4])
                cam_code = int(fname[4:7]) if fname[4:7].isdigit() else int(fname[4:8])
            except Exception:
                continue
            if cam_code in (1, 2):
                query.append((p, pid, 0, 0))
            elif cam_code in (3, 4):
                gallery.append((p, pid, 1, 0))
        self._finalize([], query, gallery, "CUHK01 (unseen)", verbose)


class CUHK02(BaseImageDataset):
    """``<root>/cuhk02/Dataset/P5/{cam1,cam2}/*.png`` (P5 is the test split)."""

    dataset_dir = "cuhk02"

    def __init__(self, root: str = "", verbose: bool = True, **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.p5_cam1 = osp.join(self.dataset_dir, "Dataset", "P5", "cam1")
        self.p5_cam2 = osp.join(self.dataset_dir, "Dataset", "P5", "cam2")
        for d in (self.dataset_dir, self.p5_cam1, self.p5_cam2):
            if not osp.exists(d):
                raise RuntimeError(f"'{d}' is not available")

        query = self._collect(self.p5_cam1, camid=0)
        gallery = self._collect(self.p5_cam2, camid=1)
        self._finalize([], query, gallery, "CUHK02 (unseen)", verbose)

    @staticmethod
    def _parse_pid(fname: str) -> int:
        stem = osp.splitext(fname)[0]
        head = stem.split("_")[0]
        if head.isdigit():
            return int(head)
        digits = ""
        for ch in stem:
            if ch.isdigit():
                digits += ch
            else:
                break
        return int(digits) if digits else -1

    def _collect(self, dirpath: str, camid: int):
        items = []
        for p in sorted(glob.glob(osp.join(dirpath, "*.png"))):
            pid = self._parse_pid(osp.basename(p))
            if pid >= 0:
                items.append((p, pid, camid, 0))
        return items


class GRID(BaseImageDataset):
    """``<root>/grid/{probe, gallery}/*.jpeg`` with ``0000_*`` distractors."""

    dataset_dir = "grid"

    def __init__(self, root: str = "", verbose: bool = True, **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.probe_dir = osp.join(self.dataset_dir, "probe")
        self.gallery_dir = osp.join(self.dataset_dir, "gallery")
        for d in (self.dataset_dir, self.probe_dir, self.gallery_dir):
            if not osp.exists(d):
                raise RuntimeError(f"'{d}' is not available")

        query = self._collect(self.probe_dir, camid=0, allow_distractor=False)
        gallery = self._collect(self.gallery_dir, camid=1, allow_distractor=True)
        self._finalize([], query, gallery, "GRID (unseen)", verbose)

    @staticmethod
    def _parse_pid(fname: str, allow_distractor: bool) -> int:
        if len(fname) < 4 or not fname[:4].isdigit():
            return -1
        pid = int(fname[:4])
        if pid == 0:
            return 0 if allow_distractor else -1
        return pid

    def _collect(self, dirpath: str, camid: int, allow_distractor: bool):
        paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            paths.extend(glob.glob(osp.join(dirpath, ext)))
        paths = sorted(paths)
        if not paths:
            raise RuntimeError(f"no images under {dirpath}")
        items = []
        for p in paths:
            pid = self._parse_pid(osp.basename(p), allow_distractor)
            if pid >= 0:
                items.append((p, pid, camid, 0))
        return items


class ILIDS(BaseImageDataset):
    """``<root>/ilids/i-LIDS_Pedestrian/Persons/*.jpg`` with ``PPPPSSS.jpg`` names."""

    dataset_dir = "ilids"

    def __init__(self, root: str = "", verbose: bool = True, **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.persons_dir = osp.join(self.dataset_dir, "i-LIDS_Pedestrian", "Persons")
        for d in (self.dataset_dir, self.persons_dir):
            if not osp.exists(d):
                raise RuntimeError(f"'{d}' is not available")

        paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            paths.extend(glob.glob(osp.join(self.persons_dir, ext)))
        paths = sorted(paths)
        if not paths:
            raise RuntimeError(f"no images under {self.persons_dir}")

        by_pid = defaultdict(list)
        for p in paths:
            fname = osp.basename(p)
            if len(fname) < 7:
                continue
            try:
                pid = int(fname[:4])
                seq = int(fname[4:7])
            except Exception:
                continue
            by_pid[pid].append((seq, p))

        query, gallery = [], []
        for pid, items in by_pid.items():
            items.sort(key=lambda x: x[0])
            if len(items) < 2:
                continue
            seq_q, pth_q = items[0]
            query.append((pth_q, pid, max(0, seq_q - 1), 0))
            for seq_g, pth_g in items[1:]:
                gallery.append((pth_g, pid, max(0, seq_g - 1), 0))
        self._finalize([], query, gallery, "iLIDS (unseen)", verbose)


class PRID(BaseImageDataset):
    """``<root>/prid/single_shot/{cam_a, cam_b}/person_XXXX.png``."""

    dataset_dir = "prid"

    def __init__(self, root: str = "", verbose: bool = True, **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.cam_a_dir = osp.join(self.dataset_dir, "single_shot", "cam_a")
        self.cam_b_dir = osp.join(self.dataset_dir, "single_shot", "cam_b")
        for d in (self.dataset_dir, self.cam_a_dir, self.cam_b_dir):
            if not osp.exists(d):
                raise RuntimeError(f"'{d}' is not available")

        a_map = self._build_map(self.cam_a_dir)
        b_map = self._build_map(self.cam_b_dir)
        overlap = sorted(set(a_map) & set(b_map))

        query, gallery = [], []
        for pid in overlap:
            query.append((a_map[pid], pid, 0, 0))
            gallery.append((b_map[pid], pid, 1, 0))
        for pid in sorted(set(b_map) - set(overlap)):
            gallery.append((b_map[pid], pid, 1, 0))
        self._finalize([], query, gallery, "PRID (unseen)", verbose)

    @staticmethod
    def _parse_pid(fname: str):
        if "person_" not in fname:
            return None
        stem = fname.split("person_", 1)[1]
        digits = ""
        for ch in stem:
            if ch.isdigit():
                digits += ch
            else:
                break
        return int(digits) if digits else None

    def _build_map(self, dirpath: str):
        paths = []
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            paths.extend(glob.glob(osp.join(dirpath, ext)))
        m = {}
        for p in sorted(paths):
            pid = self._parse_pid(osp.basename(p))
            if pid is not None:
                m.setdefault(pid, p)
        if not m:
            raise RuntimeError(f"no valid images in {dirpath}")
        return m


class SenseReID(BaseImageDataset):
    """``<root>/sensereid/SenseReID/{test_probe, test_gallery}/<pid>_<camid>.jpg``."""

    dataset_dir = "sensereid"

    def __init__(self, root: str = "", verbose: bool = True, **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.query_dir = osp.join(self.dataset_dir, "SenseReID", "test_probe")
        self.gallery_dir = osp.join(self.dataset_dir, "SenseReID", "test_gallery")
        for d in (self.dataset_dir, self.query_dir, self.gallery_dir):
            if not osp.exists(d):
                raise RuntimeError(f"'{d}' is not available")

        self._finalize([], self._collect(self.query_dir), self._collect(self.gallery_dir),
                       "SenseReID (unseen)", verbose)

    @staticmethod
    def _parse(fname: str):
        stem = osp.splitext(fname)[0]
        if "_" not in stem:
            return None, None
        a, b = stem.split("_", 1)
        if not a.isdigit() or not b.isdigit():
            return None, None
        return int(a), int(b)

    def _collect(self, dirpath: str):
        paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            paths.extend(glob.glob(osp.join(dirpath, ext)))
        paths = sorted(paths)
        if not paths:
            raise RuntimeError(f"no images under {dirpath}")
        items = []
        for p in paths:
            pid, camid = self._parse(osp.basename(p))
            if pid is not None:
                items.append((p, pid, camid, 0))
        return items


class VIPeR(BaseImageDataset):
    """``<root>/viper/VIPeR/{cam_a, cam_b}/*.bmp``; i-th images share the same identity."""

    dataset_dir = "viper"

    def __init__(self, root: str = "", verbose: bool = True, **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.cam_a_dir = self._resolve(osp.join(self.dataset_dir, "VIPeR", "cam_a"))
        self.cam_b_dir = self._resolve(osp.join(self.dataset_dir, "VIPeR", "cam_b"))
        for d in (self.dataset_dir, self.cam_a_dir, self.cam_b_dir):
            if not osp.exists(d):
                raise RuntimeError(f"'{d}' is not available")

        a = self._glob(self.cam_a_dir)
        b = self._glob(self.cam_b_dir)
        if not a or not b or len(a) != len(b):
            raise RuntimeError(f"VIPeR: cam_a={len(a)} cam_b={len(b)} (must be equal and nonempty)")

        query, gallery = [], []
        for i in range(len(a)):
            pid = i + 1
            query.append((a[i], pid, 0, 0))
            gallery.append((b[i], pid, 1, 0))
        self._finalize([], query, gallery, "VIPeR (unseen)", verbose)

    @staticmethod
    def _resolve(base: str) -> str:
        maybe = osp.join(base, "images")
        return maybe if osp.isdir(maybe) else base

    @staticmethod
    def _glob(dirpath: str):
        paths = []
        for ext in ("*.bmp", "*.BMP", "*.jpg", "*.jpeg", "*.png"):
            paths.extend(glob.glob(osp.join(dirpath, ext)))
        return sorted(paths)
