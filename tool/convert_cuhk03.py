#!/usr/bin/env python3
"""
Convert the raw CUHK03 release (``cuhk-03.mat``) into the Market-style
ReID layout expected by ``pad.data.seen.CUHK03``.

Usage::

    python tools/convert_cuhk03.py \\
        --src_dir /path/to/cuhk03_release \\
        --out_dir data/cuhk03 \\
        --mode detected

``--src_dir`` should contain ``cuhk-03.mat`` and, if available,
``cuhk03_new_protocol_config_{detected,labeled}.mat`` (the widely-used
"new protocol" split introduced by Zhong et al., CVPR 2017). If the
new-protocol file is missing, the script falls back to the first of
the 20 legacy train/test splits shipped inside ``cuhk-03.mat``.

Output filenames follow ``{pid:04d}_c{camid}_{idx:05d}.jpg`` with
``camid`` in ``[1, 10]`` (5 camera pairs x 2 views), matching the
pattern parsed by ``pad.data.seen.CUHK03``.
"""

import argparse
import os
import os.path as osp
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import scipy.io as sio
from PIL import Image
from tqdm import tqdm

# Different releases of the "new protocol" config file use slightly
# different key names for the same three splits.
_TRAIN_KEYS = ["train_idx", "train_inds", "train", "train_index"]
_QUERY_KEYS = ["query_idx", "query_inds", "query", "query_index"]
_GALLERY_KEYS = ["gallery_idx", "gallery_inds", "gallery", "gallery_index"]


def _load_mat(path: str) -> Dict[str, Any]:
    """Load a .mat file, falling back to ``mat73`` for v7.3 (HDF5) files."""
    try:
        return sio.loadmat(path)
    except NotImplementedError:
        import mat73
        return mat73.loadmat(path)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _save_image(arr: np.ndarray, path: str) -> None:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        peak = float(arr.max()) if arr.size else 0.0
        if peak <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path, quality=95)


def _flatten_images_to_list(cell) -> List[np.ndarray]:
    """Recursively flatten a cuhk-03.mat image cell into a list of arrays."""
    if isinstance(cell, np.ndarray):
        if cell.dtype != object:
            if cell.ndim == 2:
                return [np.stack([cell] * 3, axis=-1)]
            if cell.ndim == 3:
                return [cell]
            return []
        out = []
        for item in cell.ravel():
            out.extend(_flatten_images_to_list(item))
        return out
    if isinstance(cell, (list, tuple)):
        out = []
        for item in cell:
            out.extend(_flatten_images_to_list(item))
        return out
    return []


def flatten_cuhk03(mat: Dict[str, Any], split_key: str):
    """Flatten the 5x1 pair/identity cell array into a flat image list.

    Returns ``(images, id_map)`` where ``images`` is a list of dicts
    with keys ``pair`` (1-5), ``id_in_pair``, ``cam`` (0/1), ``np_img``,
    and ``id_map`` maps ``(pair, id_in_pair) -> global pid``.
    """
    assert split_key in mat, f"'{split_key}' not found in cuhk-03.mat (keys={list(mat.keys())})"
    pairs = mat[split_key]
    if not isinstance(pairs, np.ndarray):
        pairs = np.array(pairs, dtype=object)

    images, id_map = [], {}
    global_pid = 0
    for pair_idx in range(pairs.shape[0]):
        pair_cell = pairs[pair_idx, 0] if pairs.ndim == 2 else pairs[pair_idx]
        if not isinstance(pair_cell, np.ndarray):
            pair_cell = np.array(pair_cell, dtype=object)
        num_ids = len(pair_cell) if pair_cell.ndim == 1 else pair_cell.shape[0]

        for id_in_pair in range(num_ids):
            id_map[(pair_idx + 1, id_in_pair + 1)] = global_pid
            # 10 cells per identity: cells 0-4 are cam A, cells 5-9 are cam B.
            for cell_col in range(10):
                cam = 0 if cell_col < 5 else 1
                row = pair_cell[id_in_pair] if pair_cell.ndim == 1 else pair_cell[id_in_pair, :]
                for np_img in _flatten_images_to_list(row[cell_col]):
                    images.append(dict(pair=pair_idx + 1, id_in_pair=id_in_pair + 1,
                                       cam=cam, np_img=np_img))
            global_pid += 1
    return images, id_map


def _parse_new_protocol(cfg_mat: Dict[str, Any]) -> Optional[Dict[str, np.ndarray]]:
    def _pick(candidates):
        for k in candidates:
            if k in cfg_mat:
                return np.asarray(cfg_mat[k]).squeeze()
        return None

    train, query, gallery = _pick(_TRAIN_KEYS), _pick(_QUERY_KEYS), _pick(_GALLERY_KEYS)
    if train is None or query is None or gallery is None:
        print(f"[WARN] new-protocol keys not found; available keys: {list(cfg_mat.keys())}")
        return None
    return {"train": train, "query": query, "gallery": gallery}


def _indices_to_image_ids(indices: np.ndarray, images: List[dict], id_map: dict) -> List[int]:
    """Map new-protocol indices (flat image index or (pair, id) pairs) to image list positions."""
    if indices.ndim == 1:
        idx = indices.astype(np.int64).ravel()
        n = len(images)
        if idx.min() >= 1 and idx.max() <= n:
            idx = idx - 1  # 1-based -> 0-based
        elif not (idx.min() >= 0 and idx.max() <= n - 1):
            raise ValueError(f"indices out of range: min={idx.min()} max={idx.max()} n_images={n}")
        return idx.tolist()

    if indices.ndim == 2 and indices.shape[1] == 2:
        target = {(int(p), int(i)) for p, i in indices}
        return [k for k, rec in enumerate(images) if (rec["pair"], rec["id_in_pair"]) in target]

    raise ValueError(f"unsupported indices shape: {indices.shape}")


def _legacy_split(mat: Dict[str, Any], images: List[dict], id_map: dict):
    """Fall back to the first of the 20 legacy train/test splits in cuhk-03.mat."""
    test_pairs = np.asarray(mat["testsets"][0, 0])
    test_ids = {(int(p), int(i)) for p, i in test_pairs}

    train_idx, test_idx = [], []
    for k, rec in enumerate(images):
        (test_idx if (rec["pair"], rec["id_in_pair"]) in test_ids else train_idx).append(k)

    # Per (pid, camid) bucket, the first image becomes the query and the
    # rest become gallery.
    buckets = defaultdict(list)
    for k in test_idx:
        rec = images[k]
        pid = id_map[(rec["pair"], rec["id_in_pair"])]
        camid = 2 * rec["pair"] - (1 if rec["cam"] == 0 else 0)
        buckets[(pid, camid)].append(k)

    query_idx, gallery_idx = [], []
    for ks in buckets.values():
        ks = sorted(ks)
        query_idx.append(ks[0])
        gallery_idx.extend(ks[1:])
    return train_idx, query_idx, gallery_idx


def _dump_split(images: List[dict], indices: List[int], out_dir: str, id_map: dict) -> int:
    _ensure_dir(out_dir)
    for i, k in enumerate(tqdm(indices, ncols=80, leave=False)):
        rec = images[k]
        pid = id_map[(rec["pair"], rec["id_in_pair"])]
        camid = 2 * rec["pair"] - (1 if rec["cam"] == 0 else 0)
        fname = f"{pid:04d}_c{camid}_{i+1:05d}.jpg"
        _save_image(rec["np_img"], osp.join(out_dir, fname))
    return len(indices)


def main(args):
    mat_path = osp.join(args.src_dir, "cuhk-03.mat")
    if not osp.exists(mat_path):
        raise FileNotFoundError(mat_path)

    print(f"[Load] {mat_path}")
    mat = _load_mat(mat_path)
    images, id_map = flatten_cuhk03(mat, split_key=args.mode)
    print(f"[Info] {len(images)} images, {len(id_map)} identities")

    cfg_path = osp.join(args.src_dir, f"cuhk03_new_protocol_config_{args.mode}.mat")
    split = None
    if osp.exists(cfg_path):
        print(f"[Load] new-protocol config: {cfg_path}")
        split = _parse_new_protocol(_load_mat(cfg_path))

    if split is not None:
        train_idx = _indices_to_image_ids(split["train"], images, id_map)
        query_idx = _indices_to_image_ids(split["query"], images, id_map)
        gallery_idx = _indices_to_image_ids(split["gallery"], images, id_map)
        print(f"[Split] new-protocol | train={len(train_idx)} query={len(query_idx)} gallery={len(gallery_idx)}")
    else:
        print("[Info] new-protocol config not found; falling back to legacy split #1")
        train_idx, query_idx, gallery_idx = _legacy_split(mat, images, id_map)
        print(f"[Split] legacy | train={len(train_idx)} query={len(query_idx)} gallery={len(gallery_idx)}")

    print("[Write] train/query/gallery ...")
    n_train = _dump_split(images, train_idx, osp.join(args.out_dir, "bounding_box_train"), id_map)
    n_query = _dump_split(images, query_idx, osp.join(args.out_dir, "query"), id_map)
    n_gallery = _dump_split(images, gallery_idx, osp.join(args.out_dir, "bounding_box_test"), id_map)
    print(f"[Done] {args.out_dir}: train={n_train} query={n_query} gallery={n_gallery}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert cuhk-03.mat to ReID-style folders")
    parser.add_argument("--src_dir", required=True,
                        help="directory containing cuhk-03.mat (and cuhk03_new_protocol_config_*.mat)")
    parser.add_argument("--out_dir", required=True, help="output directory, e.g. data/cuhk03")
    parser.add_argument("--mode", default="detected", choices=["detected", "labeled"])
    main(parser.parse_args())
