#!/usr/bin/env python3
"""
Download UCR time series datasets required by the Improved notebooks.

Datasets saved to:
  {DatasetName}/{DatasetName}_TRAIN.txt
  {DatasetName}/{DatasetName}_TEST.txt
(comma-delimited, first column = class label, remaining columns = features)

Usage:
  python download_datasets.py
"""

import os
import io
import sys
import zipfile
import urllib.request
import numpy as np

DATASETS = ["Adiac", "GunPoint", "Coffee", "InlineSkate", "MedicalImages"]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UCR_BASE_URL = "https://www.timeseriesclassification.com/Downloads"


def save_splits(name, x_tr, y_tr, x_te, y_te, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for split, x, y in [("TRAIN", x_tr, y_tr), ("TEST", x_te, y_te)]:
        # Ensure y is a 1-D numeric column
        y = np.asarray(y).reshape(-1, 1)
        # If labels are strings, encode to integers
        if y.dtype.kind in ("U", "S", "O"):
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(y.ravel()).reshape(-1, 1).astype(float)
        x = np.asarray(x, dtype=float)
        data = np.hstack([y, x])
        path = os.path.join(out_dir, f"{name}_{split}.txt")
        np.savetxt(path, data, delimiter=",", fmt="%.6g")
        print(f"    Saved {os.path.relpath(path)} ({len(y)} samples, {x.shape[1]} timesteps)")


def download_with_aeon(name, out_dir):
    """Load via aeon (handles caching/download) and persist as local CSV."""
    from aeon.datasets import load_classification

    print(f"  [aeon] Fetching '{name}' ...")
    x_tr, y_tr = load_classification(name, split="train")
    x_te, y_te = load_classification(name, split="test")

    # aeon returns shape (n_samples, n_channels, n_timesteps) for univariate -> drop channel dim
    if x_tr.ndim == 3:
        x_tr = x_tr[:, 0, :]
        x_te = x_te[:, 0, :]

    save_splits(name, x_tr, y_tr, x_te, y_te, out_dir)


def _parse_ts_file(lines):
    """Parse aeon .ts format into (X, y) numpy arrays."""
    in_data = False
    x_list, y_list = [], []
    for line in lines:
        line = line.strip()
        if line.lower() == "@data":
            in_data = True
            continue
        if not in_data or not line or line.startswith("@"):
            continue
        # Format: comma-separated series values, last token is the label
        parts = line.split(",")
        # Last element is the class label
        y_list.append(parts[-1].strip())
        x_list.append([float(v) for v in parts[:-1]])
    return np.array(x_list, dtype=float), np.array(y_list)


def _parse_txt_file(raw_bytes):
    """Parse old-style UCR whitespace-separated .txt files into (X, y)."""
    data = np.loadtxt(io.BytesIO(raw_bytes))
    return data[:, 1:], data[:, 0]


def download_from_ucr(name, out_dir):
    """Download dataset zip directly from timeseriesclassification.com."""
    url = f"{UCR_BASE_URL}/{name}.zip"
    print(f"  [ucr]  Downloading '{name}' from {url} ...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Python/urllib"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            zdata = resp.read()
    except Exception as exc:
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc

    results = {}  # split -> (x, y)
    with zipfile.ZipFile(io.BytesIO(zdata)) as zf:
        members = zf.namelist()
        for split in ("TRAIN", "TEST"):
            # Try .ts first (aeon/newer format), then plain .txt / no extension
            candidates = [
                f"{name}/{name}_{split}.ts",
                f"{name}_{split}.ts",
                f"{name}/{name}_{split}.txt",
                f"{name}_{split}.txt",
                f"{name}/{name}_{split}",
                f"{name}_{split}",
            ]
            found = None
            for c in candidates:
                if c in members:
                    found = c
                    break
            if found is None:
                # Try case-insensitive match
                lower_map = {m.lower(): m for m in members}
                for c in candidates:
                    actual = lower_map.get(c.lower())
                    if actual:
                        found = actual
                        break
            if found is None:
                available = [m for m in members if split.lower() in m.lower()]
                raise FileNotFoundError(
                    f"Cannot find {split} file for '{name}' in zip.\n"
                    f"  Available: {available}"
                )
            raw = zf.read(found)
            if found.endswith(".ts"):
                x, y = _parse_ts_file(raw.decode("utf-8").splitlines())
            else:
                x, y = _parse_txt_file(raw)
            results[split] = (x, y)

    save_splits(
        name,
        results["TRAIN"][0], results["TRAIN"][1],
        results["TEST"][0],  results["TEST"][1],
        out_dir,
    )


def already_downloaded(name, out_dir):
    for split in ("TRAIN", "TEST"):
        found = any(
            os.path.exists(os.path.join(out_dir, f"{name}_{split}{ext}"))
            for ext in ("", ".txt", ".tsv")
        )
        if not found:
            return False
    return True


def main():
    print(f"Downloading {len(DATASETS)} datasets to: {BASE_DIR}\n")
    failed = []
    for name in DATASETS:
        out_dir = os.path.join(BASE_DIR, name)
        print(f"[{name}]")
        if already_downloaded(name, out_dir):
            print(f"  Already present, skipping.\n")
            continue
        # Try aeon first, fall back to direct UCR download
        try:
            download_with_aeon(name, out_dir)
        except Exception as aeon_err:
            print(f"  aeon failed ({aeon_err}), trying direct download ...")
            try:
                download_from_ucr(name, out_dir)
            except Exception as ucr_err:
                print(f"  ERROR: Both methods failed.\n    aeon: {aeon_err}\n    ucr:  {ucr_err}")
                failed.append(name)
        print()

    if failed:
        print(f"Failed datasets: {failed}")
        sys.exit(1)
    else:
        print("All datasets ready.")


if __name__ == "__main__":
    main()
