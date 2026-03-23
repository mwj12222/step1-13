#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
step9_uniform_eval_sequence_static_only.py

目的：
  使用 Step7 生成的 kps_npz 中的 all_pts / kept_pts，
  在整条序列上做 8x8（或其他配置）网格统计，比较：
    - 掩膜边缘剔除前（all_pts）
    - 掩膜边缘剔除后（kept_pts）
  的均匀性指标（std, CV, 非空网格数）。

依赖：
  - step7_masked_uniform_seq/kps_npz/*.npz  (all_pts / kept_pts)
  - rgb_white 目录中的任意一张图来获取 H, W
"""
from pathlib import Path
import argparse
import sys

THIS_FILE = Path(__file__).resolve()

# 统一用 configs/ + pipelines/ 来判定项目根目录
PROJECT_ROOT = None
for p in [THIS_FILE.parent, *THIS_FILE.parent.parents]:
    if (p / "configs").is_dir() and ((p / "pipelines").is_dir() or (p / " pipelines").is_dir()):
        PROJECT_ROOT = p
        break
if PROJECT_ROOT is None:
    raise RuntimeError(f"Cannot locate project root from {THIS_FILE}")

# 统一把 src 加入 PYTHONPATH
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "common"))


import os
import csv
import json
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm

from config_utils import load_cfg, get_common_paths


def resolve_default_cfg():
    cfg_in_pipeline = PROJECT_ROOT / "configs" / "pipeline" / "config_kitti_00.yaml"
    if cfg_in_pipeline.exists():
        return str(cfg_in_pipeline)
    return str(PROJECT_ROOT / "configs" / "reference" / "config_kitti_00.yaml")


# 读取 YAML 配置
_default_cfg = resolve_default_cfg()
ap = argparse.ArgumentParser()
ap.add_argument("--cfg", default=_default_cfg)
ap.add_argument("--out_tag", default="", help="Optional tag to read/write variant-specific outputs.")
args = ap.parse_args()

cfg = load_cfg(args.cfg)
paths = get_common_paths(cfg)

SEQ_NAME = cfg["dataset"]["name"]

KPS_DIR       = os.path.join(paths["step7_root"], "kps_npz")
IMG_WHITE_DIR = paths["white_dir"]  # 用来拿 H,W
EVAL_ROOT     = paths["eval_root"]
if args.out_tag:
    KPS_DIR = os.path.join(paths["step7_root"], args.out_tag, "kps_npz")
    EVAL_ROOT = os.path.join(EVAL_ROOT, args.out_tag)
os.makedirs(EVAL_ROOT, exist_ok=True)

OUT_TXT = os.path.join(EVAL_ROOT, "step9_uniform_eval_sequence.txt")
OUT_FRAME_CSV = os.path.join(EVAL_ROOT, "uniform_frame_metrics.csv")
OUT_SUMMARY_CSV = os.path.join(EVAL_ROOT, "uniform_summary.csv")
OUT_SUMMARY_JSON = os.path.join(EVAL_ROOT, "uniform_summary.json")

GRID_ROWS = cfg["grid_eval"]["rows"]
GRID_COLS = cfg["grid_eval"]["cols"]
EPS = 1e-6


def get_image_size():
    img_paths = sorted(glob(os.path.join(IMG_WHITE_DIR, "*.png")))
    assert len(img_paths) > 0, f"no images found in {IMG_WHITE_DIR}"
    img = cv2.imread(img_paths[0])
    assert img is not None, f"cannot read image: {img_paths[0]}"
    h, w = img.shape[:2]
    return h, w


def grid_stats_for_points(pts, H, W, n_rows=8, n_cols=8):
    """
    对一帧的点集做 n_rows x n_cols 网格划分，统计每格的点数，然后返回：
      - std:          每格点数的标准差
      - cv:           变异系数 = std / mean
      - n_non_empty:  非空格数量
    """
    if pts.size == 0:
        return 0.0, 0.0, 0.0

    # 网格大小
    cell_h = H / float(n_rows)
    cell_w = W / float(n_cols)

    counts = np.zeros((n_rows, n_cols), dtype=np.int32)

    xs = pts[:, 0]
    ys = pts[:, 1]

    # 映射到网格索引
    col_idx = np.clip((xs / cell_w).astype(int), 0, n_cols - 1)
    row_idx = np.clip((ys / cell_h).astype(int), 0, n_rows - 1)

    for r, c in zip(row_idx, col_idx):
        counts[r, c] += 1

    counts_flat = counts.reshape(-1).astype(np.float32)
    mean_val = float(np.mean(counts_flat))
    std_val  = float(np.std(counts_flat))
    cv_val   = std_val / (mean_val + EPS)

    n_non_empty = int(np.count_nonzero(counts_flat > 0))

    return std_val, cv_val, float(n_non_empty)


def grid_detail_metrics(pts, H, W, n_rows=8, n_cols=8):
    if pts.size == 0:
        return {
            "coverage_ratio": 0.0,
            "grid_entropy": 0.0,
            "min_cell_count": 0,
            "min_nonzero_cell_count": 0,
        }

    cell_h = H / float(n_rows)
    cell_w = W / float(n_cols)
    counts = np.zeros((n_rows, n_cols), dtype=np.int32)

    xs = pts[:, 0]
    ys = pts[:, 1]
    col_idx = np.clip((xs / cell_w).astype(int), 0, n_cols - 1)
    row_idx = np.clip((ys / cell_h).astype(int), 0, n_rows - 1)
    for r, c in zip(row_idx, col_idx):
        counts[r, c] += 1

    flat = counts.reshape(-1).astype(np.float32)
    total_cells = max(1, n_rows * n_cols)
    coverage_ratio = float(np.count_nonzero(flat > 0) / float(total_cells))
    prob_sum = float(np.sum(flat))
    if prob_sum <= EPS or total_cells <= 1:
        entropy = 0.0
    else:
        probs = flat / prob_sum
        nz = probs > 1e-12
        entropy = float(-np.sum(probs[nz] * np.log(probs[nz])) / np.log(float(total_cells)))
    nonzero = flat[flat > 0]
    min_nonzero = int(np.min(nonzero)) if nonzero.size > 0 else 0
    return {
        "coverage_ratio": coverage_ratio,
        "grid_entropy": entropy,
        "min_cell_count": int(np.min(flat)) if flat.size > 0 else 0,
        "min_nonzero_cell_count": min_nonzero,
    }


def occupied_grid_mask(pts, H, W, n_rows=8, n_cols=8):
    occ = np.zeros((n_rows, n_cols), dtype=bool)
    if pts.size == 0:
        return occ
    cell_h = H / float(n_rows)
    cell_w = W / float(n_cols)
    xs = pts[:, 0]
    ys = pts[:, 1]
    col_idx = np.clip((xs / cell_w).astype(int), 0, n_cols - 1)
    row_idx = np.clip((ys / cell_h).astype(int), 0, n_rows - 1)
    occ[row_idx, col_idx] = True
    return occ


def grid_overlap_ratio(prev_pts, curr_pts, H, W, n_rows=8, n_cols=8):
    prev_occ = occupied_grid_mask(np.asarray(prev_pts, dtype=np.float32).reshape(-1, 2), H, W, n_rows, n_cols)
    curr_occ = occupied_grid_mask(np.asarray(curr_pts, dtype=np.float32).reshape(-1, 2), H, W, n_rows, n_cols)
    union = np.logical_or(prev_occ, curr_occ)
    if not np.any(union):
        return 0.0
    inter = np.logical_and(prev_occ, curr_occ)
    return float(np.sum(inter) / float(np.sum(union)))


def point_persistence_proxy(prev_pts, curr_pts, radius_px=8.0, chunk_size=256):
    prev = np.asarray(prev_pts, dtype=np.float32).reshape(-1, 2)
    curr = np.asarray(curr_pts, dtype=np.float32).reshape(-1, 2)
    if prev.size == 0 or curr.size == 0:
        return {
            "proxy_retention_ratio": 0.0,
            "proxy_med_disp_px": 0.0,
            "proxy_p90_disp_px": 0.0,
            "proxy_match_count": 0,
        }
    min_dists = []
    for i in range(0, prev.shape[0], chunk_size):
        chunk = prev[i:i + chunk_size]
        diff = chunk[:, None, :] - curr[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        min_dists.append(np.sqrt(np.min(d2, axis=1)))
    min_dists = np.concatenate(min_dists, axis=0)
    matched = min_dists <= float(radius_px)
    matched_dists = min_dists[matched]
    return {
        "proxy_retention_ratio": float(np.mean(matched.astype(np.float32))) if matched.size else 0.0,
        "proxy_med_disp_px": float(np.median(matched_dists)) if matched_dists.size else 0.0,
        "proxy_p90_disp_px": float(np.percentile(matched_dists, 90)) if matched_dists.size else 0.0,
        "proxy_match_count": int(np.sum(matched)),
    }


def main():
    npz_paths = sorted(glob(os.path.join(KPS_DIR, "*.npz")))
    if not npz_paths:
        print("[ERROR] no .npz files found in", KPS_DIR)
        return

    H, W = get_image_size()
    print(f"[Step9] 序列: {SEQ_NAME}")
    print(f"  image size: H={H}, W={W}")
    print("  GRID_ROWS :", GRID_ROWS)
    print("  GRID_COLS :", GRID_COLS)
    print("  KPS_DIR   :", KPS_DIR)
    print("  OUT_TXT   :", OUT_TXT)

    std_all_list, cv_all_list, nonempty_all_list = [], [], []
    std_kept_list, cv_kept_list, nonempty_kept_list = [], [], []
    frame_rows = []
    proxy_radius_px = float(cfg.get("grid_eval", {}).get("proxy_match_radius_px", 8.0))
    prev_all_pts = None
    prev_kept_pts = None

    for frame_index, npz_path in enumerate(tqdm(npz_paths, desc="Step9 uniform eval (sequence)")):
        frame_name = os.path.splitext(os.path.basename(npz_path))[0]
        data = np.load(npz_path)
        all_pts  = data["all_pts"]   # Nx2
        kept_pts = data["kept_pts"]  # Mx2

        std_a, cv_a, ne_a = grid_stats_for_points(all_pts,  H, W, GRID_ROWS, GRID_COLS)
        std_k, cv_k, ne_k = grid_stats_for_points(kept_pts, H, W, GRID_ROWS, GRID_COLS)
        detail_a = grid_detail_metrics(all_pts, H, W, GRID_ROWS, GRID_COLS)
        detail_k = grid_detail_metrics(kept_pts, H, W, GRID_ROWS, GRID_COLS)
        proxy_all = point_persistence_proxy(prev_all_pts, all_pts, radius_px=proxy_radius_px) if prev_all_pts is not None else {}
        proxy_kept = point_persistence_proxy(prev_kept_pts, kept_pts, radius_px=proxy_radius_px) if prev_kept_pts is not None else {}
        overlap_all = grid_overlap_ratio(prev_all_pts, all_pts, H, W, GRID_ROWS, GRID_COLS) if prev_all_pts is not None else 0.0
        overlap_kept = grid_overlap_ratio(prev_kept_pts, kept_pts, H, W, GRID_ROWS, GRID_COLS) if prev_kept_pts is not None else 0.0

        std_all_list.append(std_a)
        cv_all_list.append(cv_a)
        nonempty_all_list.append(ne_a)

        std_kept_list.append(std_k)
        cv_kept_list.append(cv_k)
        nonempty_kept_list.append(ne_k)
        frame_rows.append({
            "frame_index": int(frame_index),
            "frame_name": frame_name,
            "all_std": float(std_a),
            "all_cv": float(cv_a),
            "all_non_empty": float(ne_a),
            "all_coverage_ratio": float(detail_a["coverage_ratio"]),
            "all_grid_entropy": float(detail_a["grid_entropy"]),
            "all_min_cell_count": int(detail_a["min_cell_count"]),
            "all_min_nonzero_cell_count": int(detail_a["min_nonzero_cell_count"]),
            "kept_std": float(std_k),
            "kept_cv": float(cv_k),
            "kept_non_empty": float(ne_k),
            "kept_coverage_ratio": float(detail_k["coverage_ratio"]),
            "kept_grid_entropy": float(detail_k["grid_entropy"]),
            "kept_min_cell_count": int(detail_k["min_cell_count"]),
            "kept_min_nonzero_cell_count": int(detail_k["min_nonzero_cell_count"]),
            "proxy_radius_px": proxy_radius_px,
            "all_proxy_retention_ratio": float(proxy_all.get("proxy_retention_ratio", 0.0)),
            "all_proxy_med_disp_px": float(proxy_all.get("proxy_med_disp_px", 0.0)),
            "all_proxy_p90_disp_px": float(proxy_all.get("proxy_p90_disp_px", 0.0)),
            "all_grid_overlap_ratio": float(overlap_all),
            "kept_proxy_retention_ratio": float(proxy_kept.get("proxy_retention_ratio", 0.0)),
            "kept_proxy_med_disp_px": float(proxy_kept.get("proxy_med_disp_px", 0.0)),
            "kept_proxy_p90_disp_px": float(proxy_kept.get("proxy_p90_disp_px", 0.0)),
            "kept_grid_overlap_ratio": float(overlap_kept),
        })
        prev_all_pts = all_pts.astype(np.float32, copy=False)
        prev_kept_pts = kept_pts.astype(np.float32, copy=False)

    std_all_list      = np.array(std_all_list,      dtype=np.float32)
    cv_all_list       = np.array(cv_all_list,       dtype=np.float32)
    nonempty_all_list = np.array(nonempty_all_list, dtype=np.float32)

    std_kept_list      = np.array(std_kept_list,      dtype=np.float32)
    cv_kept_list       = np.array(cv_kept_list,       dtype=np.float32)
    nonempty_kept_list = np.array(nonempty_kept_list, dtype=np.float32)

    mean_std_all      = float(std_all_list.mean())
    mean_cv_all       = float(cv_all_list.mean())
    mean_nonempty_all = float(nonempty_all_list.mean())

    mean_std_kept      = float(std_kept_list.mean())
    mean_cv_kept       = float(cv_kept_list.mean())
    mean_nonempty_kept = float(nonempty_kept_list.mean())

    print("===== Step9: Uniformity stats (sequence, grid) =====")
    print("---- Before border removal (all_pts) ----")
    print(f"mean std per frame      : {mean_std_all:.3f}")
    print(f"mean CV  per frame      : {mean_cv_all:.3f}")
    print(f"mean non-empty cells    : {mean_nonempty_all:.3f} / {GRID_ROWS * GRID_COLS}")
    print("---- After  border removal (kept_pts) ----")
    print(f"mean std per frame      : {mean_std_kept:.3f}")
    print(f"mean CV  per frame      : {mean_cv_kept:.3f}")
    print(f"mean non-empty cells    : {mean_nonempty_kept:.3f} / {GRID_ROWS * GRID_COLS}")

    # 写入 txt
    with open(OUT_TXT, "w") as f:
        f.write("# Step9: Uniformity stats (grid, sequence)\n")
        f.write("GRID_ROWS\t{}\n".format(GRID_ROWS))
        f.write("GRID_COLS\t{}\n".format(GRID_COLS))
        f.write("---- before border removal (all_pts) ----\n")
        f.write("mean_std\t{:.6f}\n".format(mean_std_all))
        f.write("mean_cv\t{:.6f}\n".format(mean_cv_all))
        f.write("mean_non_empty\t{:.6f}\n".format(mean_nonempty_all))
        f.write("---- after border removal (kept_pts) ----\n")
        f.write("mean_std\t{:.6f}\n".format(mean_std_kept))
        f.write("mean_cv\t{:.6f}\n".format(mean_cv_kept))
        f.write("mean_non_empty\t{:.6f}\n".format(mean_nonempty_kept))

    if frame_rows:
        with open(OUT_FRAME_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(frame_rows[0].keys()))
            writer.writeheader()
            writer.writerows(frame_rows)

        summary_row = {
            "script_name": Path(__file__).name,
            "sequence": SEQ_NAME,
            "num_frames": len(frame_rows),
            "mean_all_std": mean_std_all,
            "mean_all_cv": mean_cv_all,
            "mean_all_non_empty": mean_nonempty_all,
            "mean_kept_std": mean_std_kept,
            "mean_kept_cv": mean_cv_kept,
            "mean_kept_non_empty": mean_nonempty_kept,
            "proxy_match_radius_px": proxy_radius_px,
            "proxy_metric_note": "Proxy frame-to-frame stability from point-set nearest-neighbor matching and grid overlap; not true tracks or epipolar inliers.",
        }
        for key in frame_rows[0].keys():
            if key == "frame_name":
                continue
            vals = []
            for row in frame_rows:
                val = row.get(key)
                if isinstance(val, (int, float, np.integer, np.floating)):
                    vals.append(float(val))
            if vals:
                summary_row[f"mean_{key}"] = float(np.mean(vals))

        with open(OUT_SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
            writer.writeheader()
            writer.writerow(summary_row)
        with open(OUT_SUMMARY_JSON, "w", encoding="utf-8") as f:
            json.dump(summary_row, f, ensure_ascii=False, indent=2)

    print(f"[INFO] frame metrics saved to {OUT_FRAME_CSV}")
    print(f"[INFO] summary metrics saved to {OUT_SUMMARY_CSV}")
    print(f"[INFO] summary json saved to {OUT_SUMMARY_JSON}")
    print(f"[INFO] results saved to {OUT_TXT}")


if __name__ == "__main__":
    main()
