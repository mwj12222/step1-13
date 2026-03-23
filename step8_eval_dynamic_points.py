#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
step8_eval_dynamic_points.py

目的：
  在整条序列上统计“动态区域（掩膜=255）内的特征点数量”：
    - 四叉树均匀化之后（all_pts）
    - 掩膜边缘剔除之后（kept_pts）

前置依赖：
  - Step6 生成的 mask_person/*.png
  - Step7 生成的 kps_npz/*.npz（里面有 all_pts / kept_pts）

输出：
  一个 txt 文件，给出：
    - 全图特征点数量的均值（before/after）
    - 动态区域特征点数量的均值（before/after）
    - 动态区域内特征点保留比例 ratio_dyn
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

MASK_DIR  = paths["mask_dir"]
KPS_DIR   = os.path.join(paths["step7_root"], "kps_npz")
EVAL_ROOT = paths["eval_root"]
if args.out_tag:
    KPS_DIR = os.path.join(paths["step7_root"], args.out_tag, "kps_npz")
    EVAL_ROOT = os.path.join(EVAL_ROOT, args.out_tag)
os.makedirs(EVAL_ROOT, exist_ok=True)

OUT_TXT = os.path.join(EVAL_ROOT, "step8_dynamic_eval.txt")
OUT_FRAME_CSV = os.path.join(EVAL_ROOT, "feature_frame_metrics.csv")
OUT_SUMMARY_CSV = os.path.join(EVAL_ROOT, "feature_summary.csv")
OUT_SUMMARY_JSON = os.path.join(EVAL_ROOT, "feature_summary.json")

# 用一张白底图来获取默认的图像尺寸，防止 mask 缺失时需要创建全 0 掩膜
IMG_WHITE_DIR = paths["white_dir"]
_default_img_paths = sorted(glob(os.path.join(IMG_WHITE_DIR, "*.png")))
if _default_img_paths:
    _tmp_img = cv2.imread(_default_img_paths[0])
    if _tmp_img is not None:
        DEFAULT_H, DEFAULT_W = _tmp_img.shape[:2]
    else:
        DEFAULT_H, DEFAULT_W = 480, 640
else:
    DEFAULT_H, DEFAULT_W = 480, 640


def count_in_mask(pts, mask, mask_val=255):
    """
    统计点集中落在掩膜=mask_val 区域内的点数。
    pts: (N,2) 的 xy 坐标（浮点，原图坐标系）
    mask: HxW 的 uint8 图，动态区域为 mask_val
    """
    if pts.size == 0:
        return 0

    h, w = mask.shape[:2]
    xs = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
    ys = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)

    return int(np.sum(mask[ys, xs] == mask_val))


def safe_mean(values):
    if values.size == 0:
        return 0.0
    return float(np.mean(values))


def safe_percentile(values, q):
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, q))


def select_points_in_mask(pts, mask, mask_val=255):
    if pts.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    h, w = mask.shape[:2]
    xs = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
    ys = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
    keep = mask[ys, xs] == mask_val
    return pts[keep].astype(np.float32, copy=False)


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

    print(f"[Step8] 序列: {SEQ_NAME}")
    print("  MASK_DIR :", MASK_DIR)
    print("  KPS_DIR  :", KPS_DIR)
    print("  OUT_TXT  :", OUT_TXT)
    print(f"  DEFAULT image size for empty mask: H={DEFAULT_H}, W={DEFAULT_W}")

    all_total_before = []
    all_total_after  = []
    dyn_before       = []
    dyn_after        = []
    frame_rows = []
    proxy_radius_px = float(cfg.get("grid_eval", {}).get("proxy_match_radius_px", 8.0))
    prev_all_pts = None
    prev_kept_pts = None
    prev_all_dyn_pts = None
    prev_kept_dyn_pts = None

    for frame_index, npz_path in enumerate(tqdm(npz_paths, desc="Step8 dynamic eval")):
        name = os.path.splitext(os.path.basename(npz_path))[0]
        mask_path = os.path.join(MASK_DIR, name + ".png")

        mask = cv2.imread(mask_path, 0)
        if mask is None:
            # 这一帧没有掩膜，就当作全 0（没有动态区域）
            h, w = DEFAULT_H, DEFAULT_W
            mask = np.zeros((h, w), dtype=np.uint8)

        data = np.load(npz_path)
        all_pts  = data["all_pts"]   # Nx2
        kept_pts = data["kept_pts"]  # Mx2
        all_r = data["all_r"] if "all_r" in data else np.ones((all_pts.shape[0],), dtype=np.float32)
        kept_r = data["kept_r"] if "kept_r" in data else np.ones((kept_pts.shape[0],), dtype=np.float32)
        all_in_band = data["all_in_band"] if "all_in_band" in data else np.zeros((all_pts.shape[0],), dtype=np.uint8)
        kept_in_band = data["kept_in_band"] if "kept_in_band" in data else np.zeros((kept_pts.shape[0],), dtype=np.uint8)
        all_dist = data["all_dist"] if "all_dist" in data else np.zeros((all_pts.shape[0],), dtype=np.float32)
        kept_dist = data["kept_dist"] if "kept_dist" in data else np.zeros((kept_pts.shape[0],), dtype=np.float32)
        mask_dyn_ratio = float(data["mask_dyn_ratio"][0]) if "mask_dyn_ratio" in data else float(np.mean(mask == 255))

        n_all_total = all_pts.shape[0]
        n_kept_total = kept_pts.shape[0]

        n_all_dyn  = count_in_mask(all_pts, mask)   # 动态区域内，剔除前
        n_kept_dyn = count_in_mask(kept_pts, mask)  # 动态区域内，剔除后
        all_dyn_pts = select_points_in_mask(all_pts, mask)
        kept_dyn_pts = select_points_in_mask(kept_pts, mask)

        all_total_before.append(n_all_total)
        all_total_after.append(n_kept_total)
        dyn_before.append(n_all_dyn)
        dyn_after.append(n_kept_dyn)
        proxy_all = point_persistence_proxy(prev_all_pts, all_pts, radius_px=proxy_radius_px) if prev_all_pts is not None else {}
        proxy_kept = point_persistence_proxy(prev_kept_pts, kept_pts, radius_px=proxy_radius_px) if prev_kept_pts is not None else {}
        proxy_all_dyn = point_persistence_proxy(prev_all_dyn_pts, all_dyn_pts, radius_px=proxy_radius_px) if prev_all_dyn_pts is not None else {}
        proxy_kept_dyn = point_persistence_proxy(prev_kept_dyn_pts, kept_dyn_pts, radius_px=proxy_radius_px) if prev_kept_dyn_pts is not None else {}
        frame_rows.append({
            "frame_index": int(frame_index),
            "frame_name": name,
            "mask_dyn_ratio": mask_dyn_ratio,
            "n_all": int(n_all_total),
            "n_kept": int(n_kept_total),
            "all_dyn_ratio": float(n_all_dyn / max(1, n_all_total)),
            "kept_dyn_ratio": float(n_kept_dyn / max(1, n_kept_total)),
            "all_p_static": safe_mean(all_r),
            "kept_p_static": safe_mean(kept_r),
            "all_r_mean": safe_mean(all_r),
            "kept_r_mean": safe_mean(kept_r),
            "all_r_p10": safe_percentile(all_r, 10),
            "kept_r_p10": safe_percentile(kept_r, 10),
            "all_band_ratio": safe_mean(all_in_band.astype(np.float32)),
            "kept_band_ratio": safe_mean(kept_in_band.astype(np.float32)),
            "all_dist_mean": safe_mean(all_dist),
            "kept_dist_mean": safe_mean(kept_dist),
            "proxy_radius_px": proxy_radius_px,
            "all_proxy_retention_ratio": float(proxy_all.get("proxy_retention_ratio", 0.0)),
            "all_proxy_med_disp_px": float(proxy_all.get("proxy_med_disp_px", 0.0)),
            "all_proxy_p90_disp_px": float(proxy_all.get("proxy_p90_disp_px", 0.0)),
            "kept_proxy_retention_ratio": float(proxy_kept.get("proxy_retention_ratio", 0.0)),
            "kept_proxy_med_disp_px": float(proxy_kept.get("proxy_med_disp_px", 0.0)),
            "kept_proxy_p90_disp_px": float(proxy_kept.get("proxy_p90_disp_px", 0.0)),
            "all_dyn_proxy_retention_ratio": float(proxy_all_dyn.get("proxy_retention_ratio", 0.0)),
            "all_dyn_proxy_med_disp_px": float(proxy_all_dyn.get("proxy_med_disp_px", 0.0)),
            "kept_dyn_proxy_retention_ratio": float(proxy_kept_dyn.get("proxy_retention_ratio", 0.0)),
            "kept_dyn_proxy_med_disp_px": float(proxy_kept_dyn.get("proxy_med_disp_px", 0.0)),
        })
        prev_all_pts = all_pts.astype(np.float32, copy=False)
        prev_kept_pts = kept_pts.astype(np.float32, copy=False)
        prev_all_dyn_pts = all_dyn_pts
        prev_kept_dyn_pts = kept_dyn_pts

    all_total_before = np.array(all_total_before, dtype=np.float32)
    all_total_after  = np.array(all_total_after,  dtype=np.float32)
    dyn_before       = np.array(dyn_before,       dtype=np.float32)
    dyn_after        = np.array(dyn_after,        dtype=np.float32)

    # ====== 计算平均值 ======
    mean_all_before = float(all_total_before.mean())
    mean_all_after  = float(all_total_after.mean())
    mean_dyn_before = float(dyn_before.mean())
    mean_dyn_after  = float(dyn_after.mean())

    # 为了更直观，看一下动态区域内特征点的“保留比例”
    ratio_dyn = mean_dyn_after / max(mean_dyn_before, 1e-6)

    print("===== Step8 Dynamic Region Evaluation =====")
    print(f"mean #kps total   (before) : {mean_all_before:.2f}")
    print(f"mean #kps total   (after ) : {mean_all_after:.2f}")
    print()
    print(f"mean #kps in mask (before) : {mean_dyn_before:.2f}")
    print(f"mean #kps in mask (after ) : {mean_dyn_after:.2f}")
    print(f"ratio kept in dynamic region: {ratio_dyn * 100:.2f}%")

    # ====== 写入 txt，后面写论文可以直接看 ======
    with open(OUT_TXT, "w") as f:
        f.write("# Step8: Dynamic region feature statistics\n")
        f.write(f"mean_total_before\t{mean_all_before:.3f}\n")
        f.write(f"mean_total_after\t{mean_all_after:.3f}\n")
        f.write(f"mean_dyn_before\t{mean_dyn_before:.3f}\n")
        f.write(f"mean_dyn_after\t{mean_dyn_after:.3f}\n")
        f.write(f"ratio_dyn_kept\t{ratio_dyn:.5f}\n")

    if frame_rows:
        with open(OUT_FRAME_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(frame_rows[0].keys()))
            writer.writeheader()
            writer.writerows(frame_rows)

        summary_row = {
            "script_name": Path(__file__).name,
            "sequence": SEQ_NAME,
            "num_frames": len(frame_rows),
            "mean_total_before": mean_all_before,
            "mean_total_after": mean_all_after,
            "mean_dyn_before": mean_dyn_before,
            "mean_dyn_after": mean_dyn_after,
            "ratio_dyn_kept": ratio_dyn,
            "proxy_match_radius_px": proxy_radius_px,
            "proxy_metric_note": "Proxy frame-to-frame stability from point-set nearest-neighbor matching; not true tracks or epipolar inliers.",
        }
        for key in frame_rows[0].keys():
            if key in ("frame_name",):
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
    print(f"\n[INFO] results saved to {OUT_TXT}")


if __name__ == "__main__":
    main()
