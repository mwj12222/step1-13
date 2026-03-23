#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import sys

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = None
for p in [THIS_FILE.parent, *THIS_FILE.parents]:
    if (p / "configs").is_dir() and ((p / "pipelines").is_dir() or (p / " pipelines").is_dir()):
        PROJECT_ROOT = p
        break
if PROJECT_ROOT is None:
    raise RuntimeError(f"Cannot locate project root from {THIS_FILE}")

sys.path.insert(0, str(PROJECT_ROOT / "src" / "common"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "feature"))

import os
import glob
import cv2
import numpy as np

from config_utils import load_cfg, get_common_paths
from feature_uniform_utils import (
    build_pyramid,
    compute_N_per_level,
    detect_orb_candidates_per_level,
    quadtree_uniform,
    quadtree_uniform_plus,
    grid_quadtree_uniform,
    upscale_keypoints,
)


def build_uniform_func(cfg):
    use_plus = bool(cfg["quadtree"].get("use_plus", True))
    mode = cfg["quadtree"].get("mode", "plus" if use_plus else "plain")
    max_depth = int(cfg["quadtree"].get("max_depth", 4))
    min_pts_split = int(cfg["quadtree"].get("min_pts_split", 8))
    min_side = int(cfg["quadtree"].get("min_side", 20))
    grid_rows = int(cfg["quadtree"].get("grid_rows", 0))
    grid_cols = int(cfg["quadtree"].get("grid_cols", 0))
    score_alpha = float(cfg["quadtree"].get("score_alpha", 1.0))
    score_beta = float(cfg["quadtree"].get("score_beta", 0.0))
    base_quota = int(cfg["quadtree"].get("base_quota", 1))

    if mode == "plain":
        return lambda kps, img_shape, ni: quadtree_uniform(kps, img_shape, ni)
    if mode == "plus":
        return lambda kps, img_shape, ni: quadtree_uniform_plus(
            kps,
            img_shape,
            ni,
            max_depth=max_depth,
            min_pts_split=min_pts_split,
            min_side=min_side,
        )
    if mode == "grid_plus":
        return lambda kps, img_shape, ni: grid_quadtree_uniform(
            kps,
            img_shape,
            ni,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            score_alpha=score_alpha,
            score_beta=score_beta,
            base_quota=base_quota,
            qt_params=dict(max_depth=max_depth, min_pts_split=min_pts_split, min_side=min_side),
        )
    return lambda kps, img_shape, ni: quadtree_uniform_plus(
        kps,
        img_shape,
        ni,
        max_depth=max_depth,
        min_pts_split=min_pts_split,
        min_side=min_side,
    )


def compute_grid_stats_from_kps(kps, h, w, grid_rows, grid_cols):
    counts = np.zeros((grid_rows, grid_cols), dtype=np.int32)

    for kp in kps:
        x, y = kp.pt
        col = int(x * grid_cols / w)
        row = int(y * grid_rows / h)
        if 0 <= col < grid_cols and 0 <= row < grid_rows:
            counts[row, col] += 1

    vals = counts.reshape(-1).astype(np.float32)
    mean = float(vals.mean())
    std = float(vals.std())
    nz = int((vals > 0).sum())
    cv = std / (mean + 1e-6)

    return {
        "mean": mean,
        "std": std,
        "min": int(vals.min()) if vals.size else 0,
        "max": int(vals.max()) if vals.size else 0,
        "cv": cv,
        "nz": nz,
    }


def main():
    cfg_in_pipeline = PROJECT_ROOT / "configs" / "pipeline" / "config_kitti_00.yaml"
    default_cfg = str(cfg_in_pipeline) if cfg_in_pipeline.exists() else str(
        PROJECT_ROOT / "configs" / "reference" / "config_kitti_00.yaml"
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default=default_cfg)
    ap.add_argument("--frame_index", type=int, default=0)
    ap.add_argument("--out_tag", default="", help="Optional subdir tag under eval_root.")
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    paths = get_common_paths(cfg)

    img_dir = paths["img_dir"]
    img_ext = cfg["dataset"]["img_ext"]

    n_total = int(cfg["orb"]["n_features"])
    n_levels = int(cfg["orb"]["n_levels"])
    scale = float(cfg["orb"]["scale_factor"])
    fast_th = int(cfg["orb"]["fast_threshold"])
    over_factor = int(cfg["orb"]["over_factor"])
    candi_min = int(cfg["orb"]["candi_min"])
    candi_k = int(cfg["orb"]["candi_k"])

    grid_rows = int(cfg["grid_eval"]["rows"])
    grid_cols = int(cfg["grid_eval"]["cols"])

    eval_root = paths["eval_root"]
    if args.out_tag:
        eval_root = os.path.join(eval_root, args.out_tag)
    os.makedirs(eval_root, exist_ok=True)
    table_path = os.path.join(eval_root, "step4_uniform_eval_table.txt")

    img_paths = sorted(glob.glob(os.path.join(img_dir, img_ext)))
    if not img_paths:
        print("没有找到图片，请检查 cfg.dataset 配置")
        return

    idx = min(max(0, int(args.frame_index)), len(img_paths) - 1)
    img_path = img_paths[idx]

    img_color = cv2.imread(img_path)
    if img_color is None:
        print("读取图片失败:", img_path)
        return

    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    orb_raw = cv2.ORB_create(
        nfeatures=n_total,
        scaleFactor=scale,
        nlevels=n_levels,
        fastThreshold=fast_th,
    )
    kps_raw = orb_raw.detect(gray, mask=None)

    pyr = build_pyramid(gray, n_levels=n_levels, scale=scale)
    n_per = compute_N_per_level(h, w, n_levels, scale, n_total)
    cand_kps_per_level = detect_orb_candidates_per_level(
        pyr,
        n_per,
        over_factor=over_factor,
        fast_th=fast_th,
        candi_min=candi_min,
        candi_k=candi_k,
    )

    uniform_func = build_uniform_func(cfg)
    kps_uni = []
    for lvl, (img_l, cand_kps, n_i) in enumerate(zip(pyr, cand_kps_per_level, n_per)):
        if n_i <= 0 or len(cand_kps) == 0:
            continue
        kps_lvl = uniform_func(cand_kps, img_l.shape[:2], int(n_i))
        kps_uni.extend(upscale_keypoints(kps_lvl, level=lvl, scale=scale))

    stats_raw = compute_grid_stats_from_kps(kps_raw, h, w, grid_rows, grid_cols)
    stats_uni = compute_grid_stats_from_kps(kps_uni, h, w, grid_rows, grid_cols)

    print(f"[Step4] cfg={args.cfg}")
    print(f"[Step4] frame={os.path.basename(img_path)}")
    print(f"[Step4] RAW: std={stats_raw['std']:.3f}, cv={stats_raw['cv']:.3f}, nz={stats_raw['nz']}/{grid_rows*grid_cols}")
    print(f"[Step4] UNI: std={stats_uni['std']:.3f}, cv={stats_uni['cv']:.3f}, nz={stats_uni['nz']}/{grid_rows*grid_cols}")

    with open(table_path, "w", encoding="utf-8") as f:
        f.write(f"# frame: {os.path.basename(img_path)}\n")
        f.write(f"# grid: {grid_rows}x{grid_cols}\n\n")
        f.write("| type | n_pts | mean | std | min | max | cv | non_empty |\n")
        f.write("|------|-------|------|-----|-----|-----|----|-----------|\n")
        f.write(
            f"| raw_orb | {len(kps_raw)} | {stats_raw['mean']:.3f} | {stats_raw['std']:.3f} | "
            f"{stats_raw['min']} | {stats_raw['max']} | {stats_raw['cv']:.3f} | {stats_raw['nz']} |\n"
        )
        f.write(
            f"| uniform_orb | {len(kps_uni)} | {stats_uni['mean']:.3f} | {stats_uni['std']:.3f} | "
            f"{stats_uni['min']} | {stats_uni['max']} | {stats_uni['cv']:.3f} | {stats_uni['nz']} |\n"
        )

    print(f"[Step4] 已输出: {table_path}")


if __name__ == "__main__":
    main()
