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


def compute_grid_stats(kps, h, w, grid_rows, grid_cols):
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
    cv = std / (mean + 1e-6)
    nz = int((vals > 0).sum())
    return {"std": std, "cv": cv, "nz": nz}


def uniform_kps_for_one_frame(gray, cfg, uniform_func):
    n_total = int(cfg["orb"]["n_features"])
    n_levels = int(cfg["orb"]["n_levels"])
    scale = float(cfg["orb"]["scale_factor"])
    fast_th = int(cfg["orb"]["fast_threshold"])
    over_factor = int(cfg["orb"]["over_factor"])
    candi_min = int(cfg["orb"]["candi_min"])
    candi_k = int(cfg["orb"]["candi_k"])

    h, w = gray.shape[:2]

    orb_raw = cv2.ORB_create(
        nfeatures=n_total,
        nlevels=n_levels,
        scaleFactor=scale,
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

    kps_uni = []
    for lvl, (img_l, cand_kps, n_i) in enumerate(zip(pyr, cand_kps_per_level, n_per)):
        if n_i <= 0 or len(cand_kps) == 0:
            continue
        kps_lvl = uniform_func(cand_kps, img_l.shape[:2], int(n_i))
        kps_uni.extend(upscale_keypoints(kps_lvl, level=lvl, scale=scale))

    return kps_raw, kps_uni


def main():
    cfg_in_pipeline = PROJECT_ROOT / "configs" / "pipeline" / "config_kitti_00.yaml"
    default_cfg = str(cfg_in_pipeline) if cfg_in_pipeline.exists() else str(
        PROJECT_ROOT / "configs" / "reference" / "config_kitti_00.yaml"
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default=default_cfg)
    ap.add_argument("--start_frame", type=int, default=0)
    ap.add_argument("--max_frames", type=int, default=0, help="0=all")
    ap.add_argument("--frame_stride", type=int, default=1)
    ap.add_argument("--out_tag", default="", help="Optional subdir tag under eval_root.")
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    paths = get_common_paths(cfg)

    img_dir = paths["img_dir"]
    img_ext = cfg["dataset"]["img_ext"]
    grid_rows = int(cfg["grid_eval"]["rows"])
    grid_cols = int(cfg["grid_eval"]["cols"])

    eval_root = paths["eval_root"]
    if args.out_tag:
        eval_root = os.path.join(eval_root, args.out_tag)
    os.makedirs(eval_root, exist_ok=True)
    out_txt = os.path.join(eval_root, "step5_uniform_eval_sequence.txt")

    img_paths = sorted(glob.glob(os.path.join(img_dir, img_ext)))
    if not img_paths:
        print("没有找到图片，请检查 cfg.dataset 配置")
        return

    start_frame = max(0, int(args.start_frame))
    stride = max(1, int(args.frame_stride))
    img_paths = img_paths[start_frame::stride]
    if int(args.max_frames) > 0:
        img_paths = img_paths[: int(args.max_frames)]
    if not img_paths:
        print("筛选帧后没有可用图片")
        return

    uniform_func = build_uniform_func(cfg)

    raw_stats = []
    uni_stats = []

    for idx, img_path in enumerate(img_paths):
        if idx % 50 == 0:
            print(f"[Step5] {idx+1}/{len(img_paths)} {os.path.basename(img_path)}")

        img = cv2.imread(img_path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        kps_raw, kps_uni = uniform_kps_for_one_frame(gray, cfg, uniform_func)
        raw_stats.append(compute_grid_stats(kps_raw, h, w, grid_rows, grid_cols))
        uni_stats.append(compute_grid_stats(kps_uni, h, w, grid_rows, grid_cols))

    if not raw_stats:
        print("[Step5] 没有可用帧")
        return

    mean_raw_std = float(np.mean([s["std"] for s in raw_stats]))
    mean_raw_cv = float(np.mean([s["cv"] for s in raw_stats]))
    mean_raw_nz = float(np.mean([s["nz"] for s in raw_stats]))

    mean_uni_std = float(np.mean([s["std"] for s in uni_stats]))
    mean_uni_cv = float(np.mean([s["cv"] for s in uni_stats]))
    mean_uni_nz = float(np.mean([s["nz"] for s in uni_stats]))

    total_cells = grid_rows * grid_cols

    print("[Step5] 序列均值结果")
    print(f"  RAW: std={mean_raw_std:.3f}, cv={mean_raw_cv:.3f}, nz={mean_raw_nz:.2f}/{total_cells}")
    print(f"  UNI: std={mean_uni_std:.3f}, cv={mean_uni_cv:.3f}, nz={mean_uni_nz:.2f}/{total_cells}")

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("# Step5 uniform evaluation (sequence)\n")
        f.write(f"frames\t{len(raw_stats)}\n")
        f.write(f"grid_rows\t{grid_rows}\n")
        f.write(f"grid_cols\t{grid_cols}\n")
        f.write(f"raw_mean_std\t{mean_raw_std:.6f}\n")
        f.write(f"raw_mean_cv\t{mean_raw_cv:.6f}\n")
        f.write(f"raw_mean_non_empty\t{mean_raw_nz:.6f}\n")
        f.write(f"uniform_mean_std\t{mean_uni_std:.6f}\n")
        f.write(f"uniform_mean_cv\t{mean_uni_cv:.6f}\n")
        f.write(f"uniform_mean_non_empty\t{mean_uni_nz:.6f}\n")

    print(f"[Step5] 已输出: {out_txt}")


if __name__ == "__main__":
    main()
