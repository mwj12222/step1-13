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
import csv

import cv2
import numpy as np
from tqdm import tqdm

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


def resolve_default_cfg():
    cfg_in_pipeline = PROJECT_ROOT / "configs" / "pipeline" / "config_kitti_00.yaml"
    if cfg_in_pipeline.exists():
        return str(cfg_in_pipeline)
    return str(PROJECT_ROOT / "configs" / "reference" / "config_kitti_00.yaml")


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


def detect_raw_and_uniform(gray, cfg, uniform_func):
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

    raw_pts = np.array([kp.pt for kp in kps_raw], dtype=np.float32).reshape(-1, 2)
    uni_pts = np.array([kp.pt for kp in kps_uni], dtype=np.float32).reshape(-1, 2)
    return raw_pts, uni_pts


def count_in_mask(pts, mask, mask_val=255):
    if pts.size == 0:
        return 0
    h, w = mask.shape[:2]
    xs = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
    ys = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
    return int(np.sum(mask[ys, xs] == mask_val))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default=resolve_default_cfg())
    ap.add_argument("--start_frame", type=int, default=0)
    ap.add_argument("--max_frames", type=int, default=0, help="0=all")
    ap.add_argument("--frame_stride", type=int, default=1)
    ap.add_argument("--out_tag", default="", help="Optional subdir tag under eval_root.")
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    paths = get_common_paths(cfg)
    seq_name = cfg["dataset"]["name"]
    img_dir = paths["img_dir"]
    mask_dir = paths["mask_dir"]
    img_ext = cfg["dataset"]["img_ext"]

    eval_root = paths["eval_root"]
    if args.out_tag:
        eval_root = os.path.join(eval_root, args.out_tag)
    os.makedirs(eval_root, exist_ok=True)
    out_txt = os.path.join(eval_root, "step5b_dynamic_baseline_eval.txt")
    out_frame_csv = os.path.join(eval_root, "step5b_dynamic_baseline_frame_metrics.csv")
    out_summary_csv = os.path.join(eval_root, "step5b_dynamic_baseline_summary.csv")

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
    frame_rows = []
    raw_dyn_counts = []
    uni_dyn_counts = []
    raw_total_counts = []
    uni_total_counts = []

    print(f"[Step5b] 序列: {seq_name}")
    print(f"  IMG_DIR : {img_dir}")
    print(f"  MASK_DIR: {mask_dir}")
    print(f"  OUT_TXT : {out_txt}")

    for idx, img_path in enumerate(tqdm(img_paths, desc="Step5b dynamic baseline eval")):
        name = os.path.splitext(os.path.basename(img_path))[0]
        img = cv2.imread(img_path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask_path = os.path.join(mask_dir, name + ".png")
        mask = cv2.imread(mask_path, 0)
        if mask is None:
            h, w = gray.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)

        raw_pts, uni_pts = detect_raw_and_uniform(gray, cfg, uniform_func)
        n_raw_total = int(raw_pts.shape[0])
        n_uni_total = int(uni_pts.shape[0])
        n_raw_dyn = count_in_mask(raw_pts, mask)
        n_uni_dyn = count_in_mask(uni_pts, mask)

        raw_total_counts.append(n_raw_total)
        uni_total_counts.append(n_uni_total)
        raw_dyn_counts.append(n_raw_dyn)
        uni_dyn_counts.append(n_uni_dyn)
        frame_rows.append(
            {
                "frame_index": int(start_frame + idx * stride),
                "frame_name": name,
                "mask_dyn_ratio": float(np.mean(mask == 255)),
                "raw_total": n_raw_total,
                "uniform_total": n_uni_total,
                "raw_dyn": n_raw_dyn,
                "uniform_dyn": n_uni_dyn,
                "raw_dyn_ratio": float(n_raw_dyn / max(1, n_raw_total)),
                "uniform_dyn_ratio": float(n_uni_dyn / max(1, n_uni_total)),
            }
        )

    if not frame_rows:
        print("[Step5b] 没有可用帧")
        return

    mean_raw_total = float(np.mean(raw_total_counts))
    mean_uni_total = float(np.mean(uni_total_counts))
    mean_raw_dyn = float(np.mean(raw_dyn_counts))
    mean_uni_dyn = float(np.mean(uni_dyn_counts))
    ratio_uniform_vs_raw = mean_uni_dyn / max(mean_raw_dyn, 1e-6)

    print("===== Step5b Dynamic Baseline Evaluation =====")
    print(f"mean #kps total (raw)    : {mean_raw_total:.2f}")
    print(f"mean #kps total (uniform): {mean_uni_total:.2f}")
    print(f"mean #kps in mask (raw)  : {mean_raw_dyn:.2f}")
    print(f"mean #kps in mask (uni)  : {mean_uni_dyn:.2f}")
    print(f"ratio dyn kept after uniform-only: {ratio_uniform_vs_raw * 100:.2f}%")

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("# Step5b: Dynamic contamination for baseline frontend\n")
        f.write(f"mean_raw_total\t{mean_raw_total:.6f}\n")
        f.write(f"mean_uniform_total\t{mean_uni_total:.6f}\n")
        f.write(f"mean_raw_dyn\t{mean_raw_dyn:.6f}\n")
        f.write(f"mean_uniform_dyn\t{mean_uni_dyn:.6f}\n")
        f.write(f"ratio_uniform_vs_raw_dyn\t{ratio_uniform_vs_raw:.6f}\n")

    fieldnames = list(frame_rows[0].keys())
    with open(out_frame_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(frame_rows)

    with open(out_summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "num_frames",
                "mean_raw_total",
                "mean_uniform_total",
                "mean_raw_dyn",
                "mean_uniform_dyn",
                "ratio_uniform_vs_raw_dyn",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "num_frames": len(frame_rows),
                "mean_raw_total": mean_raw_total,
                "mean_uniform_total": mean_uni_total,
                "mean_raw_dyn": mean_raw_dyn,
                "mean_uniform_dyn": mean_uni_dyn,
                "ratio_uniform_vs_raw_dyn": ratio_uniform_vs_raw,
            }
        )

    print(f"[Step5b] frame metrics -> {out_frame_csv}")
    print(f"[Step5b] summary -> {out_summary_csv}")
    print(f"[Step5b] txt -> {out_txt}")


if __name__ == "__main__":
    main()
