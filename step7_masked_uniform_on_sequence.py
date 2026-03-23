#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
step7_masked_uniform_on_sequence.py

在整个序列上执行：
  1) 使用白色掩膜图像提取 ORB 候选点；
  2) 按论文 3.2.2 / 3.2.4 做四叉树均匀化；
  3) 根据掩膜删除掩膜边缘附近的特征点（半径 r）；
  4) 保存两种可视化结果 (a)/(b)；
  5) 把每帧 all_kps / kept_kps 的坐标存成 npz，供后续数值实验使用。
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
sys.path.insert(0, str(PROJECT_ROOT / "src" / "feature"))


import os
import csv
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm

from feature_uniform_utils import (
    build_pyramid,
    compute_N_per_level,
    detect_orb_candidates_per_level,
    quadtree_uniform,
    quadtree_uniform_plus,
    grid_quadtree_uniform,
    upscale_keypoints,
    remove_kps_near_mask_border,
)

from config_utils import load_cfg, get_common_paths


def resolve_default_cfg():
    cfg_in_pipeline = PROJECT_ROOT / "configs" / "pipeline" / "config_kitti_00.yaml"
    if cfg_in_pipeline.exists():
        return str(cfg_in_pipeline)
    return str(PROJECT_ROOT / "configs" / "reference" / "config_kitti_00.yaml")


def str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def build_reliability_map(mask, use_dt_weight, band_radius, weight_mode, soft_scale):
    static = np.ones_like(mask, dtype=np.uint8)
    static[mask == 255] = 0
    dist = cv2.distanceTransform(static, cv2.DIST_L2, 3).astype(np.float32)

    if not use_dt_weight:
        rel = (mask != 255).astype(np.float32)
        return dist, rel

    rel = np.zeros_like(dist, dtype=np.float32)
    valid = mask != 255
    d_eff = np.maximum(dist - float(band_radius), 0.0)
    scale = max(float(soft_scale), 1e-6)
    mode = str(weight_mode).lower()

    if mode == "logistic":
        rel_valid = 2.0 / (1.0 + np.exp(-d_eff / scale)) - 1.0
    elif mode == "exp":
        rel_valid = 1.0 - np.exp(-d_eff / scale)
    else:
        rel_valid = np.clip(d_eff / scale, 0.0, 1.0)

    rel[valid] = np.clip(rel_valid[valid], 0.0, 1.0)
    return dist, rel


def sample_point_fields(points, mask, dist_map, rel_map, band_radius):
    if points.size == 0:
        empty_f32 = np.zeros((0,), dtype=np.float32)
        empty_u8 = np.zeros((0,), dtype=np.uint8)
        return {
            "dist": empty_f32,
            "rel": empty_f32,
            "dyn": empty_u8,
            "band": empty_u8,
        }

    h, w = mask.shape[:2]
    xs = np.clip(np.round(points[:, 0]).astype(np.int32), 0, w - 1)
    ys = np.clip(np.round(points[:, 1]).astype(np.int32), 0, h - 1)

    dist = dist_map[ys, xs].astype(np.float32)
    rel = rel_map[ys, xs].astype(np.float32)
    dyn = (mask[ys, xs] == 255).astype(np.uint8)
    band = ((dyn == 0) & (dist <= float(band_radius) + 1e-6)).astype(np.uint8)
    return {
        "dist": dist,
        "rel": rel,
        "dyn": dyn,
        "band": band,
    }


def compute_grid_metrics(points, h, w, grid_rows, grid_cols):
    total_cells = max(1, int(grid_rows) * int(grid_cols))
    if points.size == 0 or grid_rows <= 0 or grid_cols <= 0:
        return {
            "coverage_ratio": 0.0,
            "grid_entropy": 0.0,
            "min_cell_count": 0,
        }

    counts = np.zeros((grid_rows, grid_cols), dtype=np.int32)
    xs = np.clip(np.round(points[:, 0]).astype(np.int32), 0, w - 1)
    ys = np.clip(np.round(points[:, 1]).astype(np.int32), 0, h - 1)
    cols = np.clip((xs * grid_cols / max(w, 1)).astype(np.int32), 0, grid_cols - 1)
    rows = np.clip((ys * grid_rows / max(h, 1)).astype(np.int32), 0, grid_rows - 1)

    for r, c in zip(rows, cols):
        counts[r, c] += 1

    coverage_ratio = float(np.count_nonzero(counts) / float(total_cells))
    probs = counts.reshape(-1).astype(np.float32)
    prob_sum = float(np.sum(probs))
    if prob_sum <= 1e-6 or total_cells <= 1:
        entropy = 0.0
    else:
        probs = probs / prob_sum
        nz = probs > 1e-12
        entropy = float(-np.sum(probs[nz] * np.log(probs[nz])) / np.log(float(total_cells)))

    return {
        "coverage_ratio": coverage_ratio,
        "grid_entropy": entropy,
        "min_cell_count": int(np.min(counts)),
    }


def safe_mean(values):
    if values.size == 0:
        return 0.0
    return float(np.mean(values))


def safe_percentile(values, q):
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, q))


def point_metrics_row(frame_index, frame_name, points, fields, mask_dyn_ratio, grid_metrics, prefix):
    n_pts = int(points.shape[0])
    dyn_ratio = safe_mean(fields["dyn"].astype(np.float32))
    band_ratio = safe_mean(fields["band"].astype(np.float32))
    row = {
        "frame_index": int(frame_index),
        "frame_name": frame_name,
        "mask_dyn_ratio": float(mask_dyn_ratio),
        f"{prefix}_n": n_pts,
        f"{prefix}_dyn_ratio": dyn_ratio,
        f"{prefix}_r_mean": safe_mean(fields["rel"]),
        f"{prefix}_r_p10": safe_percentile(fields["rel"], 10),
        f"{prefix}_dist_mean": safe_mean(fields["dist"]),
        f"{prefix}_dist_p10": safe_percentile(fields["dist"], 10),
        f"{prefix}_band_ratio": band_ratio,
        f"{prefix}_coverage_ratio": float(grid_metrics["coverage_ratio"]),
        f"{prefix}_grid_entropy": float(grid_metrics["grid_entropy"]),
        f"{prefix}_min_cell_count": int(grid_metrics["min_cell_count"]),
    }
    return row


# 读取 YAML 配置
_default_cfg = resolve_default_cfg()
ap = argparse.ArgumentParser()
ap.add_argument("--cfg", default=_default_cfg)
ap.add_argument("--out_tag", default="", help="Optional subdir tag under step7_root.")
ap.add_argument("--start_frame", type=int, default=0, help="Optional global start frame index for partial sequence runs.")
ap.add_argument("--max_frames", type=int, default=-1, help="Optional max number of frames to process for flow checks / debugging.")
ap.add_argument("--qt_mode", default="", choices=["", "plain", "plus", "grid_plus"])
ap.add_argument("--border_radius", type=int, default=-1)
ap.add_argument("--use_dt_weight", type=str2bool, default=None)
ap.add_argument("--weight_mode", default="", choices=["", "linear", "logistic", "exp"])
ap.add_argument("--soft_scale", type=float, default=-1.0)
ap.add_argument("--score_gamma", type=float, default=-1.0)
ap.add_argument("--save_vis", type=str2bool, default=True, help="Whether to save per-frame visualization images.")
args = ap.parse_args()

cfg = load_cfg(args.cfg)
paths = get_common_paths(cfg)

SEQ_NAME = cfg["dataset"]["name"]

IMG_WHITE_DIR = paths["white_dir"]      # Step6 生成的白底图
MASK_DIR      = paths["mask_dir"]       # Step6 生成的二值掩膜

OUT_ROOT  = paths["step7_root"]
if args.out_tag:
    OUT_ROOT = os.path.join(OUT_ROOT, args.out_tag)
OUT_A_DIR = os.path.join(OUT_ROOT, "a_uniform_on_white")      # 图 3-8(a)
OUT_B_DIR = os.path.join(OUT_ROOT, "b_uniform_after_delete")  # 图 3-8(b)
OUT_KP_DIR = os.path.join(OUT_ROOT, "kps_npz")                # 存放 all_pts / kept_pts

if args.save_vis:
    os.makedirs(OUT_A_DIR, exist_ok=True)
    os.makedirs(OUT_B_DIR, exist_ok=True)
os.makedirs(OUT_KP_DIR, exist_ok=True)

# ORB / 四叉树相关参数
N_TOTAL     = cfg["orb"]["n_features"]
N_LEVELS    = cfg["orb"]["n_levels"]
SCALE       = cfg["orb"]["scale_factor"]
FAST_TH     = cfg["orb"]["fast_threshold"]
OVER_FACTOR = cfg["orb"]["over_factor"]
CANDI_MIN   = cfg["orb"]["candi_min"]
CANDI_K     = cfg["orb"]["candi_k"]

RADIUS_BORDER = cfg["mask"]["border_radius"]
USE_QT_PLUS   = cfg["quadtree"]["use_plus"]
QT_MAX_DEPTH     = cfg["quadtree"]["max_depth"]
QT_MIN_PTS_SPLIT = cfg["quadtree"]["min_pts_split"]
QT_MIN_SIDE      = cfg["quadtree"]["min_side"]
QT_MODE        = cfg["quadtree"].get("mode", "plus" if USE_QT_PLUS else "plain")
QT_GRID_ROWS   = cfg["quadtree"].get("grid_rows", 0)
QT_GRID_COLS   = cfg["quadtree"].get("grid_cols", 0)
QT_SCORE_ALPHA = cfg["quadtree"].get("score_alpha", 1.0)
QT_SCORE_BETA  = cfg["quadtree"].get("score_beta", 0.0)
QT_BASE_QUOTA  = cfg["quadtree"].get("base_quota", 1)
# 语义 / 静态感知加权的配置（可选）
SEM_CFG = cfg.get("semantic", {})
SEM_USE_DT_WEIGHT = SEM_CFG.get("use_dt_weight", False)
SEM_DT_WEIGHT     = float(SEM_CFG.get("dt_weight", 0.0))
SEM_WEIGHT_MODE   = SEM_CFG.get("weight_mode", "linear")
SEM_BAND_RADIUS   = int(SEM_CFG.get("band_radius", RADIUS_BORDER))
SEM_SOFT_SCALE    = float(SEM_CFG.get("soft_scale", max(1.0, float(RADIUS_BORDER))))
SEM_SCORE_GAMMA   = float(SEM_CFG.get("score_gamma", 1.0))
GRID_ROWS         = int(cfg.get("grid_eval", {}).get("rows", 8))
GRID_COLS         = int(cfg.get("grid_eval", {}).get("cols", 8))

if args.qt_mode:
    QT_MODE = args.qt_mode
if args.border_radius >= 0:
    RADIUS_BORDER = int(args.border_radius)
    SEM_BAND_RADIUS = int(args.border_radius)
if args.use_dt_weight is not None:
    SEM_USE_DT_WEIGHT = bool(args.use_dt_weight)
if args.weight_mode:
    SEM_WEIGHT_MODE = args.weight_mode
if args.soft_scale > 0:
    SEM_SOFT_SCALE = float(args.soft_scale)
if args.score_gamma > 0:
    SEM_SCORE_GAMMA = float(args.score_gamma)

def main():
    # Step6 输出的白底图统一为 .png
    img_paths = sorted(glob(os.path.join(IMG_WHITE_DIR, "*.png")))
    if not img_paths:
        print("[ERROR] no white images found in", IMG_WHITE_DIR)
        return

    start_frame = max(0, int(args.start_frame))
    if start_frame > 0:
        img_paths = img_paths[start_frame:]
    if args.max_frames > 0:
        img_paths = img_paths[: int(args.max_frames)]
    if not img_paths:
        print("[ERROR] no white images selected after applying start/max frame filters.")
        return

    print(f"[Step7] 序列: {SEQ_NAME}")
    print("  IMG_WHITE_DIR:", IMG_WHITE_DIR)
    print("  MASK_DIR     :", MASK_DIR)
    print("  OUT_ROOT     :", OUT_ROOT)
    print("  START_FRAME  :", start_frame)
    print("  NUM_FRAMES   :", len(img_paths))

    all_counts_before = []
    all_counts_after  = []
    frame_rows = []

    # 根据 mode 选择四叉树 / Grid+四叉树 的实现
    if QT_MODE == "plain":
        def uniform_func(kps, img_shape, N_i):
            return quadtree_uniform(kps, img_shape, N_i)

    elif QT_MODE == "plus":
        def uniform_func(kps, img_shape, N_i):
            return quadtree_uniform_plus(
                kps,
                img_shape,
                N_i,
                max_depth=QT_MAX_DEPTH,
                min_pts_split=QT_MIN_PTS_SPLIT,
                min_side=QT_MIN_SIDE,
            )

    elif QT_MODE == "grid_plus":
        def uniform_func(kps, img_shape, N_i):
            return grid_quadtree_uniform(
                kps,
                img_shape,
                N_i,
                grid_rows=QT_GRID_ROWS,
                grid_cols=QT_GRID_COLS,
                score_alpha=QT_SCORE_ALPHA,
                score_beta=QT_SCORE_BETA,
                base_quota=QT_BASE_QUOTA,
                qt_params=dict(
                    max_depth=QT_MAX_DEPTH,
                    min_pts_split=QT_MIN_PTS_SPLIT,
                    min_side=QT_MIN_SIDE,
                ),
            )
    else:
        # 兜底：默认走 plus
        def uniform_func(kps, img_shape, N_i):
            return quadtree_uniform_plus(
                kps,
                img_shape,
                N_i,
                max_depth=QT_MAX_DEPTH,
                min_pts_split=QT_MIN_PTS_SPLIT,
                min_side=QT_MIN_SIDE,
            )


    for img_path in tqdm(img_paths, desc="Step7 masked uniform on sequence"):
        name = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(MASK_DIR, name + ".png")

        img_white = cv2.imread(img_path)
        mask      = cv2.imread(mask_path, 0)

        if img_white is None:
            print(f"[WARN] cannot read image: {img_path}, skip")
            continue

        if mask is None:
            # 如果某帧没有 mask，就认为没有动态目标：mask 全 0
            h, w = img_white.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)

        gray = cv2.cvtColor(img_white, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape[:2]
        mask_dyn_ratio = float(np.mean(mask == 255))
        dist_full, rel_full = build_reliability_map(
            mask=mask,
            use_dt_weight=bool(SEM_USE_DT_WEIGHT and SEM_DT_WEIGHT > 0.0),
            band_radius=SEM_BAND_RADIUS,
            weight_mode=SEM_WEIGHT_MODE,
            soft_scale=SEM_SOFT_SCALE,
        )

        # ---- 1. 金字塔 + 每层配额 ----
        pyr = build_pyramid(gray, n_levels=N_LEVELS, scale=SCALE)
        Ns  = compute_N_per_level(H, W, N_LEVELS, SCALE, N_TOTAL)

        # ---- 2. 每层提取 ORB 候选点 ----
        cand_kps_per_level = detect_orb_candidates_per_level(
            pyr,
            Ns,
            over_factor=OVER_FACTOR,
            fast_th=FAST_TH,
            candi_min=CANDI_MIN,
            candi_k=CANDI_K,
        )

        # ---- 2.5 静态区域距离图 + 响应重排（可选）----
        if SEM_USE_DT_WEIGHT and SEM_DT_WEIGHT > 0.0:
            rel_pyr = []
            for lvl_img in pyr:
                h_l, w_l = lvl_img.shape[:2]
                rel_l = cv2.resize(rel_full, (w_l, h_l), interpolation=cv2.INTER_LINEAR)
                rel_pyr.append(rel_l)

            for kps_cand, rel_l in zip(cand_kps_per_level, rel_pyr):
                for kp in kps_cand:
                    x, y = kp.pt  # 当前层坐标
                    xi = int(round(x))
                    yi = int(round(y))
                    if (
                        xi < 0 or yi < 0
                        or xi >= rel_l.shape[1]
                        or yi >= rel_l.shape[0]
                    ):
                        continue

                    rel_val = float(rel_l[yi, xi])
                    kp.response = kp.response * (1.0 + SEM_DT_WEIGHT * (rel_val ** SEM_SCORE_GAMMA))

        # ---- 3. 四叉树均匀化（在白底图上）----
        all_kps = []
        for lvl, (lvl_img, kps_cand, N_i) in enumerate(
            zip(pyr, cand_kps_per_level, Ns)
        ):
            if N_i <= 0 or len(kps_cand) == 0:
                continue
            kps_uniform_lvl = uniform_func(
                kps_cand,
                lvl_img.shape[:2],
                int(N_i),
            )
            kps_up = upscale_keypoints(kps_uniform_lvl, level=lvl, scale=SCALE)
            all_kps.extend(kps_up)


        all_counts_before.append(len(all_kps))

        # ---- 4. 可视化 (a)：白底图 + 四叉树均匀化 ----
        if args.save_vis:
            img_a = img_white.copy()
            for kp in all_kps:
                x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
                cv2.circle(img_a, (x, y), 2, (0, 255, 0), -1)
            cv2.imwrite(os.path.join(OUT_A_DIR, name + ".png"), img_a)

        # ---- 5. 根据掩膜删除边缘附近特征点 ----
        kps_kept, kept_idx = remove_kps_near_mask_border(
            all_kps, mask, radius=RADIUS_BORDER
        )
        all_counts_after.append(len(kps_kept))

        # ---- 6. 可视化 (b)：掩膜边缘特征点剔除后的结果 ----
        if args.save_vis:
            img_b = img_white.copy()
            for kp in kps_kept:
                x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
                cv2.circle(img_b, (x, y), 2, (0, 255, 0), -1)
            cv2.imwrite(os.path.join(OUT_B_DIR, name + ".png"), img_b)

        # ---- 7. 保存 all_pts / kept_pts 到 npz ----
        all_pts = np.array([[kp.pt[0], kp.pt[1]] for kp in all_kps], dtype=np.float32)
        kept_pts = np.array([[kp.pt[0], kp.pt[1]] for kp in kps_kept], dtype=np.float32)
        all_fields = sample_point_fields(all_pts, mask, dist_full, rel_full, SEM_BAND_RADIUS)
        kept_fields = sample_point_fields(kept_pts, mask, dist_full, rel_full, SEM_BAND_RADIUS)
        all_grid_metrics = compute_grid_metrics(all_pts, H, W, GRID_ROWS, GRID_COLS)
        kept_grid_metrics = compute_grid_metrics(kept_pts, H, W, GRID_ROWS, GRID_COLS)

        np.savez(
            os.path.join(OUT_KP_DIR, name + ".npz"),
            all_pts=all_pts,
            kept_pts=kept_pts,
            all_r=all_fields["rel"],
            kept_r=kept_fields["rel"],
            all_dist=all_fields["dist"],
            kept_dist=kept_fields["dist"],
            all_dyn=all_fields["dyn"],
            kept_dyn=kept_fields["dyn"],
            all_in_band=all_fields["band"],
            kept_in_band=kept_fields["band"],
            mask_dyn_ratio=np.asarray([mask_dyn_ratio], dtype=np.float32),
            band_radius=np.asarray([SEM_BAND_RADIUS], dtype=np.int32),
        )

        row = {
            "frame_index": len(frame_rows),
            "frame_name": name,
        }
        row.update(
            point_metrics_row(
                frame_index=len(frame_rows),
                frame_name=name,
                points=all_pts,
                fields=all_fields,
                mask_dyn_ratio=mask_dyn_ratio,
                grid_metrics=all_grid_metrics,
                prefix="all",
            )
        )
        row.update(
            point_metrics_row(
                frame_index=len(frame_rows),
                frame_name=name,
                points=kept_pts,
                fields=kept_fields,
                mask_dyn_ratio=mask_dyn_ratio,
                grid_metrics=kept_grid_metrics,
                prefix="kept",
            )
        )
        frame_rows.append(row)

    # ====== 全序列统计：平均特征点数量 ======
    if all_counts_before:
        all_counts_before = np.array(all_counts_before, dtype=np.float32)
        all_counts_after  = np.array(all_counts_after,  dtype=np.float32)

        mean_before = float(all_counts_before.mean())
        mean_after  = float(all_counts_after.mean())
        frame_metrics_csv = os.path.join(OUT_ROOT, "frame_metrics.csv")
        summary_csv = os.path.join(OUT_ROOT, "summary_metrics.csv")

        print("\n[Step7 Summary]")
        print(f"mean #kps before border removal: {mean_before:.1f}")
        print(f"mean #kps after  border removal: {mean_after:.1f}")
        print(f"average ratio kept: {mean_after / max(mean_before, 1e-6) * 100:.1f}%")

        if frame_rows:
            fieldnames = list(frame_rows[0].keys())
            with open(frame_metrics_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(frame_rows)

            summary_row = {
                "num_frames": len(frame_rows),
                "mean_kps_before": mean_before,
                "mean_kps_after": mean_after,
                "ratio_kept": mean_after / max(mean_before, 1e-6),
            }
            numeric_keys = [k for k in frame_rows[0].keys() if k not in ("frame_name",)]
            for key in numeric_keys:
                vals = []
                for row in frame_rows:
                    val = row.get(key)
                    if isinstance(val, (int, float, np.integer, np.floating)):
                        vals.append(float(val))
                if vals:
                    summary_row[f"mean_{key}"] = float(np.mean(vals))

            with open(summary_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
                writer.writeheader()
                writer.writerow(summary_row)

        stats_path = os.path.join(OUT_ROOT, "kps_count_summary.txt")
        with open(stats_path, "w") as f:
            f.write(f"mean_kps_before\t{mean_before:.3f}\n")
            f.write(f"mean_kps_after\t{mean_after:.3f}\n")
            f.write(f"ratio_kept\t{mean_after / max(mean_before, 1e-6):.3f}\n")
        print(f"[Step7] saved frame metrics to {frame_metrics_csv}")
        print(f"[Step7] saved summary metrics to {summary_csv}")
        print(f"[Step7] saved counts to {stats_path}")
    else:
        print("[Step7] no frames processed.")


if __name__ == "__main__":
    main()
