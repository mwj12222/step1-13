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
import time
import cv2

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


def draw_keypoints(img_color, kps, color=(0, 255, 0)):
    vis = img_color.copy()
    for kp in kps:
        x, y = kp.pt
        cv2.circle(vis, (int(x), int(y)), 2, color, thickness=-1, lineType=cv2.LINE_AA)
    return vis


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


def process_one_image(img_path, out_raw, out_uni, uniform_func, n_total, n_levels, scale, fast_th, over_factor, candi_min, candi_k):
    img_color = cv2.imread(img_path)
    if img_color is None:
        print(f"[WARN] 读取失败，跳过: {img_path}")
        return False

    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    orb_raw = cv2.ORB_create(
        nfeatures=n_total,
        scaleFactor=scale,
        nlevels=n_levels,
        fastThreshold=fast_th,
    )
    kps_raw = orb_raw.detect(gray, mask=None)

    base = os.path.basename(img_path)
    cv2.imwrite(os.path.join(out_raw, base), draw_keypoints(img_color, kps_raw))

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

    all_uniform_kps_on_base = []
    for lvl, (img_l, cand_kps, n_i) in enumerate(zip(pyr, cand_kps_per_level, n_per)):
        if n_i <= 0 or len(cand_kps) == 0:
            continue
        kps_uniform = uniform_func(cand_kps, img_l.shape[:2], int(n_i))
        kps_up = upscale_keypoints(kps_uniform, level=lvl, scale=scale)
        all_uniform_kps_on_base.extend(kps_up)

    cv2.imwrite(os.path.join(out_uni, base), draw_keypoints(img_color, all_uniform_kps_on_base))
    return True


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
    ap.add_argument("--out_tag", default="", help="Optional subdir tag under step3_root.")
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    paths = get_common_paths(cfg)

    img_dir = paths["img_dir"]
    img_ext = cfg["dataset"]["img_ext"]

    out_dir = paths["step3_root"]
    if args.out_tag:
        out_dir = os.path.join(out_dir, args.out_tag)
    out_raw = os.path.join(out_dir, "raw_orb")
    out_uni = os.path.join(out_dir, "uniform_orb")
    os.makedirs(out_raw, exist_ok=True)
    os.makedirs(out_uni, exist_ok=True)

    n_total = int(cfg["orb"]["n_features"])
    n_levels = int(cfg["orb"]["n_levels"])
    scale = float(cfg["orb"]["scale_factor"])
    fast_th = int(cfg["orb"]["fast_threshold"])
    over_factor = int(cfg["orb"]["over_factor"])
    candi_min = int(cfg["orb"]["candi_min"])
    candi_k = int(cfg["orb"]["candi_k"])

    uniform_func = build_uniform_func(cfg)

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

    print(f"[Step3] cfg={args.cfg}")
    print(f"[Step3] 输入: {img_dir}")
    print(f"[Step3] 输出: {out_dir}")
    print(f"[Step3] 帧数: {len(img_paths)}")

    t0 = time.perf_counter()
    n_ok = 0
    for idx, img_path in enumerate(img_paths):
        if idx % 50 == 0:
            print(f"  [{idx+1}/{len(img_paths)}] {os.path.basename(img_path)}")
        ok = process_one_image(
            img_path,
            out_raw,
            out_uni,
            uniform_func,
            n_total,
            n_levels,
            scale,
            fast_th,
            over_factor,
            candi_min,
            candi_k,
        )
        if ok:
            n_ok += 1

    dt = time.perf_counter() - t0
    avg = dt / n_ok if n_ok > 0 else 0.0
    fps = 1.0 / avg if avg > 0 else 0.0

    print("\n[Step3] 完成")
    print(f"  成功帧数: {n_ok}/{len(img_paths)}")
    print(f"  总耗时: {dt:.2f}s")
    print(f"  平均耗时: {avg*1000:.1f}ms ({fps:.1f} FPS)")


if __name__ == "__main__":
    main()
