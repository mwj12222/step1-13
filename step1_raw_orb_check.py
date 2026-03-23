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

import os
import glob
import cv2

from config_utils import load_cfg, get_common_paths


def main():
    cfg_in_pipeline = PROJECT_ROOT / "configs" / "pipeline" / "config_kitti_00.yaml"
    default_cfg = str(cfg_in_pipeline) if cfg_in_pipeline.exists() else str(
        PROJECT_ROOT / "configs" / "reference" / "config_kitti_00.yaml"
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default=default_cfg)
    ap.add_argument("--max_frames", type=int, default=20)
    ap.add_argument("--start_frame", type=int, default=0)
    ap.add_argument("--frame_stride", type=int, default=1)
    ap.add_argument("--out_tag", default="", help="Optional subdir tag under step1_raw_vis.")
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    paths = get_common_paths(cfg)

    img_dir = paths["img_dir"]
    img_ext = cfg["dataset"]["img_ext"]
    out_dir = paths["step1_raw_vis"]
    if args.out_tag:
        out_dir = os.path.join(out_dir, args.out_tag)
    os.makedirs(out_dir, exist_ok=True)

    n_features = int(cfg["orb"]["n_features"])
    n_levels = int(cfg["orb"]["n_levels"])
    scale = float(cfg["orb"]["scale_factor"])

    img_paths = sorted(glob.glob(os.path.join(img_dir, img_ext)))
    if not img_paths:
        print("没有找到图片，检查 cfg.dataset.root/name/img_subdir/img_ext")
        print("img_dir:", img_dir)
        return

    start_frame = max(0, int(args.start_frame))
    stride = max(1, int(args.frame_stride))
    img_paths = img_paths[start_frame::stride]
    if not img_paths:
        print("筛选帧后没有可用图片")
        return

    print(f"[Step1] cfg={args.cfg}")
    print(f"[Step1] 找到 {len(img_paths)} 张图，输出到 {out_dir}")

    orb = cv2.ORB_create(
        nfeatures=n_features,
        nlevels=n_levels,
        scaleFactor=scale,
    )

    max_frames = max(1, int(args.max_frames))
    for i, img_path in enumerate(img_paths[:max_frames]):
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] 读取失败: {img_path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kps = orb.detect(gray, mask=None)
        print(f"[{i+1}/{min(len(img_paths), max_frames)}] {os.path.basename(img_path)} -> {len(kps)}")

        vis = cv2.drawKeypoints(
            img,
            kps,
            None,
            color=(0, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT,
        )
        cv2.imwrite(os.path.join(out_dir, os.path.basename(img_path)), vis)


if __name__ == "__main__":
    main()
