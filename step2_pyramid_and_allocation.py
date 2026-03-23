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

from config_utils import load_cfg, get_common_paths
from feature_uniform_utils import build_pyramid, compute_N_per_level


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

    img_paths = sorted(glob.glob(os.path.join(img_dir, img_ext)))
    if not img_paths:
        print("没有找到图片，检查 cfg.dataset 配置")
        return

    idx = min(max(0, int(args.frame_index)), len(img_paths) - 1)
    img_path = img_paths[idx]

    img = cv2.imread(img_path)
    if img is None:
        print("读取图片失败:", img_path)
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    pyr = build_pyramid(gray, n_levels=n_levels, scale=scale)
    n_per = compute_N_per_level(h, w, n_levels, scale, n_total)

    eval_root = paths["eval_root"]
    if args.out_tag:
        eval_root = os.path.join(eval_root, args.out_tag)
    os.makedirs(eval_root, exist_ok=True)
    out_txt = os.path.join(eval_root, "step2_pyramid_and_allocation.txt")

    print(f"[Step2] cfg={args.cfg}")
    print(f"[Step2] 样例图: {img_path}")
    print(f"[Step2] 原图尺寸: H={h}, W={w}")
    print(f"[Step2] N_total={n_total}, n_levels={n_levels}, scale={scale}")

    for i, im in enumerate(pyr):
        hi, wi = im.shape[:2]
        print(f"  level {i}: H={hi}, W={wi}, N_i={n_per[i]}")

    print(f"sum(N_i)={sum(n_per)}")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"cfg\t{args.cfg}\n")
        f.write(f"frame\t{os.path.basename(img_path)}\n")
        f.write(f"orig_h\t{h}\n")
        f.write(f"orig_w\t{w}\n")
        f.write(f"n_total\t{n_total}\n")
        f.write(f"n_levels\t{n_levels}\n")
        f.write(f"scale\t{scale}\n")
        for i, im in enumerate(pyr):
            hi, wi = im.shape[:2]
            f.write(f"level_{i}\tH={hi},W={wi},N_i={n_per[i]}\n")
        f.write(f"sum_N_i\t{sum(n_per)}\n")
    print(f"[Step2] 已输出: {out_txt}")


if __name__ == "__main__":
    main()
