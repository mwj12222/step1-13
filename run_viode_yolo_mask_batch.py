#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = None
for p in [THIS_FILE.parent, *THIS_FILE.parent.parents]:
    if (p / "configs").is_dir() and ((p / "pipelines").is_dir() or (p / " pipelines").is_dir()):
        PROJECT_ROOT = p
        break
if PROJECT_ROOT is None:
    raise RuntimeError(f"Cannot locate project root from {THIS_FILE}")

sys.path.insert(0, str(PROJECT_ROOT / "src" / "common"))

from config_utils import load_cfg


YOLO_ROOT_DEFAULT = Path("/home/ml/projects/yolov7-segmentation")
WEIGHTS_DEFAULT = YOLO_ROOT_DEFAULT / "yolov7-seg.pt"
TMP_ROOT_DEFAULT = Path("/tmp/viode_yolo_mask_batch")


def default_cfgs() -> list[str]:
    cfg_dir = PROJECT_ROOT / "configs" / "viode"
    return [str(p) for p in sorted(cfg_dir.glob("config_viode_*_full_compare.yaml"))]


def resolve_img_dir(cfg: dict) -> Path:
    dataset = cfg.get("dataset", {})
    if dataset.get("img_dir"):
        return Path(dataset["img_dir"]).expanduser().resolve()
    root = Path(dataset["root"]).expanduser().resolve()
    name = str(dataset["name"])
    subdir = str(dataset.get("img_subdir", "")).strip()
    return (root / name / subdir).resolve() if subdir else (root / name).resolve()


def resolve_target_mask_dir(cfg: dict) -> Path:
    mask_cfg = cfg.get("mask", {})
    return (Path(mask_cfg["yolo_mask_root"]).expanduser().resolve() / str(mask_cfg["yolo_run_name"])).resolve()


def resolve_target_vis_dir(cfg: dict) -> Path:
    mask_cfg = cfg.get("mask", {})
    mask_root = Path(mask_cfg["yolo_mask_root"]).expanduser().resolve()
    run_name = str(mask_cfg["yolo_run_name"])
    vis_root = mask_root.parent / "yolo_vis"
    return (vis_root / run_name).resolve()


def count_images(img_dir: Path, pattern: str) -> int:
    return len(sorted(img_dir.glob(pattern)))


def copy_tree_flat(src_dir: Path, dst_dir: Path) -> int:
    if not src_dir.is_dir():
        raise FileNotFoundError(f"mask src dir not found: {src_dir}")
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src_path in sorted(src_dir.glob("*.png")):
        dst_path = dst_dir / src_path.name
        try:
            shutil.copy2(src_path, dst_path)
        except PermissionError:
            # drvfs can reject metadata updates (e.g. utime/copystat) even when
            # the file payload itself is writable. Fall back to content-only copy.
            shutil.copyfile(src_path, dst_path)
        copied += 1
    return copied


def copy_predict_outputs(src_dir: Path, dst_dir: Path) -> int:
    if not src_dir.is_dir():
        raise FileNotFoundError(f"predict src dir not found: {src_dir}")
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src_path in sorted(src_dir.iterdir()):
        if src_path.is_file():
            dst_path = dst_dir / src_path.name
            try:
                shutil.copy2(src_path, dst_path)
            except PermissionError:
                shutil.copyfile(src_path, dst_path)
            copied += 1
    return copied


def build_predict_cmd(
    python_bin: str,
    yolo_root: Path,
    weights: Path,
    source_dir: Path,
    tmp_root: Path,
    run_name: str,
    imgsz: int,
    conf_thres: float,
    iou_thres: float,
    device: str,
    half: bool,
) -> list[str]:
    cmd = [
        python_bin,
        str(yolo_root / "segment" / "predict.py"),
        "--weights",
        str(weights),
        "--source",
        str(source_dir),
        "--project",
        str(tmp_root / "predict-seg"),
        "--name",
        run_name,
        "--imgsz",
        str(imgsz),
        "--conf-thres",
        str(conf_thres),
        "--iou-thres",
        str(iou_thres),
        "--exist-ok",
    ]
    if device:
        cmd.extend(["--device", device])
    if half:
        cmd.append("--half")
    return cmd


def run_one(
    cfg_path: Path,
    python_bin: str,
    yolo_root: Path,
    weights: Path,
    tmp_root: Path,
    imgsz: int,
    conf_thres: float,
    iou_thres: float,
    device: str,
    half: bool,
    skip_existing: bool,
) -> dict:
    cfg = load_cfg(str(cfg_path))
    source_dir = resolve_img_dir(cfg)
    target_dir = resolve_target_mask_dir(cfg)
    target_vis_dir = resolve_target_vis_dir(cfg)
    run_name = str(cfg["mask"]["yolo_run_name"])
    img_pattern = str(cfg["dataset"].get("img_ext", "*.png"))

    if not source_dir.is_dir():
        raise FileNotFoundError(f"image dir not found: {source_dir}")

    num_source = count_images(source_dir, img_pattern)
    if num_source <= 0:
        raise RuntimeError(f"no images found in {source_dir} with pattern {img_pattern}")

    existing_masks = len(list(target_dir.glob("*.png"))) if target_dir.is_dir() else 0
    existing_vis = len(list(target_vis_dir.glob("*.png"))) if target_vis_dir.is_dir() else 0
    if skip_existing and existing_masks == num_source and existing_vis == num_source:
        return {
            "config": str(cfg_path),
            "run_name": run_name,
            "source_dir": str(source_dir),
            "target_dir": str(target_dir),
            "target_vis_dir": str(target_vis_dir),
            "num_source_images": int(num_source),
            "num_masks_written": int(existing_masks),
            "num_vis_written": int(existing_vis),
            "status": "skipped_existing",
            "duration_sec": 0.0,
        }

    work_root = tmp_root / run_name
    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True, exist_ok=True)

    cmd = build_predict_cmd(
        python_bin=python_bin,
        yolo_root=yolo_root,
        weights=weights,
        source_dir=source_dir,
        tmp_root=work_root,
        run_name=run_name,
        imgsz=imgsz,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        device=device,
        half=half,
    )

    print(f"[MASK] cfg={cfg_path}")
    print(f"[MASK] src={source_dir}")
    print(f"[MASK] dst={target_dir}")
    print(f"[MASK] vis={target_vis_dir}")
    print(f"[MASK] cmd={' '.join(cmd)}")

    t0 = time.time()
    env = os.environ.copy()
    # PyTorch >= 2.6 defaults torch.load(..., weights_only=True), which breaks
    # legacy YOLOv7 checkpoints unless we explicitly opt back into the old behavior.
    env.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    subprocess.run(cmd, check=True, cwd=str(yolo_root), env=env)
    src_mask_dir = work_root / "masks" / run_name
    src_vis_dir = work_root / "predict-seg" / run_name
    if target_dir.exists():
        shutil.rmtree(target_dir)
    if target_vis_dir.exists():
        shutil.rmtree(target_vis_dir)
    copied = copy_tree_flat(src_mask_dir, target_dir)
    copied_vis = copy_predict_outputs(src_vis_dir, target_vis_dir)
    dt = time.time() - t0

    summary = {
        "config": str(cfg_path),
        "run_name": run_name,
        "source_dir": str(source_dir),
        "target_dir": str(target_dir),
        "target_vis_dir": str(target_vis_dir),
        "num_source_images": int(num_source),
        "num_masks_written": int(copied),
        "num_vis_written": int(copied_vis),
        "status": "ok",
        "duration_sec": float(dt),
        "weights": str(weights),
        "imgsz": int(imgsz),
        "conf_thres": float(conf_thres),
        "iou_thres": float(iou_thres),
        "device": str(device),
        "half": bool(half),
    }

    with open(target_dir / "_mask_run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if copied != num_source:
        raise RuntimeError(
            f"mask count mismatch for {run_name}: images={num_source}, masks={copied}, target={target_dir}"
        )
    if copied_vis != num_source:
        raise RuntimeError(
            f"visual count mismatch for {run_name}: images={num_source}, vis={copied_vis}, target={target_vis_dir}"
        )
    return summary


def main():
    ap = argparse.ArgumentParser(
        description="Batch-run YOLOv7 segmentation masks for VIODE cam0 sequences and save masks to the final G:/Result layout."
    )
    ap.add_argument(
        "--cfg",
        action="append",
        default=[],
        help="Repeatable config path. Defaults to all VIODE full-compare configs in this workspace.",
    )
    ap.add_argument("--python_bin", default=sys.executable)
    ap.add_argument("--yolo_root", default=str(YOLO_ROOT_DEFAULT))
    ap.add_argument("--weights", default=str(WEIGHTS_DEFAULT))
    ap.add_argument("--tmp_root", default=str(TMP_ROOT_DEFAULT))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf_thres", type=float, default=0.25)
    ap.add_argument("--iou_thres", type=float, default=0.45)
    ap.add_argument("--device", default="")
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()

    cfgs = args.cfg or default_cfgs()

    yolo_root = Path(args.yolo_root).expanduser().resolve()
    weights = Path(args.weights).expanduser().resolve()
    tmp_root = Path(args.tmp_root).expanduser().resolve()

    if not yolo_root.is_dir():
        raise FileNotFoundError(f"yolo_root not found: {yolo_root}")
    if not weights.is_file():
        raise FileNotFoundError(f"weights not found: {weights}")

    all_rows = []
    for cfg_path in cfgs:
        row = run_one(
            cfg_path=Path(cfg_path).expanduser().resolve(),
            python_bin=args.python_bin,
            yolo_root=yolo_root,
            weights=weights,
            tmp_root=tmp_root,
            imgsz=int(args.imgsz),
            conf_thres=float(args.conf_thres),
            iou_thres=float(args.iou_thres),
            device=str(args.device),
            half=bool(args.half),
            skip_existing=bool(args.skip_existing),
        )
        all_rows.append(row)

    print("[MASK] batch finished")
    for row in all_rows:
        print(
            f"[MASK] {row['run_name']}: status={row['status']} "
            f"images={row['num_source_images']} masks={row['num_masks_written']} "
            f"dst={row['target_dir']}"
        )


if __name__ == "__main__":
    main()
