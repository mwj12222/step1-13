#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import math

import cv2
import numpy as np
import yaml


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def clean_row_keys(row: dict[str, str]) -> dict[str, str]:
    return {str(k).strip(): v for k, v in row.items()}


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def ns_to_sec(ns_text: str) -> float:
    return float(int(str(ns_text).strip())) * 1e-9


def quat_wxyz_to_rot(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    q = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = (q / n).tolist()
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def rot_to_quat_xyzw(R: np.ndarray) -> tuple[float, float, float, float]:
    tr = float(np.trace(R))
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    q = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    q /= max(float(np.linalg.norm(q)), 1e-12)
    return float(q[0]), float(q[1]), float(q[2]), float(q[3])


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def build_undistort_camera(cam_cfg: dict, width: int, height: int, balance: float) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    intr = cam_cfg["intrinsics"]
    K = np.array(
        [
            [float(intr[0]), 0.0, float(intr[2])],
            [0.0, float(intr[1]), float(intr[3])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    D = np.asarray(cam_cfg["distortion_coeffs"], dtype=np.float64).reshape(-1, 1)
    size = (int(width), int(height))
    P = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, size, np.eye(3), balance=float(balance))
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), P, size, cv2.CV_16SC2)
    return P.astype(np.float64), map1, map2


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Prepare one TUM VI cam0 sequence into VIODE-style step1~11 inputs.")
    ap.add_argument("--tumvi_root", required=True, help="Inner TUM VI root containing mav0/ and dso/.")
    ap.add_argument("--out_root", required=True, help="Output root containing <seq_name>/rgb + csv sidecars.")
    ap.add_argument("--seq_name", required=True, help="Prepared sequence name under out_root, e.g. room2_cam0.")
    ap.add_argument("--camera", default="cam0", choices=["cam0", "cam1"])
    ap.add_argument("--image_mode", default="undistort_pinhole", choices=["copy_raw", "undistort_pinhole"])
    ap.add_argument("--undistort_balance", type=float, default=0.0)
    ap.add_argument("--start_frame", type=int, default=0)
    ap.add_argument("--max_frames", type=int, default=-1)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    tumvi_root = Path(args.tumvi_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    seq_root = out_root / str(args.seq_name)
    rgb_dir = seq_root / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)

    cam = str(args.camera)
    cam_csv = tumvi_root / "mav0" / cam / "data.csv"
    cam_data_dir = tumvi_root / "mav0" / cam / "data"
    imu_csv = tumvi_root / "mav0" / "imu0" / "data.csv"
    mocap_csv = tumvi_root / "mav0" / "mocap0" / "data.csv"
    camchain_yaml = tumvi_root / "dso" / "camchain.yaml"

    if not cam_csv.is_file():
        raise FileNotFoundError(f"camera csv not found: {cam_csv}")
    if not imu_csv.is_file():
        raise FileNotFoundError(f"imu csv not found: {imu_csv}")
    if not mocap_csv.is_file():
        raise FileNotFoundError(f"mocap csv not found: {mocap_csv}")
    if not camchain_yaml.is_file():
        raise FileNotFoundError(f"camchain yaml not found: {camchain_yaml}")

    cam_rows_all = load_csv_rows(cam_csv)
    start = max(0, int(args.start_frame))
    if start >= len(cam_rows_all):
        raise ValueError(f"start_frame out of range: {start} >= {len(cam_rows_all)}")
    if int(args.max_frames) > 0:
        cam_rows = cam_rows_all[start:start + int(args.max_frames)]
    else:
        cam_rows = cam_rows_all[start:]

    imu_rows_raw = load_csv_rows(imu_csv)
    mocap_rows_raw = load_csv_rows(mocap_csv)
    camchain = yaml.safe_load(camchain_yaml.read_text(encoding="utf-8"))
    if cam not in camchain:
        raise KeyError(f"{cam} not found in {camchain_yaml}")

    T_cam_imu = np.asarray(camchain[cam]["T_cam_imu"], dtype=np.float64)
    T_imu_cam = inv_T(T_cam_imu)

    first_row = clean_row_keys(cam_rows[0])
    first_img = cv2.imread(str(cam_data_dir / str(first_row["filename"]).strip()), cv2.IMREAD_UNCHANGED)
    if first_img is None:
        raise RuntimeError("failed to read first source image")
    src_h, src_w = first_img.shape[:2]
    src_ch = 1 if first_img.ndim == 2 else int(first_img.shape[2])

    undistort_K = None
    map1 = None
    map2 = None
    if str(args.image_mode) == "undistort_pinhole":
        undistort_K, map1, map2 = build_undistort_camera(
            cam_cfg=camchain[cam],
            width=src_w,
            height=src_h,
            balance=float(args.undistort_balance),
        )

    image_rows: list[dict] = []
    first_stamp = None
    out_w = None
    out_h = None
    out_ch = None

    for local_index, raw_row in enumerate(cam_rows):
        row = clean_row_keys(raw_row)
        global_index = start + local_index
        stamp_ns = int(str(row["#timestamp [ns]"]).strip())
        src_name = str(row["filename"]).strip()
        src_path = cam_data_dir / src_name
        if not src_path.is_file():
            raise FileNotFoundError(f"image not found: {src_path}")

        frame_name = f"{local_index:06d}"
        dst_path = rgb_dir / f"{frame_name}.png"
        img = cv2.imread(str(src_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"cv2.imread failed: {src_path}")
        if str(args.image_mode) == "undistort_pinhole":
            out_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        else:
            out_img = img

        if bool(args.overwrite) or not dst_path.exists():
            if not cv2.imwrite(str(dst_path), out_img):
                raise RuntimeError(f"cv2.imwrite failed: {dst_path}")

        stamp_sec = float(stamp_ns) * 1e-9
        if first_stamp is None:
            first_stamp = stamp_sec
            out_h, out_w = out_img.shape[:2]
            out_ch = 1 if out_img.ndim == 2 else int(out_img.shape[2])

        image_rows.append(
            {
                "frame_index": int(local_index),
                "frame_name": frame_name,
                "stamp_sec": f"{stamp_sec:.9f}",
                "rel_sec": f"{(stamp_sec - first_stamp):.9f}",
                "topic": f"/mav0/{cam}/image_raw",
                "path": str(dst_path),
                "width": int(out_w),
                "height": int(out_h),
                "channels": int(out_ch),
                "source_filename": src_name,
                "source_timestamp_ns": stamp_ns,
                "source_frame_index": int(global_index),
            }
        )

    imu_rows: list[dict] = []
    for raw_row in imu_rows_raw:
        row = clean_row_keys(raw_row)
        stamp_sec = ns_to_sec(row["#timestamp [ns]"])
        imu_rows.append(
            {
                "stamp_sec": f"{stamp_sec:.9f}",
                "gx": float(row["w_RS_S_x [rad s^-1]"]),
                "gy": float(row["w_RS_S_y [rad s^-1]"]),
                "gz": float(row["w_RS_S_z [rad s^-1]"]),
                "ax": float(row["a_RS_S_x [m s^-2]"]),
                "ay": float(row["a_RS_S_y [m s^-2]"]),
                "az": float(row["a_RS_S_z [m s^-2]"]),
            }
        )

    gt_rows: list[dict] = []
    for raw_row in mocap_rows_raw:
        row = clean_row_keys(raw_row)
        stamp_sec = ns_to_sec(row["#timestamp [ns]"])
        R_ref_imu = quat_wxyz_to_rot(
            float(row["q_RS_w []"]),
            float(row["q_RS_x []"]),
            float(row["q_RS_y []"]),
            float(row["q_RS_z []"]),
        )
        t_ref_imu = np.array(
            [
                float(row["p_RS_R_x [m]"]),
                float(row["p_RS_R_y [m]"]),
                float(row["p_RS_R_z [m]"]),
            ],
            dtype=np.float64,
        )
        T_ref_imu = make_T(R_ref_imu, t_ref_imu)
        T_ref_cam = T_ref_imu @ T_imu_cam
        qx, qy, qz, qw = rot_to_quat_xyzw(T_ref_cam[:3, :3])
        gt_rows.append(
            {
                "stamp_sec": f"{stamp_sec:.9f}",
                "px": float(T_ref_cam[0, 3]),
                "py": float(T_ref_cam[1, 3]),
                "pz": float(T_ref_cam[2, 3]),
                "qx": qx,
                "qy": qy,
                "qz": qz,
                "qw": qw,
            }
        )

    write_csv(
        seq_root / "timestamps.csv",
        [
            "frame_index",
            "frame_name",
            "stamp_sec",
            "rel_sec",
            "topic",
            "path",
            "width",
            "height",
            "channels",
            "source_filename",
            "source_timestamp_ns",
            "source_frame_index",
        ],
        image_rows,
    )
    write_csv(seq_root / "imu.csv", ["stamp_sec", "gx", "gy", "gz", "ax", "ay", "az"], imu_rows)
    write_csv(seq_root / "gt_pose.csv", ["stamp_sec", "px", "py", "pz", "qx", "qy", "qz", "qw"], gt_rows)

    summary = {
        "tumvi_root": str(tumvi_root),
        "out_root": str(out_root),
        "seq_root": str(seq_root),
        "seq_name": str(args.seq_name),
        "camera": cam,
        "image_mode": str(args.image_mode),
        "undistort_balance": None if str(args.image_mode) != "undistort_pinhole" else float(args.undistort_balance),
        "start_frame": int(start),
        "max_frames": int(args.max_frames),
        "num_images": int(len(image_rows)),
        "num_imu_rows": int(len(imu_rows)),
        "num_gt_rows": int(len(gt_rows)),
        "image_shape": {"width": int(out_w), "height": int(out_h), "channels": int(out_ch)},
        "source_image_shape": {"width": int(src_w), "height": int(src_h), "channels": int(src_ch)},
        "camera_intrinsics_raw": {
            "fx": float(camchain[cam]["intrinsics"][0]),
            "fy": float(camchain[cam]["intrinsics"][1]),
            "cx": float(camchain[cam]["intrinsics"][2]),
            "cy": float(camchain[cam]["intrinsics"][3]),
            "distortion_model": str(camchain[cam]["distortion_model"]),
            "distortion_coeffs": [float(x) for x in camchain[cam]["distortion_coeffs"]],
        },
        "camera_intrinsics_output": {
            "fx": float(undistort_K[0, 0]) if undistort_K is not None else float(camchain[cam]["intrinsics"][0]),
            "fy": float(undistort_K[1, 1]) if undistort_K is not None else float(camchain[cam]["intrinsics"][1]),
            "cx": float(undistort_K[0, 2]) if undistort_K is not None else float(camchain[cam]["intrinsics"][2]),
            "cy": float(undistort_K[1, 2]) if undistort_K is not None else float(camchain[cam]["intrinsics"][3]),
            "k1": 0.0,
            "k2": 0.0,
            "p1": 0.0,
            "p2": 0.0,
            "k3": 0.0,
        },
        "T_cam_imu": [[float(v) for v in row] for row in T_cam_imu.tolist()],
        "T_imu_cam": [[float(v) for v in row] for row in T_imu_cam.tolist()],
    }
    with open(seq_root / "prep_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
