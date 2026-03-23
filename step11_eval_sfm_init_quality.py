#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
step11_eval_sfm_init_quality.py

补充实验 E1：Static-SfM 初始化质量评定。

当前脚本已支持：
- KITTI poses txt
- VIODE 提取目录中的 gt_pose.csv + timestamps.csv

核心思想（与你前面确认的“评定指标”完全一致）：
1) 每个 start（起始帧）跑一次初始化窗口 [start, start+W]。
2) 同一个 start 跑两套：
   - noBA: 关闭 enable_two_view_ba / enable_global_ba（对照组）
   - withBA: 使用配置默认开关（实验组）
3) 输出四类指标：
   A) 两视图候选统计：pivot、E/H、parallax、tri_points、pnp_eval_success等（来自 init_candidates_metrics.json）
   B) 几何一致性：重投影误差 RMS/median/p90、bad_depth_ratio、cheirality_ratio、tri_angle(三角化夹角)
   C) 窗口可用性：pose_coverage、num_poses、num_points、num_observations
   D) 真值对照（尺度无关）：rot_err（deg）、trans_dir_err（deg）、scale_fit（可选诊断）

输出目录：
  <eval_root>/sfm_init_quality_v2/<seq_name>/
    summary.csv
    summary.json
    start_000000/
      noBA/  (npz + init_candidates_metrics.json + per_run_metrics.json)
      withBA/(npz + init_candidates_metrics.json + per_run_metrics.json)
"""

from pathlib import Path
import sys
import os
import json
import csv
import copy
import time
import argparse
from typing import Dict, Tuple, List, Optional

import numpy as np

THIS_FILE = Path(__file__).resolve()
THIS_DIR = THIS_FILE.parent

# 自动向上找项目根目录：以 config_utils.py / config_kitti_00.yaml 是否存在作为标志
PROJECT_ROOT = None
for p in [THIS_DIR, *THIS_DIR.parents]:
    if (p / "configs").is_dir() and ((p / "pipelines").is_dir() or (p / " pipelines").is_dir()):
        PROJECT_ROOT = p
        break
if PROJECT_ROOT is None:
    raise RuntimeError(f"Cannot locate project root from {THIS_FILE}")

sys.path.insert(0, str(PROJECT_ROOT / "src" / "common"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "sfm"))
sys.path.insert(0, str(THIS_DIR))

from config_utils import load_cfg, get_common_paths
from sfm_static_utils import run_static_sfm_v2, get_camera_from_cfg
from q_gate_utils import (
    aggregate_front_window_metrics,
    build_experiment_protocol_record,
    build_q_post_metrics,
    build_q_pre_metrics,
    decide_gate,
    evaluate_post_geom_quality,
    validate_experiment_protocol,
    load_candidate_metrics_json,
    load_step7_frame_metrics,
    summarize_candidate_pool,
)


def resolve_default_cfg():
    cfg_in_pipeline = PROJECT_ROOT / "configs" / "pipeline" / "config_kitti_00.yaml"
    if cfg_in_pipeline.exists():
        return str(cfg_in_pipeline)
    return str(PROJECT_ROOT / "configs" / "reference" / "config_kitti_00.yaml")


# -----------------------------
# GT pose loaders
# -----------------------------
def load_kitti_poses_T0i(pose_txt: str) -> Dict[int, np.ndarray]:
    """
    读取 KITTI odometry poses/xx.txt。
    每行 12 个数，按行优先组成 3x4；补齐成 4x4。
    该 3x4 矩阵（devkit 口径）可理解为：把 i 帧坐标系的点变换到 0 帧坐标系（i -> 0）。
    """
    Ts = {}
    with open(pose_txt, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            nums = [float(x) for x in line.split()]
            if len(nums) != 12:
                raise ValueError(f"Bad pose line (need 12 floats): {line[:80]}")
            T = np.eye(4, dtype=np.float64)
            T[:3, :4] = np.array(nums, dtype=np.float64).reshape(3, 4)
            Ts[int(idx)] = T
    return Ts


def quat_xyzw_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    q = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = (q / n).tolist()
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


def pose_row_to_T_parent_child(row: Dict) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_xyzw_to_rot(
        float(row["qx"]),
        float(row["qy"]),
        float(row["qz"]),
        float(row["qw"]),
    )
    T[:3, 3] = np.array(
        [float(row["px"]), float(row["py"]), float(row["pz"])],
        dtype=np.float64,
    )
    return T


def load_csv_rows(csv_path: str) -> List[Dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def nearest_index(sorted_vals: np.ndarray, x: float) -> int:
    idx = int(np.searchsorted(sorted_vals, x, side="left"))
    if idx <= 0:
        return 0
    if idx >= len(sorted_vals):
        return len(sorted_vals) - 1
    prev_i = idx - 1
    if abs(float(sorted_vals[prev_i]) - x) <= abs(float(sorted_vals[idx]) - x):
        return prev_i
    return idx


def load_viode_gt_T0i(gt_pose_csv: str, timestamps_csv: str) -> Tuple[Dict[int, np.ndarray], Dict[str, float]]:
    """
    将 VIODE 提取目录中的：
    - gt_pose.csv       : stamp_sec, px,py,pz,qx,qy,qz,qw
    - timestamps.csv    : frame_index, frame_name, stamp_sec, ...
    对齐成按图像 frame_index 索引的 gt_T0i。

    返回：
    - gt_by_frame: {global_frame_idx: T_0_i}，含义与 KITTI loader 一致，都是 i -> 0
    - meta: 对齐统计
    """
    gt_rows = load_csv_rows(gt_pose_csv)
    ts_rows = load_csv_rows(timestamps_csv)
    if not gt_rows:
        raise RuntimeError(f"Empty gt_pose.csv: {gt_pose_csv}")
    if not ts_rows:
        raise RuntimeError(f"Empty timestamps.csv: {timestamps_csv}")

    gt_stamps = np.asarray([float(r["stamp_sec"]) for r in gt_rows], dtype=np.float64)
    gt_T_w_i = [pose_row_to_T_parent_child(r) for r in gt_rows]

    frame_entries = []
    dts = []
    for row in ts_rows:
        fi = int(row["frame_index"])
        stamp = float(row["stamp_sec"])
        gi = nearest_index(gt_stamps, stamp)
        dt = abs(float(gt_stamps[gi]) - stamp)
        frame_entries.append((fi, gi, dt))
        dts.append(dt)

    frame_entries.sort(key=lambda x: x[0])
    ref_gt_idx = frame_entries[0][1]
    T_w_0 = gt_T_w_i[ref_gt_idx]
    T_0_w = inv_T(T_w_0)

    gt_by_frame: Dict[int, np.ndarray] = {}
    for fi, gi, _ in frame_entries:
        gt_by_frame[int(fi)] = T_0_w @ gt_T_w_i[gi]

    dt_arr = np.asarray(dts, dtype=np.float64)
    meta = {
        "num_image_frames": float(len(frame_entries)),
        "num_gt_poses": float(len(gt_rows)),
        "match_dt_mean_sec": float(np.mean(dt_arr)),
        "match_dt_med_sec": float(np.median(dt_arr)),
        "match_dt_p90_sec": float(np.percentile(dt_arr, 90)),
        "match_dt_max_sec": float(np.max(dt_arr)),
    }
    return gt_by_frame, meta


def infer_gt_spec(cfg: Dict, gt_path_arg: Optional[str], gt_type_arg: str) -> Tuple[Optional[str], Optional[Dict[str, str]]]:
    dataset = cfg.get("dataset", {})
    seq_root = Path(str(dataset.get("root", ""))).expanduser()
    seq_name = str(dataset.get("name", ""))
    seq_dir = seq_root / seq_name

    if gt_path_arg:
        p = Path(gt_path_arg).expanduser().resolve()
        if gt_type_arg == "kitti":
            return "kitti", {"pose_txt": str(p)}
        if gt_type_arg == "viode":
            if p.name.lower() == "gt_pose.csv":
                return "viode", {
                    "gt_pose_csv": str(p),
                    "timestamps_csv": str((p.parent / "timestamps.csv").resolve()),
                }
            return "viode", {
                "gt_pose_csv": str((p / "gt_pose.csv").resolve()),
                "timestamps_csv": str((p / "timestamps.csv").resolve()),
            }

        if p.suffix.lower() == ".txt":
            return "kitti", {"pose_txt": str(p)}
        if p.suffix.lower() == ".csv":
            return "viode", {
                "gt_pose_csv": str(p),
                "timestamps_csv": str((p.parent / "timestamps.csv").resolve()),
            }
        if p.is_dir():
            gt_csv = p / "gt_pose.csv"
            ts_csv = p / "timestamps.csv"
            if gt_csv.exists() and ts_csv.exists():
                return "viode", {"gt_pose_csv": str(gt_csv.resolve()), "timestamps_csv": str(ts_csv.resolve())}

    gt_csv = seq_dir / "gt_pose.csv"
    ts_csv = seq_dir / "timestamps.csv"
    if gt_csv.exists() and ts_csv.exists() and gt_type_arg in ("auto", "viode"):
        return "viode", {"gt_pose_csv": str(gt_csv.resolve()), "timestamps_csv": str(ts_csv.resolve())}

    dataset_root = seq_root.parent if seq_root.name == "sequences" else seq_root
    pose_txt = dataset_root / "poses" / f"{seq_name}.txt"
    if pose_txt.exists() and gt_type_arg in ("auto", "kitti"):
        return "kitti", {"pose_txt": str(pose_txt.resolve())}

    return None, None


def inv_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3:4]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3:4] = -R.T @ t
    return Ti


def rot_angle_deg(R: np.ndarray) -> float:
    """
    输入旋转矩阵 R（应接近 SO(3)），输出旋转角（度）。
    """
    tr = float(np.trace(R))
    x = (tr - 1.0) * 0.5
    x = max(-1.0, min(1.0, x))
    return float(np.degrees(np.arccos(x)))


def angle_between_deg(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return None
    c = float(np.dot(a, b) / (na * nb))
    c = max(-1.0, min(1.0, c))
    return float(np.degrees(np.arccos(c)))


# -----------------------------
# SFM metrics computation
# -----------------------------
def project_uv(K: np.ndarray, R: np.ndarray, t: np.ndarray, Xw: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
    """
    world->cam: Xc = R*Xw + t
    returns (uv, depth_z) ; uv=None if depth<=0
    """
    Xc = R @ Xw.reshape(3) + t.reshape(3)
    z = float(Xc[2])
    if z <= 1e-9:
        return None, z
    x = Xc[:2] / z
    uv = (K[:2, :2] @ x.reshape(2, 1)).reshape(2) + K[:2, 2]
    return uv.astype(np.float64), z


def compute_reproj_stats(
    K: np.ndarray,
    poses: Dict[int, Tuple[np.ndarray, np.ndarray]],
    map_points: List[Dict],
    tracks: List[Dict],
    tri_min_depth: float = 0.0,
) -> Dict:
    """
    用 tracks 里的真实观测，统计在已估计 pose 的帧上，3D 点的重投影误差分布。
    同时统计 bad_depth_ratio（深度<=0 或 <=tri_min_depth 的比例）。
    """
    track_map = {int(tr["id"]): tr for tr in tracks}

    errs = []
    bad_depth = 0
    total_obs = 0

    # 可用的 pose 帧集合
    pose_fids = set(int(k) for k in poses.keys())

    for mp in map_points:
        tid = int(mp.get("track_id", -1))
        if tid not in track_map:
            continue
        X = np.asarray(mp["xyz"], dtype=np.float64).reshape(3)
        tr = track_map[tid]

        # 收集该 track 在“有 pose 的帧”里的观测
        for fi, uv in zip(tr["frames"], tr["uvs"]):
            fi = int(fi)
            if fi not in pose_fids:
                continue
            total_obs += 1
            R, t = poses[fi]
            uv_hat, z = project_uv(K, R, t, X)
            if (uv_hat is None) or (z <= tri_min_depth + 1e-9):
                bad_depth += 1
                continue
            uv = np.asarray(uv, dtype=np.float64).reshape(2)
            errs.append(float(np.linalg.norm(uv_hat - uv)))

    errs = np.asarray(errs, dtype=np.float64)
    out = {
        "num_observations": int(total_obs),
        "num_valid_observations": int(errs.size),
        "bad_depth_ratio": float(bad_depth / max(1, total_obs)),
    }
    if errs.size == 0:
        out.update(
            {
                "reproj_rms_px": None,
                "reproj_med_px": None,
                "reproj_p90_px": None,
            }
        )
        return out

    out.update(
        {
            "reproj_rms_px": float(np.sqrt(np.mean(errs ** 2))),
            "reproj_med_px": float(np.median(errs)),
            "reproj_p90_px": float(np.percentile(errs, 90)),
        }
    )
    return out


def compute_cheirality_ratio(
    poses: Dict[int, Tuple[np.ndarray, np.ndarray]],
    map_points: List[Dict],
    tracks: List[Dict],
    tri_min_depth: float = 0.0,
) -> float:
    """
    cheirality_ratio：点在其“可观测的所有 pose 帧”里深度都为正（>tri_min_depth）的比例。
    这是退化/解错的强信号：前方率低通常说明两视图/三角化不可靠。
    """
    track_map = {int(tr["id"]): tr for tr in tracks}
    pose_fids = set(int(k) for k in poses.keys())

    ok = 0
    tot = 0
    for mp in map_points:
        tid = int(mp.get("track_id", -1))
        if tid not in track_map:
            continue
        X = np.asarray(mp["xyz"], dtype=np.float64).reshape(3)
        tr = track_map[tid]

        # 找到该 track 在 pose_fids 中的观测帧
        obs_frames = [int(fi) for fi in tr["frames"] if int(fi) in pose_fids]
        if len(obs_frames) < 2:
            continue  # 至少两帧可见才有意义
        tot += 1

        good = True
        for fi in obs_frames:
            R, t = poses[fi]
            Xc = R @ X + t.reshape(3)
            if float(Xc[2]) <= tri_min_depth + 1e-9:
                good = False
                break
        if good:
            ok += 1

    return float(ok / max(1, tot))


def compute_triangulation_angle_stats(
    poses: Dict[int, Tuple[np.ndarray, np.ndarray]],
    map_points: List[Dict],
    tracks: List[Dict],
) -> Dict:
    """
    tri_angle: 对每个点，从其可见的 pose 帧中选两帧，计算观测射线夹角（度）。
    这里使用“最大夹角”作为该点的 triangulation strength（更稳健），最后统计分位数。
    """
    track_map = {int(tr["id"]): tr for tr in tracks}
    pose_fids = set(int(k) for k in poses.keys())

    # 预计算 camera centers（世界坐标系=局部 start 坐标系）
    # pose: world->cam, camera center in world: C = -R^T t
    cam_centers = {}
    for fi, (R, t) in poses.items():
        fi = int(fi)
        C = -(R.T @ t.reshape(3, 1)).reshape(3)
        cam_centers[fi] = C

    angles = []

    for mp in map_points:
        tid = int(mp.get("track_id", -1))
        if tid not in track_map:
            continue
        X = np.asarray(mp["xyz"], dtype=np.float64).reshape(3)
        tr = track_map[tid]
        obs_frames = [int(fi) for fi in tr["frames"] if int(fi) in pose_fids]
        if len(obs_frames) < 2:
            continue

        # 计算所有帧对的夹角，取最大值
        best = None
        for i in range(len(obs_frames)):
            fi = obs_frames[i]
            vi = X - cam_centers[fi]
            for j in range(i + 1, len(obs_frames)):
                fj = obs_frames[j]
                vj = X - cam_centers[fj]
                ang = angle_between_deg(vi, vj)
                if ang is None:
                    continue
                if (best is None) or (ang > best):
                    best = ang

        if best is not None:
            angles.append(best)

    angles = np.asarray(angles, dtype=np.float64)
    if angles.size == 0:
        return {"tri_angle_med_deg": None, "tri_angle_p10_deg": None, "tri_angle_p90_deg": None}

    return {
        "tri_angle_med_deg": float(np.median(angles)),
        "tri_angle_p10_deg": float(np.percentile(angles, 10)),
        "tri_angle_p90_deg": float(np.percentile(angles, 90)),
    }


def compute_init_parallax_from_map_points(map_points: List[Dict]) -> Optional[float]:
    """
    用“初始两视图三角化点”里自带的 base_uv / pivot_uv，估计像素视差中位数。
    （增量三角化的新点不一定有 pivot_uv，所以这里只看带 pivot_uv 的那部分）
    """
    ds = []
    for mp in map_points:
        if "pivot_uv" not in mp:
            continue
        u0 = np.asarray(mp["base_uv"], dtype=np.float64).reshape(2)
        uk = np.asarray(mp["pivot_uv"], dtype=np.float64).reshape(2)
        ds.append(float(np.linalg.norm(uk - u0)))
    if not ds:
        return None
    return float(np.median(np.asarray(ds, dtype=np.float64)))


def parse_candidate_json(candidate_json: str) -> Dict:
    """
    从 init_candidates_metrics.json 里找出 score 最大的 ok 候选，作为 chosen pivot 的解释性指标。
    """
    if (not os.path.exists(candidate_json)):
        return {}

    with open(candidate_json, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    oks = [m for m in metrics if isinstance(m, dict) and m.get("status") == "ok"]
    if not oks:
        return {}

    best = max(oks, key=lambda m: float(m.get("score", -1e18)))
    # 统一字段名（summary.csv 里更好读）
    out = {
        "pivot_idx": int(best.get("pivot", -1)),
        "model": str(best.get("model", "")),
        "parallax_px_candidate": float(best.get("parallax", np.nan)),
        "tri_points_candidate": int(best.get("tri_points", -1)),
        "tri_candidate_tracks": int(best.get("tri_candidate_tracks", 0)),
        "pnp_eval_success": int(best.get("pnp_success", 0)),
        "pnp_eval_total": int(best.get("pnp_total", 0)),
        "pnp_eval_median_inliers": float(best.get("pnp_median_inliers", 0.0)),
        "E_inliers_candidate": int(best.get("nE", -1)),
        "errE_candidate": float(best.get("errE", np.nan)),
        "candidate_score": float(best.get("score", np.nan)),
        "candidate_allow_H": bool(best.get("allow_H", True)),
    }
    # success_rate（候选评估阶段）
    out["pnp_eval_success_rate"] = float(out["pnp_eval_success"] / max(1, out["pnp_eval_total"]))
    out["triangulation_ratio"] = float(
        best.get("triangulation_ratio", out["tri_points_candidate"] / max(1, out["tri_candidate_tracks"]))
    )
    return out


def poses_to_npz(poses: Dict[int, Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将 poses dict (fid -> (R,t)) 转成可保存的 arrays。
    """
    fids = np.array(sorted(poses.keys()), dtype=np.int32)
    Rs = []
    ts = []
    for fid in fids.tolist():
        R, t = poses[int(fid)]
        Rs.append(R.astype(np.float32))
        ts.append(t.reshape(3).astype(np.float32))
    Rs = np.stack(Rs, axis=0) if Rs else np.zeros((0, 3, 3), np.float32)
    ts = np.stack(ts, axis=0) if ts else np.zeros((0, 3), np.float32)
    return fids, Rs, ts


def infer_global_ids_from_frames(frames: List[Dict], fallback_start: int) -> Dict[int, int]:
    """
    将 local frame index -> global frame index
    优先用 frames[i]["name"]（如 '000123'），否则 fallback: start + i
    """
    m = {}
    for i, fr in enumerate(frames):
        name = str(fr.get("name", "")).strip()
        if name.isdigit():
            m[i] = int(name)
        else:
            m[i] = int(fallback_start + i)
    return m


def evaluate_against_gt(
    gt_T0i: Dict[int, np.ndarray],
    start_global: int,
    local_to_global: Dict[int, int],
    poses_est: Dict[int, Tuple[np.ndarray, np.ndarray]],
) -> Dict:
    """
    对照真值（尺度无关）：
    - 对每个已估计的 local frame i>0，比较 T_est(i->start) 与 T_gt(i->start)
    - 输出旋转误差（deg）分布、平移方向误差（deg）分布
    - 输出 scale_fit（诊断：把 t_est 拟合到 t_gt 的最优尺度）
    """
    if start_global not in gt_T0i:
        return {"gt_valid": False}

    T0_start = gt_T0i[start_global]  # start -> 0
    Tstart_0 = inv_T(T0_start)       # 0 -> start

    rot_errs = []
    dir_errs = []
    t_est_list = []
    t_gt_list = []

    for local_i, (R_w2c, t_w2c) in poses_est.items():
        local_i = int(local_i)
        if local_i == 0:
            continue
        g = int(local_to_global.get(local_i, start_global + local_i))
        if g not in gt_T0i:
            continue

        # ---- GT: g -> start ----
        T0_g = gt_T0i[g]                 # g -> 0
        Tgt = Tstart_0 @ T0_g            # g -> start

        # ---- EST: local_i camera -> start ----
        # pose stored as world(start)->cam : Xc = R*Xw + t
        # inverse gives cam -> world(start)
        R = R_w2c.astype(np.float64)
        t = t_w2c.reshape(3, 1).astype(np.float64)
        Test = np.eye(4, dtype=np.float64)
        Test[:3, :3] = R.T
        Test[:3, 3:4] = -R.T @ t

        # rotation error
        R_err = Test[:3, :3] @ Tgt[:3, :3].T
        rot_errs.append(rot_angle_deg(R_err))

        # translation direction error (scale-free)
        te = Test[:3, 3]
        tg = Tgt[:3, 3]
        ang = angle_between_deg(te, tg)
        if ang is not None:
            dir_errs.append(ang)
        if np.linalg.norm(te) > 1e-9 and np.linalg.norm(tg) > 1e-9:
            t_est_list.append(te)
            t_gt_list.append(tg)

    rot_errs = np.asarray(rot_errs, dtype=np.float64)
    dir_errs = np.asarray(dir_errs, dtype=np.float64)

    out = {"gt_valid": True}
    if rot_errs.size:
        out["gt_rot_med_deg"] = float(np.median(rot_errs))
        out["gt_rot_p90_deg"] = float(np.percentile(rot_errs, 90))
    else:
        out["gt_rot_med_deg"] = None
        out["gt_rot_p90_deg"] = None

    if dir_errs.size:
        out["gt_trans_dir_med_deg"] = float(np.median(dir_errs))
        out["gt_trans_dir_p90_deg"] = float(np.percentile(dir_errs, 90))
    else:
        out["gt_trans_dir_med_deg"] = None
        out["gt_trans_dir_p90_deg"] = None

    # scale_fit: argmin_s || s*t_est - t_gt ||^2  =>  s = (t_gt·t_est)/(t_est·t_est)
    if len(t_est_list) >= 1:
        Te = np.stack(t_est_list, axis=0).reshape(-1, 3)
        Tg = np.stack(t_gt_list, axis=0).reshape(-1, 3)
        num = float(np.sum(Tg * Te))
        den = float(np.sum(Te * Te)) + 1e-12
        out["gt_scale_fit"] = float(num / den)
    else:
        out["gt_scale_fit"] = None

    out["gt_frames_used"] = int(rot_errs.size)
    return out


# -----------------------------
# one-run wrapper (noBA / withBA)
# -----------------------------
def run_one(
    cfg_base: Dict,
    paths_base: Dict,
    out_dir: str,
    start_frame: int,
    front_metrics: Dict,
    q_threshold_source: str,
    mode_tag: str,
    enable_two_view_ba: bool,
    enable_global_ba: bool,
    verbose: bool,
) -> Dict:
    """
    单次运行（一个 start + 一个模式）并计算指标。
    """
    os.makedirs(out_dir, exist_ok=True)

    cfg = copy.deepcopy(cfg_base)
    cfg["sfm"] = dict(cfg.get("sfm", {}))
    cfg["sfm"]["start_frame"] = int(start_frame)
    cfg["sfm"]["enable_two_view_ba"] = bool(enable_two_view_ba)
    cfg["sfm"]["enable_global_ba"] = bool(enable_global_ba)

    paths = dict(paths_base)
    paths["sfm_out_dir"] = out_dir

    K, _ = get_camera_from_cfg(cfg)
    W = int(cfg["sfm"].get("init_window_frames", 10))
    tri_min_depth = float(cfg["sfm"].get("tri_min_depth", 0.0))

    t0 = time.perf_counter()
    try:
        poses, pts3d, map_points, frames, tracks = run_static_sfm_v2(cfg, paths, verbose=verbose)
        ok = True
        err_msg = ""
    except Exception as e:
        poses, pts3d, map_points, frames, tracks = {}, np.zeros((0, 3)), [], [], []
        ok = False
        err_msg = str(e)[:300]
    t1 = time.perf_counter()

    # 保存 npz（便于之后复现实验）
    if ok:
        fids, Rs, ts = poses_to_npz(poses)
        np.savez(
            os.path.join(out_dir, f"sfm_{mode_tag}.npz"),
            start_frame=int(start_frame),
            window_W=int(W),
            frame_ids=fids,
            Rs=Rs,
            ts=ts,
            pts3d=np.asarray(pts3d, dtype=np.float32),
        )

    # 解析候选 json（解释 pivot / model / parallax）
    cand_json = os.path.join(out_dir, "init_candidates_metrics.json")
    cand = parse_candidate_json(cand_json) if ok else {}
    cand_summary = summarize_candidate_pool(load_candidate_metrics_json(cand_json))

    # 计算几何一致性指标
    geom = {}
    triang = {}
    chei = None
    par_init = None
    pose_cov = None

    if ok:
        geom = compute_reproj_stats(K, poses, map_points, tracks, tri_min_depth=tri_min_depth)
        tri_ang = compute_triangulation_angle_stats(poses, map_points, tracks)
        triang = tri_ang
        chei = compute_cheirality_ratio(poses, map_points, tracks, tri_min_depth=tri_min_depth)
        par_init = compute_init_parallax_from_map_points(map_points)

        # pose coverage：窗口 0..W 里有 pose 的帧比例
        pose_cov = float(len([i for i in range(0, W + 1) if i in poses]) / float(W + 1))

    # 记录 per-run metrics（便于调参时快速定位）
    per_run = {
        "ok": bool(ok),
        "mode": mode_tag,
        "start_frame": int(start_frame),
        "window_W": int(W),
        "runtime_s": float(t1 - t0),
        "error": err_msg,
        "num_poses": int(len(poses)),
        "num_points": int(len(map_points)),
        "pose_coverage": pose_cov,
        "cheirality_ratio": chei,
        "parallax_init_med_px": par_init,
    }
    per_run.update({f"{k}": v for k, v in geom.items()})
    per_run.update({f"{k}": v for k, v in triang.items()})
    per_run.update({f"cand_{k}": v for k, v in cand.items()})
    per_run.update(front_metrics)

    cand_summary_missing = bool(cand and not cand_summary)
    if cand_summary_missing:
        cand_summary = {
            "cand_pivot": cand.get("pivot_idx", -1),
            "cand_model": cand.get("model", ""),
            "cand_parallax_px": cand.get("parallax_px_candidate", 0.0),
            "cand_tri_points": cand.get("tri_points_candidate", 0),
            "cand_tri_candidate_tracks": cand.get("tri_candidate_tracks", 0),
            "cand_triangulation_ratio": cand.get("triangulation_ratio", 0.0),
            "qcand_pivot": cand.get("pivot_idx", -1),
            "qcand_model": cand.get("model", ""),
            "qcand_allow_H": cand.get("candidate_allow_H", None),
            "qcand_parallax_px": cand.get("parallax_px_candidate", 0.0),
            "qcand_tri_points": cand.get("tri_points_candidate", 0),
            "qcand_tri_candidate_tracks": cand.get("tri_candidate_tracks", 0),
            "qcand_triangulation_ratio": cand.get("triangulation_ratio", 0.0),
            "qcand_pnp_success_rate": cand.get("pnp_eval_success_rate", 0.0),
            "qcand_pnp_median_inliers": cand.get("pnp_eval_median_inliers", 0.0),
            "cand_pnp_success_rate": cand.get("pnp_eval_success_rate", 0.0),
            "cand_pnp_median_inliers": cand.get("pnp_eval_median_inliers", 0.0),
            "cand_viable_ratio": 0.0,
            "cand_model_purity": 0.0,
            "cand_geom_rank_mean": 0.0,
            "cand_pareto_consistent": 0.0,
            "cand_summary_missing": 1,
        }
    else:
        cand_summary = dict(cand_summary or {})
        cand_summary.setdefault("cand_summary_missing", 0)
    tri_ratio = float(
        cand_summary.get("qcand_triangulation_ratio", cand_summary.get("cand_triangulation_ratio", cand.get("triangulation_ratio", 0.0)))
    )
    per_run["tri_candidate_tracks"] = int(
        cand_summary.get("cand_tri_candidate_tracks", cand.get("tri_candidate_tracks", 0))
    )
    per_run["triangulation_ratio"] = tri_ratio
    geom_for_q = {
        "reproj_med_px": geom.get("reproj_med_px"),
        "reproj_p90_px": geom.get("reproj_p90_px"),
        "cheirality_ratio": chei,
        "triangulation_ratio": tri_ratio,
    }
    q_pre_metrics = build_q_pre_metrics(cfg, front_metrics, cand_summary)
    q_post_metrics = build_q_post_metrics(cfg, q_pre_metrics, geom_for_q, bool(ok))
    gate_metrics = decide_gate(cfg, q_pre_metrics, q_post_metrics, bool(ok))
    geom_quality = evaluate_post_geom_quality(cfg, geom_for_q)
    solver_ok = bool(ok)
    geom_ok = bool(solver_ok and geom_quality.get("post_geom_strict_ok", False))
    success_strict = bool(solver_ok and geom_ok)
    per_run.update(q_pre_metrics)
    per_run.update(q_post_metrics)
    per_run.update(gate_metrics)
    per_run.update(geom_quality)
    per_run["cand_summary_missing"] = int(cand_summary_missing)
    per_run["solver_ok"] = solver_ok
    per_run["geom_ok"] = geom_ok
    per_run["success_strict"] = success_strict
    per_run["Q"] = float(q_post_metrics.get("Q_post", 0.0))
    per_run["gate_decision"] = gate_metrics.get("gate_post")
    per_run["q_accept_threshold"] = q_post_metrics.get("q_post_accept_threshold")
    per_run["q_delay_threshold"] = q_post_metrics.get("q_post_delay_threshold")
    per_run["q_threshold_source"] = q_threshold_source

    with open(os.path.join(out_dir, "per_run_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(per_run, f, ensure_ascii=False, indent=2)

    return per_run, frames, poses, map_points, tracks


# -----------------------------
# main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cfg",
        type=str,
        default=resolve_default_cfg(),
    )
    ap.add_argument("--seq", type=str, default=None, help="可选：覆盖 cfg.dataset.name")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=100, help="global frame end（包含）")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--gt_path", type=str, default=None, help="可选：显式指定 GT 文件或目录。支持 KITTI poses txt，或包含 VIODE gt_pose.csv/timestamps.csv 的目录。")
    ap.add_argument("--gt_type", type=str, default="auto", choices=["auto", "kitti", "viode"], help="GT 类型。默认自动识别。")
    ap.add_argument("--poses_path", type=str, default=None, help="兼容旧参数，等价于 --gt_path。")
    ap.add_argument("--out_tag", type=str, default="", help="Optional subdir tag for variant-specific outputs.")
    ap.add_argument("--step7_tag", type=str, default="", help="Optional tag under step7_root to read front-end metrics.")
    ap.add_argument("--q_pre_accept_threshold", type=float, default=-1.0)
    ap.add_argument("--q_pre_delay_threshold", type=float, default=-1.0)
    ap.add_argument("--q_post_accept_threshold", type=float, default=-1.0)
    ap.add_argument("--q_post_delay_threshold", type=float, default=-1.0)
    ap.add_argument("--q_accept_threshold", type=float, default=-1.0)
    ap.add_argument("--q_delay_threshold", type=float, default=-1.0)
    ap.add_argument("--dataset_split", type=str, default="", choices=["", "train", "val", "test", "all", "unspecified"])
    ap.add_argument("--q_threshold_mode", type=str, default="", choices=["", "frozen", "tuning"])
    ap.add_argument("--threshold_set_id", type=str, default="")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    if args.seq is not None:
        cfg["dataset"] = dict(cfg.get("dataset", {}))
        cfg["dataset"]["name"] = str(args.seq)
    if args.dataset_split or args.q_threshold_mode or args.threshold_set_id:
        cfg.setdefault("experiment_protocol", {})
    if args.dataset_split:
        cfg["experiment_protocol"]["dataset_split"] = str(args.dataset_split)
    if args.q_threshold_mode:
        cfg["experiment_protocol"]["q_threshold_mode"] = str(args.q_threshold_mode)
    if args.threshold_set_id:
        cfg["experiment_protocol"]["threshold_set_id"] = str(args.threshold_set_id)
    q_threshold_overridden = (
        args.q_pre_accept_threshold > 0
        or args.q_pre_delay_threshold > 0
        or args.q_post_accept_threshold > 0
        or args.q_post_delay_threshold > 0
        or args.q_accept_threshold > 0
        or args.q_delay_threshold > 0
    )
    validate_experiment_protocol(cfg, q_threshold_overridden)
    if q_threshold_overridden:
        cfg.setdefault("init_quality", {})
    if args.q_pre_accept_threshold > 0:
        cfg["init_quality"]["q_pre_accept_threshold"] = float(args.q_pre_accept_threshold)
    if args.q_pre_delay_threshold > 0:
        cfg["init_quality"]["q_pre_delay_threshold"] = float(args.q_pre_delay_threshold)
    if args.q_post_accept_threshold > 0:
        cfg["init_quality"]["q_post_accept_threshold"] = float(args.q_post_accept_threshold)
    if args.q_post_delay_threshold > 0:
        cfg["init_quality"]["q_post_delay_threshold"] = float(args.q_post_delay_threshold)
    if args.q_accept_threshold > 0:
        cfg["init_quality"]["q_post_accept_threshold"] = float(args.q_accept_threshold)
        cfg["init_quality"]["accept_threshold"] = float(args.q_accept_threshold)
    if args.q_delay_threshold > 0:
        cfg["init_quality"]["q_post_delay_threshold"] = float(args.q_delay_threshold)
        cfg["init_quality"]["delay_threshold"] = float(args.q_delay_threshold)
    paths = get_common_paths(cfg)
    seq_name = paths.get("seq_name", cfg.get("dataset", {}).get("name", "00"))
    step7_root = paths["step7_root"]
    if args.step7_tag:
        step7_root = os.path.join(step7_root, args.step7_tag)
        paths["step7_root"] = step7_root
        paths["kps_npz_dir_step7"] = os.path.join(step7_root, "kps_npz")
    front_frame_rows = load_step7_frame_metrics(step7_root)

    # 输出目录
    out_root = os.path.join(paths["eval_root"], "sfm_init_quality_v2", str(seq_name))
    if args.out_tag:
        out_root = os.path.join(out_root, args.out_tag)
    os.makedirs(out_root, exist_ok=True)
    protocol_record = build_experiment_protocol_record(
        cfg,
        q_threshold_overridden=q_threshold_overridden,
        script_name=Path(__file__).name,
        cfg_path=str(Path(args.cfg).expanduser().resolve()),
        out_root=out_root,
    )
    q_threshold_source = protocol_record["q_threshold_source"]
    protocol_path = os.path.join(out_root, "experiment_protocol.json")
    with open(protocol_path, "w", encoding="utf-8") as f:
        json.dump(protocol_record, f, ensure_ascii=False, indent=2)

    # GT 自动识别 / 加载
    gt_path_arg = args.gt_path if args.gt_path is not None else args.poses_path
    gt_kind, gt_spec = infer_gt_spec(cfg, gt_path_arg, args.gt_type)
    gt_T0i = None
    gt_meta = {}
    if gt_kind is None or gt_spec is None:
        print("[WARN] GT source not found.")
        print("       你仍然可以跑几何指标，但真值对照列会为空。")
    elif gt_kind == "kitti":
        pose_txt = gt_spec["pose_txt"]
        if not os.path.exists(pose_txt):
            print(f"[WARN] KITTI GT poses file not found: {pose_txt}")
            print("       你仍然可以跑几何指标，但真值对照列会为空。")
        else:
            gt_T0i = load_kitti_poses_T0i(pose_txt)
            gt_meta = {"gt_kind": "kitti", "gt_path": pose_txt, "gt_num_frames": len(gt_T0i)}
            print(f"[INFO] GT loaded as KITTI poses: {pose_txt}, N={len(gt_T0i)}")
    elif gt_kind == "viode":
        gt_pose_csv = gt_spec["gt_pose_csv"]
        timestamps_csv = gt_spec["timestamps_csv"]
        if (not os.path.exists(gt_pose_csv)) or (not os.path.exists(timestamps_csv)):
            print(f"[WARN] VIODE GT files missing: gt_pose_csv={gt_pose_csv}, timestamps_csv={timestamps_csv}")
            print("       你仍然可以跑几何指标，但真值对照列会为空。")
        else:
            gt_T0i, gt_meta = load_viode_gt_T0i(gt_pose_csv, timestamps_csv)
            gt_meta.update(
                {
                    "gt_kind": "viode",
                    "gt_pose_csv": gt_pose_csv,
                    "timestamps_csv": timestamps_csv,
                    "gt_num_frames": len(gt_T0i),
                }
            )
            print(
                "[INFO] GT loaded as VIODE poses: "
                f"{gt_pose_csv}, aligned_frames={len(gt_T0i)}, "
                f"match_dt_med={gt_meta.get('match_dt_med_sec', 0.0):.6f}s"
            )

    # 运行范围
    starts = list(range(int(args.start), int(args.end) + 1, int(args.stride)))

    rows = []
    for s in starts:
        start_dir = os.path.join(out_root, f"start_{s:06d}")
        os.makedirs(start_dir, exist_ok=True)
        window_W = int(cfg.get("sfm", {}).get("init_window_frames", 10)) + 1
        front_metrics = aggregate_front_window_metrics(front_frame_rows, s, window_W, cfg=cfg)

        # noBA
        no_dir = os.path.join(start_dir, "noBA")
        r_no, frames_no, poses_no, mps_no, tracks_no = run_one(
            cfg_base=cfg,
            paths_base=paths,
            out_dir=no_dir,
            start_frame=s,
            front_metrics=front_metrics,
            q_threshold_source=q_threshold_source,
            mode_tag="noBA",
            enable_two_view_ba=False,
            enable_global_ba=False,
            verbose=args.verbose,
        )

        # withBA
        ba_dir = os.path.join(start_dir, "withBA")
        r_ba, frames_ba, poses_ba, mps_ba, tracks_ba = run_one(
            cfg_base=cfg,
            paths_base=paths,
            out_dir=ba_dir,
            start_frame=s,
            front_metrics=front_metrics,
            q_threshold_source=q_threshold_source,
            mode_tag="withBA",
            enable_two_view_ba=bool(cfg.get("sfm", {}).get("enable_two_view_ba", True)),
            enable_global_ba=bool(cfg.get("sfm", {}).get("enable_global_ba", True)),
            verbose=args.verbose,
        )

        # 统一 local->global 映射（优先用 frames[i].name）
        # 以 withBA 的 frames 为准（正常两次一致）
        local_to_global = infer_global_ids_from_frames(frames_ba if frames_ba else frames_no, fallback_start=s)
        start_global = int(local_to_global.get(0, s))

        # 真值对照（默认对 withBA 结果做）
        gt_cmp = {}
        if gt_T0i is not None and r_ba.get("ok", False):
            gt_cmp = evaluate_against_gt(
                gt_T0i=gt_T0i,
                start_global=start_global,
                local_to_global=local_to_global,
                poses_est=poses_ba,
            )

        # 汇总成一行（summary.csv）
        row = {
            "start_frame_cfg": int(s),
            "start_frame_global": int(start_global),
            "window_W": int(cfg.get("sfm", {}).get("init_window_frames", 10)),
            # chosen candidate（以 withBA 解析到的 cand_* 为准）
            "pivot_idx": r_ba.get("cand_pivot_idx"),
            "model": r_ba.get("cand_model"),
            "parallax_px_candidate": r_ba.get("cand_parallax_px_candidate"),
            "tri_points_candidate": r_ba.get("cand_tri_points_candidate"),
            "tri_candidate_tracks": r_ba.get("cand_tri_candidate_tracks"),
            "triangulation_ratio": r_ba.get("triangulation_ratio"),
            "pnp_eval_success_rate": r_ba.get("cand_pnp_eval_success_rate"),
            "pnp_eval_median_inliers": r_ba.get("cand_pnp_eval_median_inliers"),
            "E_inliers_candidate": r_ba.get("cand_E_inliers_candidate"),
            "candidate_score": r_ba.get("cand_candidate_score"),
            "front_p_static": front_metrics.get("front_p_static"),
            "front_p_band": front_metrics.get("front_p_band"),
            "front_coverage_ratio": front_metrics.get("front_coverage_ratio"),
            "front_grid_entropy": front_metrics.get("front_grid_entropy"),
            "front_kept_dyn_ratio": front_metrics.get("front_kept_dyn_ratio"),
            # noBA metrics
            "no_ok": r_no.get("ok"),
            "no_solver_ok": r_no.get("solver_ok", r_no.get("ok")),
            "no_geom_ok": r_no.get("geom_ok"),
            "no_success_strict": r_no.get("success_strict"),
            "no_runtime_s": r_no.get("runtime_s"),
            "no_pose_coverage": r_no.get("pose_coverage"),
            "no_num_poses": r_no.get("num_poses"),
            "no_num_points": r_no.get("num_points"),
            "no_num_obs": r_no.get("num_observations"),
            "no_bad_depth_ratio": r_no.get("bad_depth_ratio"),
            "no_cheirality_ratio": r_no.get("cheirality_ratio"),
            "no_tri_angle_med_deg": r_no.get("tri_angle_med_deg"),
            "no_tri_angle_p90_deg": r_no.get("tri_angle_p90_deg"),
            "no_parallax_init_med_px": r_no.get("parallax_init_med_px"),
            "no_reproj_rms_px": r_no.get("reproj_rms_px"),
            "no_reproj_med_px": r_no.get("reproj_med_px"),
            "no_reproj_p90_px": r_no.get("reproj_p90_px"),
            "no_Q_pre": r_no.get("Q_pre"),
            "no_Q_post": r_no.get("Q_post", r_no.get("Q")),
            "no_Q_post_geom_only": r_no.get("Q_post_geom_only"),
            "no_Q": r_no.get("Q_post", r_no.get("Q")),
            "no_gate_pre": r_no.get("gate_pre"),
            "no_gate_pre_reason": r_no.get("gate_pre_reason"),
            "no_gate_post": r_no.get("gate_post", r_no.get("gate_decision")),
            "no_gate_post_reason": r_no.get("gate_post_reason", r_no.get("gate_reason")),
            "no_gate_decision": r_no.get("gate_decision"),
            "no_post_geom_failure_reason": r_no.get("post_geom_failure_reason"),
            "no_cand_summary_missing": r_no.get("cand_summary_missing"),
            "no_accepted_but_bad_geom": int(r_no.get("gate_post") == "accept" and not bool(r_no.get("geom_ok"))),
            "no_error": r_no.get("error"),
            # withBA metrics
            "ba_ok": r_ba.get("ok"),
            "ba_solver_ok": r_ba.get("solver_ok", r_ba.get("ok")),
            "ba_geom_ok": r_ba.get("geom_ok"),
            "ba_success_strict": r_ba.get("success_strict"),
            "ba_runtime_s": r_ba.get("runtime_s"),
            "ba_pose_coverage": r_ba.get("pose_coverage"),
            "ba_num_poses": r_ba.get("num_poses"),
            "ba_num_points": r_ba.get("num_points"),
            "ba_num_obs": r_ba.get("num_observations"),
            "ba_bad_depth_ratio": r_ba.get("bad_depth_ratio"),
            "ba_cheirality_ratio": r_ba.get("cheirality_ratio"),
            "ba_tri_angle_med_deg": r_ba.get("tri_angle_med_deg"),
            "ba_tri_angle_p90_deg": r_ba.get("tri_angle_p90_deg"),
            "ba_parallax_init_med_px": r_ba.get("parallax_init_med_px"),
            "ba_reproj_rms_px": r_ba.get("reproj_rms_px"),
            "ba_reproj_med_px": r_ba.get("reproj_med_px"),
            "ba_reproj_p90_px": r_ba.get("reproj_p90_px"),
            "ba_Q_pre": r_ba.get("Q_pre"),
            "ba_Q_post": r_ba.get("Q_post", r_ba.get("Q")),
            "ba_Q_post_geom_only": r_ba.get("Q_post_geom_only"),
            "ba_Q": r_ba.get("Q_post", r_ba.get("Q")),
            "ba_gate_pre": r_ba.get("gate_pre"),
            "ba_gate_pre_reason": r_ba.get("gate_pre_reason"),
            "ba_gate_post": r_ba.get("gate_post", r_ba.get("gate_decision")),
            "ba_gate_post_reason": r_ba.get("gate_post_reason", r_ba.get("gate_reason")),
            "ba_gate_decision": r_ba.get("gate_decision"),
            "ba_post_geom_failure_reason": r_ba.get("post_geom_failure_reason"),
            "ba_cand_summary_missing": r_ba.get("cand_summary_missing"),
            "ba_accepted_but_bad_geom": int(r_ba.get("gate_post") == "accept" and not bool(r_ba.get("geom_ok"))),
            "q_threshold_source": q_threshold_source,
            "ba_error": r_ba.get("error"),
        }

        # BA 改善幅度（对照组 vs 实验组）
        try:
            if row["no_reproj_med_px"] is not None and row["ba_reproj_med_px"] is not None:
                row["reproj_med_reduction_ratio"] = float(
                    (row["no_reproj_med_px"] - row["ba_reproj_med_px"]) / max(1e-9, row["no_reproj_med_px"])
                )
            else:
                row["reproj_med_reduction_ratio"] = None
        except Exception:
            row["reproj_med_reduction_ratio"] = None

        # 真值对照（尺度无关）
        row.update({
            "gt_valid": gt_cmp.get("gt_valid", False),
            "gt_frames_used": gt_cmp.get("gt_frames_used"),
            "gt_rot_med_deg": gt_cmp.get("gt_rot_med_deg"),
            "gt_rot_p90_deg": gt_cmp.get("gt_rot_p90_deg"),
            "gt_trans_dir_med_deg": gt_cmp.get("gt_trans_dir_med_deg"),
            "gt_trans_dir_p90_deg": gt_cmp.get("gt_trans_dir_p90_deg"),
            "gt_scale_fit": gt_cmp.get("gt_scale_fit"),
        })

        rows.append(row)
        print(f"[E1] start={s:6d}  no_ok={row['no_ok']}  ba_ok={row['ba_ok']}  model={row['model']}  pivot={row['pivot_idx']}")

    # 写 summary.csv
    csv_path = os.path.join(out_root, "summary.csv")
    fieldnames = []
    # 固定顺序：先把常用字段排前面，再补其余
    preferred = [
        "start_frame_cfg","start_frame_global","window_W",
        "pivot_idx","model","parallax_px_candidate","tri_points_candidate","tri_candidate_tracks","triangulation_ratio",
        "pnp_eval_success_rate","pnp_eval_median_inliers","E_inliers_candidate","candidate_score",
        "front_p_static","front_p_band","front_coverage_ratio","front_grid_entropy","front_kept_dyn_ratio",
        "no_ok","no_solver_ok","no_geom_ok","no_success_strict",
        "ba_ok","ba_solver_ok","ba_geom_ok","ba_success_strict","reproj_med_reduction_ratio",
        "no_Q_pre","no_Q_post","no_Q_post_geom_only","ba_Q_pre","ba_Q_post","ba_Q_post_geom_only","no_Q","ba_Q",
        "no_gate_pre","no_gate_pre_reason","no_gate_post","no_gate_post_reason",
        "ba_gate_pre","ba_gate_pre_reason","ba_gate_post","ba_gate_post_reason",
        "no_gate_decision","ba_gate_decision",
        "no_post_geom_failure_reason","ba_post_geom_failure_reason",
        "no_cand_summary_missing","ba_cand_summary_missing",
        "no_accepted_but_bad_geom","ba_accepted_but_bad_geom","q_threshold_source",
        "no_reproj_med_px","ba_reproj_med_px","no_reproj_p90_px","ba_reproj_p90_px",
        "no_cheirality_ratio","ba_cheirality_ratio","no_tri_angle_med_deg","ba_tri_angle_med_deg",
        "gt_rot_med_deg","gt_trans_dir_med_deg","gt_scale_fit",
        "no_error","ba_error",
    ]
    keys = set()
    for r in rows:
        keys |= set(r.keys())
    for k in preferred:
        if k in keys:
            fieldnames.append(k)
            keys.remove(k)
    fieldnames.extend(sorted(list(keys)))

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # 写 summary.json（全局统计）
    summary = {"seq": seq_name, "num_runs": len(rows)}
    summary.update(gt_meta)
    summary["q_threshold_source"] = q_threshold_source
    summary["dataset_split"] = protocol_record["dataset_split"]
    summary["q_threshold_mode"] = protocol_record["q_threshold_mode"]
    summary["threshold_set_id"] = protocol_record["threshold_set_id"]
    # success rate（以 withBA 为准）
    ba_ok = [bool(r.get("ba_ok", False)) for r in rows]
    summary["withBA_success_rate"] = float(np.mean(ba_ok)) if ba_ok else 0.0
    ba_strict = [bool(r.get("ba_success_strict", False)) for r in rows]
    summary["withBA_success_strict_rate"] = float(np.mean(ba_strict)) if ba_strict else 0.0

    # H/E 比例（仅统计成功项）
    models = [r.get("model") for r in rows if r.get("ba_ok", False)]
    summary["model_H_ratio"] = float(np.mean([1.0 if m == "H" else 0.0 for m in models])) if models else None

    # 重投影误差统计（成功项）
    def _collect(key):
        xs = [r.get(key) for r in rows if r.get("ba_ok", False) and r.get(key) is not None]
        return np.asarray(xs, dtype=np.float64) if xs else None

    med_ba = _collect("ba_reproj_med_px")
    if med_ba is not None and med_ba.size:
        summary["ba_reproj_med_px_p50"] = float(np.median(med_ba))
        summary["ba_reproj_med_px_p90"] = float(np.percentile(med_ba, 90))

    q_ba_pre = _collect("ba_Q_pre")
    if q_ba_pre is not None and q_ba_pre.size:
        summary["ba_Q_pre_p50"] = float(np.median(q_ba_pre))
        summary["ba_Q_pre_p10"] = float(np.percentile(q_ba_pre, 10))

    q_ba = _collect("ba_Q_post")
    if q_ba is not None and q_ba.size:
        summary["ba_Q_post_p50"] = float(np.median(q_ba))
        summary["ba_Q_post_p10"] = float(np.percentile(q_ba, 10))
        summary["ba_Q_p50"] = float(np.median(q_ba))
        summary["ba_Q_p10"] = float(np.percentile(q_ba, 10))

    q_ba_geom = _collect("ba_Q_post_geom_only")
    if q_ba_geom is not None and q_ba_geom.size:
        summary["ba_Q_post_geom_only_p50"] = float(np.median(q_ba_geom))
        summary["ba_Q_post_geom_only_p10"] = float(np.percentile(q_ba_geom, 10))

    gates = [str(r.get("ba_gate_post", r.get("ba_gate_decision", ""))) for r in rows]
    if gates:
        pre_gates = [str(r.get("ba_gate_pre", "")) for r in rows]
        summary["ba_gate_pre_accept_ratio"] = float(np.mean([1.0 if g == "pre_accept" else 0.0 for g in pre_gates]))
        summary["ba_gate_pre_delay_ratio"] = float(np.mean([1.0 if g == "pre_delay" else 0.0 for g in pre_gates]))
        summary["ba_gate_pre_reset_ratio"] = float(np.mean([1.0 if g == "pre_reset" else 0.0 for g in pre_gates]))
        summary["ba_gate_accept_ratio"] = float(np.mean([1.0 if g == "accept" else 0.0 for g in gates]))
        summary["ba_gate_delay_ratio"] = float(np.mean([1.0 if g == "delay" else 0.0 for g in gates]))
        summary["ba_gate_reset_ratio"] = float(np.mean([1.0 if g == "reset" else 0.0 for g in gates]))
        summary["ba_accepted_but_bad_geom_count"] = int(sum(int(r.get("ba_accepted_but_bad_geom", 0)) for r in rows))
        summary["ba_accepted_but_bad_geom_ratio"] = float(
            np.mean([float(r.get("ba_accepted_but_bad_geom", 0)) for r in rows])
        )
        summary["ba_cand_summary_missing_ratio"] = float(
            np.mean([1.0 if int(r.get("ba_cand_summary_missing", 0)) == 1 else 0.0 for r in rows])
        )

    chei_ba = _collect("ba_cheirality_ratio")
    if chei_ba is not None and chei_ba.size:
        summary["ba_cheirality_p50"] = float(np.median(chei_ba))
        summary["ba_cheirality_p10"] = float(np.percentile(chei_ba, 10))

    gt_rot = _collect("gt_rot_med_deg")
    if gt_rot is not None and gt_rot.size:
        summary["gt_rot_med_deg_p50"] = float(np.median(gt_rot))
        summary["gt_rot_med_deg_p90"] = float(np.percentile(gt_rot, 90))

    gt_dir = _collect("gt_trans_dir_med_deg")
    if gt_dir is not None and gt_dir.size:
        summary["gt_trans_dir_med_deg_p50"] = float(np.median(gt_dir))
        summary["gt_trans_dir_med_deg_p90"] = float(np.percentile(gt_dir, 90))

    json_path = os.path.join(out_root, "summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    manifest = {
        "script_name": Path(__file__).name,
        "output_root": out_root,
        "protocol_file": protocol_path,
        "summary_csv": csv_path,
        "summary_json": json_path,
        "per_run_glob": os.path.join(out_root, "start_*", "*", "per_run_metrics.json"),
    }
    manifest_path = os.path.join(out_root, "experiment_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[DONE] summary.csv -> {csv_path}")
    print(f"[DONE] summary.json -> {json_path}")
    print(f"[DONE] per-start outputs -> {out_root}/start_xxxxxx/")

if __name__ == "__main__":
    main()
