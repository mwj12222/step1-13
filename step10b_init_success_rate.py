#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
step10b_init_success_rate.py

滑窗评估 SFM 初始化成功率（论文级实验）：
- 在序列上以 stride 滑动起点
- 每个起点取 win 帧做一次“从零初始化”
- 统计 success/fail、模型(E/H)、tri_points、pnp_success_rate 等
- 输出 CSV 到 eval_root/sfm_init_success_rate/win{win}_s{stride}/init_success_summary.csv

用法示例：
  python step10b_init_success_rate.py
  python step10b_init_success_rate.py --config /abs/path/to/config_kitti_00.yaml --eval_frames 1000 --stride 5
"""

from __future__ import annotations

from pathlib import Path
import sys
import os
import time
import json
import csv
import copy
import argparse
import numpy as np

# 可选导入：减少环境差异导致的脚本崩溃
try:
    import cv2
except ImportError:
    cv2 = None

# ============= 自动定位项目根目录（通用写法） =============
THIS_FILE = Path(__file__).resolve()
THIS_DIR = THIS_FILE.parent

PROJECT_ROOT = None
for p in [THIS_DIR, *THIS_DIR.parents]:
    if (p / "configs").is_dir() and ((p / "pipelines").is_dir() or (p / " pipelines").is_dir()):
        PROJECT_ROOT = p
        break

if PROJECT_ROOT is None:
    raise RuntimeError("Cannot locate project root (need configs/ and pipelines/).")

sys.path.insert(0, str(PROJECT_ROOT / "src" / "common"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "sfm"))

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


def str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def project_uv(K: np.ndarray, R: np.ndarray, t: np.ndarray, Xw: np.ndarray):
    Xc = R @ Xw.reshape(3) + t.reshape(3)
    z = float(Xc[2])
    if z <= 1e-9:
        return None, z
    x = Xc[:2] / z
    uv = (K[:2, :2] @ x.reshape(2, 1)).reshape(2) + K[:2, 2]
    return uv.astype(np.float64), z


def compute_reproj_stats(K, poses, map_points, tracks, tri_min_depth=0.0):
    track_map = {int(tr["id"]): tr for tr in tracks}
    errs = []
    pose_fids = set(int(k) for k in poses.keys())

    for mp in map_points:
        tid = int(mp.get("track_id", -1))
        if tid not in track_map:
            continue
        X = np.asarray(mp["xyz"], dtype=np.float64).reshape(3)
        tr = track_map[tid]
        for fi, uv in zip(tr["frames"], tr["uvs"]):
            fi = int(fi)
            if fi not in pose_fids:
                continue
            R, t = poses[fi]
            uv_hat, z = project_uv(K, R, t, X)
            if uv_hat is None or z <= tri_min_depth + 1e-9:
                continue
            uv = np.asarray(uv, dtype=np.float64).reshape(2)
            errs.append(float(np.linalg.norm(uv_hat - uv)))

    if not errs:
        return {"reproj_med_px": None, "reproj_p90_px": None}
    errs = np.asarray(errs, dtype=np.float64)
    return {
        "reproj_med_px": float(np.median(errs)),
        "reproj_p90_px": float(np.percentile(errs, 90)),
    }


def compute_cheirality_ratio(poses, map_points, tracks, tri_min_depth=0.0):
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
        obs_frames = [int(fi) for fi in tr["frames"] if int(fi) in pose_fids]
        if not obs_frames:
            continue
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
    if tot == 0:
        return None
    return float(ok / float(tot))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=resolve_default_cfg(),
        help="Path to config yaml",
    )
    parser.add_argument("--eval_frames", type=int, default=200, help="How many frames to evaluate (start from 0).")
    parser.add_argument("--stride", type=int, default=10, help="Window start stride (e.g. 10 then 5).")
    parser.add_argument("--seed", type=int, default=0, help="OpenCV RNG seed for RANSAC stability.")
    parser.add_argument("--out_tag", type=str, default="", help="Extra tag appended to output folder name.")
    parser.add_argument(
        "--enable_ba",
        action="store_true",
        help="Enable two-view/global BA during scan (slower). Default: disabled for speed.",
    )
    parser.add_argument("--step7_tag", type=str, default="", help="Optional variant tag under step7_root.")
    parser.add_argument("--q_pre_accept_threshold", type=float, default=-1.0)
    parser.add_argument("--q_pre_delay_threshold", type=float, default=-1.0)
    parser.add_argument("--q_post_accept_threshold", type=float, default=-1.0)
    parser.add_argument("--q_post_delay_threshold", type=float, default=-1.0)
    parser.add_argument("--q_accept_threshold", type=float, default=-1.0)
    parser.add_argument("--q_delay_threshold", type=float, default=-1.0)
    parser.add_argument("--gate_force_success_only", type=str2bool, default=None)
    parser.add_argument("--dataset_split", type=str, default="", choices=["", "train", "val", "test", "all", "unspecified"])
    parser.add_argument("--q_threshold_mode", type=str, default="", choices=["", "frozen", "tuning"])
    parser.add_argument("--threshold_set_id", type=str, default="")
    args = parser.parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    base_cfg = load_cfg(str(cfg_path))
    base_paths = get_common_paths(base_cfg)
    if args.dataset_split or args.q_threshold_mode or args.threshold_set_id:
        base_cfg.setdefault("experiment_protocol", {})
    if args.dataset_split:
        base_cfg["experiment_protocol"]["dataset_split"] = str(args.dataset_split)
    if args.q_threshold_mode:
        base_cfg["experiment_protocol"]["q_threshold_mode"] = str(args.q_threshold_mode)
    if args.threshold_set_id:
        base_cfg["experiment_protocol"]["threshold_set_id"] = str(args.threshold_set_id)
    q_threshold_overridden = (
        args.q_pre_accept_threshold > 0
        or args.q_pre_delay_threshold > 0
        or args.q_post_accept_threshold > 0
        or args.q_post_delay_threshold > 0
        or args.q_accept_threshold > 0
        or args.q_delay_threshold > 0
        or args.gate_force_success_only is not None
    )
    validate_experiment_protocol(base_cfg, q_threshold_overridden)
    if q_threshold_overridden:
        base_cfg.setdefault("init_quality", {})
    if args.q_pre_accept_threshold > 0:
        base_cfg["init_quality"]["q_pre_accept_threshold"] = float(args.q_pre_accept_threshold)
    if args.q_pre_delay_threshold > 0:
        base_cfg["init_quality"]["q_pre_delay_threshold"] = float(args.q_pre_delay_threshold)
    if args.q_post_accept_threshold > 0:
        base_cfg["init_quality"]["q_post_accept_threshold"] = float(args.q_post_accept_threshold)
    if args.q_post_delay_threshold > 0:
        base_cfg["init_quality"]["q_post_delay_threshold"] = float(args.q_post_delay_threshold)
    if args.q_accept_threshold > 0:
        base_cfg["init_quality"]["q_post_accept_threshold"] = float(args.q_accept_threshold)
        base_cfg["init_quality"]["accept_threshold"] = float(args.q_accept_threshold)
    if args.q_delay_threshold > 0:
        base_cfg["init_quality"]["q_post_delay_threshold"] = float(args.q_delay_threshold)
        base_cfg["init_quality"]["delay_threshold"] = float(args.q_delay_threshold)
    if args.gate_force_success_only is not None:
        base_cfg["init_quality"]["gate_force_success_only"] = bool(args.gate_force_success_only)
    q_threshold_source = "cli" if q_threshold_overridden else "config"

    if args.step7_tag:
        base_paths["step7_root"] = os.path.join(base_paths["step7_root"], args.step7_tag)
        base_paths["kps_npz_dir_step7"] = os.path.join(base_paths["step7_root"], "kps_npz")
    front_frame_rows = load_step7_frame_metrics(base_paths["step7_root"])

    # 窗口长度：默认用你现有初始化的 track_max_frames
    win = int(base_cfg.get("sfm", {}).get("track_max_frames", 11))
    stride = int(args.stride)
    eval_frames = int(args.eval_frames)

    if eval_frames < win:
        raise ValueError(f"eval_frames ({eval_frames}) must be >= win ({win}).")

    # 输出目录
    tag = f"_{args.out_tag}" if args.out_tag else ""
    out_root = os.path.join(base_paths["eval_root"], "sfm_init_success_rate", f"win{win}_s{stride}{tag}")
    os.makedirs(out_root, exist_ok=True)
    protocol_record = build_experiment_protocol_record(
        base_cfg,
        q_threshold_overridden=q_threshold_overridden,
        script_name=Path(__file__).name,
        cfg_path=str(cfg_path),
        out_root=out_root,
    )
    q_threshold_source = protocol_record["q_threshold_source"]
    protocol_path = os.path.join(out_root, "experiment_protocol.json")
    with open(protocol_path, "w", encoding="utf-8") as f:
        json.dump(protocol_record, f, ensure_ascii=False, indent=2)

    # 固定随机种子，减少 RANSAC 抖动
    if cv2 is not None:
        cv2.setRNGSeed(int(args.seed))

    rows = []
    total = 0
    solver_succ = 0
    strict_succ = 0

    # 检查：是否支持 start_frame（避免“统计跑了，但每次还是从0开始”）
    # 如果你已按之前建议改了 sfm_static_utils.py，这里无需担心。
    # 这只是提醒用。
    if "sfm" not in base_cfg:
        base_cfg["sfm"] = {}

    print(f"[InitSuccess] config={cfg_path}")
    print(f"[InitSuccess] eval_frames={eval_frames}, win={win}, stride={stride}, out={out_root}")

    for start in range(0, eval_frames - win + 1, stride):
        total += 1
        cfg = copy.deepcopy(base_cfg)
        cfg.setdefault("sfm", {})
        cfg["sfm"]["start_frame"] = int(start)

        cfg["sfm"]["track_max_frames"] = int(win)
        cfg["sfm"]["max_frames"] = int(win)
        cfg["sfm"]["init_window_frames"] = int(max(2, win - 1))

        if not bool(args.enable_ba):
            cfg["sfm"]["enable_two_view_ba"] = False
            cfg["sfm"]["enable_global_ba"] = False

        one_out = os.path.join(out_root, f"start_{start:06d}")
        os.makedirs(one_out, exist_ok=True)

        paths = dict(base_paths)
        paths["sfm_out_dir"] = one_out

        t0 = time.time()

        ok = 1
        err = ""
        n_frames = 0
        n_pts3d = 0

        pivot = -1
        model = ""
        tri_points = 0
        tri_candidate_tracks = 0
        pnp_success_rate = 0.0
        pnp_med_inliers = 0.0
        allow_H = None
        parallax_px_candidate = 0.0
        triangulation_ratio = 0.0
        reproj_med_px = None
        reproj_p90_px = None
        cheirality_ratio = None
        front_metrics = aggregate_front_window_metrics(front_frame_rows, start, win, cfg=cfg)

        try:
            poses, pts3d, map_points, frames, tracks = run_static_sfm_v2(cfg, paths, verbose=False)
            n_frames = len(poses)
            n_pts3d = int(pts3d.shape[0]) if hasattr(pts3d, "shape") else 0
            K, _ = get_camera_from_cfg(cfg)
            tri_min_depth = float(cfg.get("sfm", {}).get("tri_min_depth", 0.0))
            reproj_stats = compute_reproj_stats(K, poses, map_points, tracks, tri_min_depth=tri_min_depth)
            reproj_med_px = reproj_stats.get("reproj_med_px")
            reproj_p90_px = reproj_stats.get("reproj_p90_px")
            cheirality_ratio = compute_cheirality_ratio(poses, map_points, tracks, tri_min_depth=tri_min_depth)
            solver_succ += 1
        except Exception as e:
            ok = 0
            err = str(e)[:200]

        dt = time.time() - t0

        cand_summary = {}
        cand_summary_missing = 1
        metrics_path = os.path.join(one_out, "init_candidates_metrics.json")
        if os.path.exists(metrics_path):
            try:
                cand_summary = summarize_candidate_pool(load_candidate_metrics_json(metrics_path))
                if cand_summary:
                    cand_summary_missing = int(cand_summary.get("cand_summary_missing", 0))
                    pivot = int(cand_summary.get("cand_pivot", -1))
                    model = str(cand_summary.get("cand_model", ""))
                    tri_points = int(cand_summary.get("cand_tri_points", 0))
                    tri_candidate_tracks = int(cand_summary.get("cand_tri_candidate_tracks", 0))
                    parallax_px_candidate = float(cand_summary.get("cand_parallax_px", 0.0))
                    triangulation_ratio = float(
                        cand_summary.get("qcand_triangulation_ratio", cand_summary.get("cand_triangulation_ratio", 0.0))
                    )
                    pnp_success_rate = float(cand_summary.get("cand_pnp_success_rate", 0.0))
                    pnp_med_inliers = float(cand_summary.get("cand_pnp_median_inliers", 0.0))
                    allow_H = cand_summary.get("cand_allow_H", None)
            except Exception as e:
                err = (err + " | metrics_parse_fail:" + str(e))[:200]
        if cand_summary:
            cand_summary = dict(cand_summary)
            cand_summary.setdefault("cand_summary_missing", cand_summary_missing)
        else:
            cand_summary = {"cand_summary_missing": cand_summary_missing}

        geom_metrics = {
            "reproj_med_px": reproj_med_px,
            "reproj_p90_px": reproj_p90_px,
            "cheirality_ratio": cheirality_ratio,
            "triangulation_ratio": triangulation_ratio,
        }
        q_pre_metrics = build_q_pre_metrics(cfg, front_metrics, cand_summary)
        q_post_metrics = build_q_post_metrics(cfg, q_pre_metrics, geom_metrics, bool(ok))
        gate_metrics = decide_gate(cfg, q_pre_metrics, q_post_metrics, bool(ok))
        geom_quality = evaluate_post_geom_quality(cfg, geom_metrics)
        q_metrics = {}
        q_metrics.update(q_pre_metrics)
        q_metrics.update(q_post_metrics)
        q_metrics.update(gate_metrics)
        q_metrics.update(geom_quality)
        q_metrics["Q"] = float(q_post_metrics.get("Q_post", 0.0))
        q_metrics["gate_decision"] = gate_metrics.get("gate_post")
        q_metrics["q_accept_threshold"] = q_post_metrics.get("q_post_accept_threshold")
        q_metrics["q_delay_threshold"] = q_post_metrics.get("q_post_delay_threshold")
        q_metrics["q_threshold_source"] = q_threshold_source

        solver_ok = bool(ok)
        geom_ok = bool(solver_ok and geom_quality.get("post_geom_strict_ok", False))
        success_strict = int(solver_ok and geom_ok)
        strict_succ += success_strict

        row = {
            "start": int(start),
            "solver_ok": int(solver_ok),
            "geom_ok": int(geom_ok),
            "success_strict": int(success_strict),
            "success": int(success_strict),
            "n_frames": int(n_frames),
            "n_pts3d": int(n_pts3d),
            "pivot": int(pivot),
            "model": model,
            "tri_points": int(tri_points),
            "tri_candidate_tracks": int(tri_candidate_tracks),
            "parallax_px_candidate": float(parallax_px_candidate),
            "triangulation_ratio": float(triangulation_ratio),
            "pnp_success_rate": float(pnp_success_rate),
            "pnp_median_inliers": float(pnp_med_inliers),
            "allow_H": allow_H,
            "reproj_med_px": reproj_med_px,
            "reproj_p90_px": reproj_p90_px,
            "cheirality_ratio": cheirality_ratio,
            "geom_failure_reason": geom_quality.get("post_geom_failure_reason"),
            "accepted_but_bad_geom": int(gate_metrics.get("gate_post") == "accept" and not geom_ok),
            "cand_summary_missing": int(cand_summary_missing),
            "q_threshold_source": q_threshold_source,
            "time_sec": float(dt),
            "error": err,
        }
        row.update(front_metrics)
        row.update(cand_summary)
        row.update(q_metrics)
        rows.append(row)

        per_run_path = os.path.join(one_out, "init_stage1_metrics.json")
        with open(per_run_path, "w", encoding="utf-8") as f:
            json.dump(row, f, ensure_ascii=False, indent=2)

        print(f"[InitSuccess] start={start:06d} solver_ok={int(solver_ok)} geom_ok={int(geom_ok)} "
              f"strict={success_strict} model={model} pivot={pivot} "
              f"tri={tri_points} pnp_sr={pnp_success_rate:.2f} "
              f"Qpre={row['Q_pre']:.3f} Qpost={row['Q_post']:.3f} gate={row['gate_post']} "
              f"pts3d={n_pts3d} time={dt:.2f}s")

    csv_path = os.path.join(out_root, "init_success_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        if rows:
            fieldnames = []
            seen = set()
            for r in rows:
                for k in r.keys():
                    if k not in seen:
                        seen.add(k)
                        fieldnames.append(k)
        else:
            fieldnames = ["start", "success", "n_frames", "n_pts3d", "error"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    solver_rate = solver_succ / max(1, total)
    strict_rate = strict_succ / max(1, total)
    print(f"\n[InitSuccess] solver_success_rate = {solver_succ}/{total} = {solver_rate:.3f}")
    print(f"[InitSuccess] strict_success_rate = {strict_succ}/{total} = {strict_rate:.3f}")
    print(f"[InitSuccess] saved: {csv_path}")

    summary = {
        "num_runs": int(total),
        "solver_success_rate": float(solver_rate),
        "strict_success_rate": float(strict_rate),
        "q_threshold_source": q_threshold_source,
        "dataset_split": protocol_record["dataset_split"],
        "q_threshold_mode": protocol_record["q_threshold_mode"],
        "threshold_set_id": protocol_record["threshold_set_id"],
    }
    if rows:
        gate_pre = [str(r.get("gate_pre", "")) for r in rows]
        gate_post = [str(r.get("gate_post", "")) for r in rows]
        summary["gate_pre_accept_ratio"] = float(np.mean([1.0 if g == "pre_accept" else 0.0 for g in gate_pre]))
        summary["gate_pre_delay_ratio"] = float(np.mean([1.0 if g == "pre_delay" else 0.0 for g in gate_pre]))
        summary["gate_pre_reset_ratio"] = float(np.mean([1.0 if g == "pre_reset" else 0.0 for g in gate_pre]))
        summary["gate_post_accept_ratio"] = float(np.mean([1.0 if g == "accept" else 0.0 for g in gate_post]))
        summary["gate_post_delay_ratio"] = float(np.mean([1.0 if g == "delay" else 0.0 for g in gate_post]))
        summary["gate_post_reset_ratio"] = float(np.mean([1.0 if g == "reset" else 0.0 for g in gate_post]))
        summary["accepted_but_bad_geom_count"] = int(sum(int(r.get("accepted_but_bad_geom", 0)) for r in rows))
        summary["accepted_but_bad_geom_ratio"] = float(np.mean([float(r.get("accepted_but_bad_geom", 0)) for r in rows]))
        summary["cand_summary_missing_ratio"] = float(np.mean([1.0 if int(r.get("cand_summary_missing", 0)) == 1 else 0.0 for r in rows]))
        summary["post_geom_failure_high_reprojection_ratio"] = float(
            np.mean([1.0 if r.get("post_geom_failure_reason") == "high_reprojection" else 0.0 for r in rows])
        )
        summary["post_geom_failure_bad_depth_ratio"] = float(
            np.mean([1.0 if r.get("post_geom_failure_reason") == "bad_depth_sign" else 0.0 for r in rows])
        )
        summary["post_geom_failure_weak_triangulation_ratio"] = float(
            np.mean([1.0 if r.get("post_geom_failure_reason") == "weak_triangulation" else 0.0 for r in rows])
        )
    summary_path = os.path.join(out_root, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    manifest = {
        "script_name": Path(__file__).name,
        "output_root": out_root,
        "protocol_file": protocol_path,
        "summary_csv": csv_path,
        "summary_json": summary_path,
        "per_run_glob": os.path.join(out_root, "start_*", "init_stage1_metrics.json"),
    }
    manifest_path = os.path.join(out_root, "experiment_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # 额外给一个简短统计（论文可直接写）
    # 统计成功样本中 E/H 比例
    if strict_succ > 0:
        models = [r["model"] for r in rows if int(r["success"]) == 1]
        e_cnt = sum(1 for m in models if m == "E")
        h_cnt = sum(1 for m in models if m == "H")
        print(f"[InitSuccess] among success: E={e_cnt}, H={h_cnt}, E_ratio={e_cnt/max(1,(e_cnt+h_cnt)):.3f}")
        qs = np.asarray([float(r["Q"]) for r in rows if int(r["success"]) == 1], dtype=np.float64)
        print(f"[InitSuccess] Q(success) mean={float(np.mean(qs)):.3f} p10={float(np.percentile(qs, 10)):.3f}")


if __name__ == "__main__":
    main()
