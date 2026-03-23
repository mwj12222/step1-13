#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import json
import os

import numpy as np


def load_step7_frame_metrics(step7_root: str) -> list[dict]:
    csv_path = os.path.join(step7_root, "frame_metrics.csv")
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mean_from_rows(rows: list[dict], key: str, default: float = 0.0) -> float:
    vals = []
    for row in rows:
        try:
            vals.append(float(row[key]))
        except (KeyError, TypeError, ValueError):
            continue
    if not vals:
        return float(default)
    return float(np.mean(np.asarray(vals, dtype=np.float64)))


def percentile_from_rows(rows: list[dict], key: str, q: float, default: float = 0.0) -> float:
    vals = []
    for row in rows:
        try:
            vals.append(float(row[key]))
        except (KeyError, TypeError, ValueError):
            continue
    if not vals:
        return float(default)
    return float(np.percentile(np.asarray(vals, dtype=np.float64), float(q)))


def min_from_rows(rows: list[dict], key: str, default: float = 0.0) -> float:
    vals = []
    for row in rows:
        try:
            vals.append(float(row[key]))
        except (KeyError, TypeError, ValueError):
            continue
    if not vals:
        return float(default)
    return float(np.min(np.asarray(vals, dtype=np.float64)))


def max_from_rows(rows: list[dict], key: str, default: float = 0.0) -> float:
    vals = []
    for row in rows:
        try:
            vals.append(float(row[key]))
        except (KeyError, TypeError, ValueError):
            continue
    if not vals:
        return float(default)
    return float(np.max(np.asarray(vals, dtype=np.float64)))


def _normalized_triplet(weights, fallback):
    vals = []
    for i, default in enumerate(fallback):
        try:
            vals.append(max(0.0, float(weights[i])))
        except (TypeError, ValueError, IndexError):
            vals.append(float(default))
    s = float(sum(vals))
    if s <= 1e-12:
        vals = list(map(float, fallback))
        s = float(sum(vals))
    return tuple(float(v / s) for v in vals)


def _get_front_blend_weights(cfg=None):
    q_cfg = (cfg or {}).get("init_quality", {}) if isinstance(cfg, dict) else {}
    good = _normalized_triplet(
        q_cfg.get("front_blend_good", [0.20, 0.35, 0.45]),
        (0.20, 0.35, 0.45),
    )
    bad = _normalized_triplet(
        q_cfg.get("front_blend_bad", [0.25, 0.35, 0.40]),
        (0.25, 0.35, 0.40),
    )
    return good, bad


def get_experiment_protocol(cfg) -> dict:
    proto_cfg = cfg.get("experiment_protocol", {}) if isinstance(cfg, dict) else {}
    dataset_split = str(proto_cfg.get("dataset_split", proto_cfg.get("split", "test"))).strip().lower()
    q_threshold_mode = str(proto_cfg.get("q_threshold_mode", "frozen")).strip().lower()
    threshold_set_id = str(proto_cfg.get("threshold_set_id", "viode_q_default_frozen")).strip()
    notes = str(proto_cfg.get("notes", "")).strip()

    valid_splits = {"train", "val", "test", "all", "unspecified"}
    valid_modes = {"frozen", "tuning"}
    if dataset_split not in valid_splits:
        raise ValueError(f"Invalid experiment_protocol.dataset_split: {dataset_split}")
    if q_threshold_mode not in valid_modes:
        raise ValueError(f"Invalid experiment_protocol.q_threshold_mode: {q_threshold_mode}")

    return {
        "dataset_split": dataset_split,
        "q_threshold_mode": q_threshold_mode,
        "threshold_set_id": threshold_set_id,
        "notes": notes,
    }


def validate_experiment_protocol(cfg, q_threshold_overridden: bool) -> dict:
    protocol = get_experiment_protocol(cfg)
    if q_threshold_overridden and protocol["q_threshold_mode"] == "frozen":
        raise ValueError(
            "Threshold overrides are disabled because experiment_protocol.q_threshold_mode=frozen."
        )
    if q_threshold_overridden and protocol["dataset_split"] == "test":
        raise ValueError(
            "Threshold overrides on test split are disabled. Use train/val with q_threshold_mode=tuning."
        )
    return protocol


def build_experiment_protocol_record(cfg, q_threshold_overridden: bool, script_name: str, cfg_path: str, out_root: str) -> dict:
    protocol = get_experiment_protocol(cfg)
    thresholds = _get_q_thresholds(cfg)
    front_blend_good, front_blend_bad = _get_front_blend_weights(cfg)
    return {
        "script_name": str(script_name),
        "config_path": str(cfg_path),
        "output_root": str(out_root),
        "dataset_split": protocol["dataset_split"],
        "q_threshold_mode": protocol["q_threshold_mode"],
        "threshold_set_id": protocol["threshold_set_id"],
        "notes": protocol["notes"],
        "q_threshold_overridden": bool(q_threshold_overridden),
        "q_threshold_source": "cli" if q_threshold_overridden else "config",
        "q_pre_accept_threshold": thresholds["q_pre_accept_threshold"],
        "q_pre_delay_threshold": thresholds["q_pre_delay_threshold"],
        "q_post_accept_threshold": thresholds["q_post_accept_threshold"],
        "q_post_delay_threshold": thresholds["q_post_delay_threshold"],
        "gate_force_success_only": thresholds["force_success_only"],
        "front_blend_good": [float(v) for v in front_blend_good],
        "front_blend_bad": [float(v) for v in front_blend_bad],
    }


def _blend_good(mean_v: float, p10_v: float, min_v: float, weights=(0.20, 0.35, 0.45)) -> float:
    # Good metrics should be dragged down by weak frames rather than hidden by the mean.
    w_mean, w_p10, w_min = weights
    return float(
        np.clip(
            float(w_mean) * float(mean_v) + float(w_p10) * float(p10_v) + float(w_min) * float(min_v),
            0.0,
            1.0,
        )
    )


def _blend_bad(mean_v: float, p90_v: float, max_v: float, weights=(0.25, 0.35, 0.40)) -> float:
    # Bad metrics should reflect both tail risk and the single worst frame.
    w_mean, w_p90, w_max = weights
    return float(
        np.clip(
            max(
                float(w_mean) * float(mean_v) + float(w_p90) * float(p90_v) + float(w_max) * float(max_v),
                float(p90_v),
            ),
            0.0,
            1.0,
        )
    )


def aggregate_front_window_metrics(frame_rows: list[dict], start: int, win: int, cfg=None) -> dict:
    if not frame_rows:
        return {}
    sub = frame_rows[start:start + win]
    if not sub:
        return {}
    good_weights, bad_weights = _get_front_blend_weights(cfg)

    p_static_mean = mean_from_rows(sub, "kept_r_mean")
    p_static_p10 = percentile_from_rows(sub, "kept_r_mean", 10.0, default=p_static_mean)
    p_static_min = min_from_rows(sub, "kept_r_mean", default=p_static_p10)
    coverage_mean = mean_from_rows(sub, "kept_coverage_ratio")
    coverage_p10 = percentile_from_rows(sub, "kept_coverage_ratio", 10.0, default=coverage_mean)
    coverage_min = min_from_rows(sub, "kept_coverage_ratio", default=coverage_p10)
    entropy_mean = mean_from_rows(sub, "kept_grid_entropy")
    entropy_p10 = percentile_from_rows(sub, "kept_grid_entropy", 10.0, default=entropy_mean)
    entropy_min = min_from_rows(sub, "kept_grid_entropy", default=entropy_p10)
    band_mean = mean_from_rows(sub, "kept_band_ratio")
    band_p90 = percentile_from_rows(sub, "kept_band_ratio", 90.0, default=band_mean)
    band_max = max_from_rows(sub, "kept_band_ratio", default=band_p90)
    dyn_mean = mean_from_rows(sub, "kept_dyn_ratio")
    dyn_p90 = percentile_from_rows(sub, "kept_dyn_ratio", 90.0, default=dyn_mean)
    dyn_max = max_from_rows(sub, "kept_dyn_ratio", default=dyn_p90)

    return {
        "front_num_frames": len(sub),
        "front_mask_dyn_ratio": mean_from_rows(sub, "mask_dyn_ratio"),
        "front_n_all": mean_from_rows(sub, "all_n"),
        "front_n_kept": mean_from_rows(sub, "kept_n"),
        "front_all_dyn_ratio": mean_from_rows(sub, "all_dyn_ratio"),
        "front_kept_dyn_ratio_mean": dyn_mean,
        "front_kept_dyn_ratio_p90": dyn_p90,
        "front_kept_dyn_ratio_max": dyn_max,
        "front_kept_dyn_ratio": _blend_bad(dyn_mean, dyn_p90, dyn_max, weights=bad_weights),
        "front_p_static_mean": p_static_mean,
        "front_p_static_p10": p_static_p10,
        "front_p_static_min": p_static_min,
        "front_p_static": _blend_good(p_static_mean, p_static_p10, p_static_min, weights=good_weights),
        "front_r_p10": mean_from_rows(sub, "kept_r_p10"),
        "front_p_band_mean": band_mean,
        "front_p_band_p90": band_p90,
        "front_p_band_max": band_max,
        "front_p_band": _blend_bad(band_mean, band_p90, band_max, weights=bad_weights),
        "front_coverage_mean": coverage_mean,
        "front_coverage_p10": coverage_p10,
        "front_coverage_min": coverage_min,
        "front_coverage_ratio": _blend_good(coverage_mean, coverage_p10, coverage_min, weights=good_weights),
        "front_grid_entropy_mean": entropy_mean,
        "front_grid_entropy_p10": entropy_p10,
        "front_grid_entropy_min": entropy_min,
        "front_grid_entropy": _blend_good(entropy_mean, entropy_p10, entropy_min, weights=good_weights),
        "front_blend_good_mean_w": float(good_weights[0]),
        "front_blend_good_p10_w": float(good_weights[1]),
        "front_blend_good_min_w": float(good_weights[2]),
        "front_blend_bad_mean_w": float(bad_weights[0]),
        "front_blend_bad_p90_w": float(bad_weights[1]),
        "front_blend_bad_max_w": float(bad_weights[2]),
        "front_dist_mean": mean_from_rows(sub, "kept_dist_mean"),
    }


def load_candidate_metrics_json(candidate_json: str) -> list[dict]:
    if not candidate_json or not os.path.exists(candidate_json):
        return []
    with open(candidate_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return []
    return [m for m in data if isinstance(m, dict)]


def pick_solver_best_from_metrics(metrics: list[dict]) -> dict | None:
    oks = [m for m in metrics if m.get("status") == "ok"]
    if not oks:
        return None
    oks_eonly = [m for m in oks if m.get("allow_H") is False]
    pool = oks_eonly if oks_eonly else oks
    return max(pool, key=lambda x: float(x.get("score", -1e18)))


def _candidate_score_free_metrics(m: dict) -> dict:
    pnp_total = max(1, int(m.get("pnp_total", 0)))
    return {
        "parallax": float(m.get("parallax", 0.0)),
        "tri_points": float(m.get("tri_points", 0.0)),
        "triangulation_ratio": float(m.get("triangulation_ratio", 0.0)),
        "pnp_success_rate": float(int(m.get("pnp_success", 0)) / pnp_total),
        "pnp_median_inliers": float(m.get("pnp_median_inliers", 0.0)),
    }


def _mean_quantile_rank(best_metrics: dict, pool_metrics: list[dict]) -> float:
    ranks = []
    for key, best_v in best_metrics.items():
        vals = [float(pm[key]) for pm in pool_metrics if key in pm]
        if not vals:
            continue
        ranks.append(float(sum(1 for v in vals if v <= best_v) / len(vals)))
    if not ranks:
        return 0.0
    return float(np.clip(np.mean(np.asarray(ranks, dtype=np.float64)), 0.0, 1.0))


def _is_pareto_consistent(best_metrics: dict, pool_metrics: list[dict], eps: float = 1e-9) -> float:
    for pm in pool_metrics:
        all_ge = True
        any_gt = False
        for key, best_v in best_metrics.items():
            other_v = float(pm[key])
            if other_v + eps < best_v:
                all_ge = False
                break
            if other_v > best_v + eps:
                any_gt = True
        if all_ge and any_gt:
            return 0.0
    return 1.0


def pick_quality_best_from_metrics(metrics: list[dict]) -> dict | None:
    oks = [m for m in metrics if m.get("status") == "ok"]
    if not oks:
        return None

    pool_metrics = [_candidate_score_free_metrics(m) for m in oks]

    def _key(m: dict):
        mm = _candidate_score_free_metrics(m)
        return (
            _is_pareto_consistent(mm, pool_metrics),
            _mean_quantile_rank(mm, pool_metrics),
            mm["pnp_success_rate"],
            mm["triangulation_ratio"],
            mm["tri_points"],
            mm["parallax"],
            mm["pnp_median_inliers"],
        )

    return max(oks, key=_key)


def summarize_candidate_pool(metrics: list[dict]) -> dict:
    valid = [m for m in metrics if isinstance(m, dict)]
    oks = [m for m in valid if m.get("status") == "ok"]
    if not valid:
        return {}

    solver_best = pick_solver_best_from_metrics(valid)
    quality_best = pick_quality_best_from_metrics(valid)
    if quality_best is None:
        return {
            "cand_num_total": int(len(valid)),
            "cand_num_ok": 0,
            "cand_viable_ratio": 0.0,
            "cand_model_purity": 0.0,
            "cand_geom_rank_mean": 0.0,
            "cand_pareto_consistent": 0.0,
            "qcand_tri_candidate_tracks": 0,
            "qcand_triangulation_ratio": 0.0,
            "cand_tri_candidate_tracks": 0,
            "cand_triangulation_ratio": 0.0,
        }

    pool = oks

    best_model = str(quality_best.get("model", ""))
    model_purity = float(
        sum(1 for m in pool if str(m.get("model", "")) == best_model) / max(1, len(pool))
    )
    pool_metrics = [_candidate_score_free_metrics(m) for m in pool]
    best_metrics = _candidate_score_free_metrics(quality_best)
    geom_rank_mean = _mean_quantile_rank(best_metrics, pool_metrics)
    pareto_consistent = _is_pareto_consistent(best_metrics, pool_metrics)
    pnp_success = int(quality_best.get("pnp_success", 0))
    pnp_total = int(quality_best.get("pnp_total", 0))
    best_score = float(quality_best.get("score", 0.0))
    solver_best = solver_best or quality_best

    return {
        "cand_num_total": int(len(valid)),
        "cand_num_ok": int(len(oks)),
        "cand_viable_ratio": float(len(oks) / max(1, len(valid))),
        "cand_model_purity": model_purity,
        "cand_geom_rank_mean": geom_rank_mean,
        "cand_pareto_consistent": pareto_consistent,
        "qcand_pivot": int(quality_best.get("pivot", -1)),
        "qcand_model": str(quality_best.get("model", "")),
        "qcand_allow_H": quality_best.get("allow_H", None),
        "qcand_parallax_px": float(quality_best.get("parallax", 0.0)),
        "qcand_tri_points": int(quality_best.get("tri_points", 0)),
        "qcand_tri_candidate_tracks": int(quality_best.get("tri_candidate_tracks", 0)),
        "qcand_triangulation_ratio": float(quality_best.get("triangulation_ratio", 0.0)),
        "qcand_pnp_success_rate": float(pnp_success / max(1, pnp_total)),
        "qcand_pnp_median_inliers": float(quality_best.get("pnp_median_inliers", 0.0)),
        "qcand_score_proxy": best_score,
        "cand_pivot": int(solver_best.get("pivot", -1)),
        "cand_model": str(solver_best.get("model", "")),
        "cand_allow_H": solver_best.get("allow_H", None),
        "cand_parallax_px": float(solver_best.get("parallax", 0.0)),
        "cand_tri_points": int(solver_best.get("tri_points", 0)),
        "cand_tri_candidate_tracks": int(solver_best.get("tri_candidate_tracks", 0)),
        "cand_triangulation_ratio": float(solver_best.get("triangulation_ratio", 0.0)),
        "cand_pnp_success_rate": float(int(solver_best.get("pnp_success", 0)) / max(1, int(solver_best.get("pnp_total", 0)))),
        "cand_pnp_median_inliers": float(solver_best.get("pnp_median_inliers", 0.0)),
        "cand_score": float(solver_best.get("score", 0.0)),
        "cand_E_inliers": int(solver_best.get("nE", 0)),
        "cand_errE": float(solver_best.get("errE", 0.0)),
    }


def norm_clip(value, lo, hi):
    if value is None:
        return 0.0
    if hi <= lo:
        return 0.0
    return float(np.clip((float(value) - float(lo)) / float(hi - lo), 0.0, 1.0))


def norm_clip_inv(value, good_lo, bad_hi):
    if value is None:
        return 0.0
    if bad_hi <= good_lo:
        return 0.0
    return float(np.clip((float(bad_hi) - float(value)) / float(bad_hi - good_lo), 0.0, 1.0))


def safe_norm_clip(value, lo, hi, missing="zero"):
    if value is None:
        return 0.0 if missing != "neutral" else 0.5
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0 if missing != "neutral" else 0.5
    if np.isnan(v):
        return 0.0 if missing != "neutral" else 0.5
    return norm_clip(v, lo, hi)


def safe_norm_clip_inv(value, good_lo, bad_hi, missing="zero"):
    if value is None:
        return 0.0 if missing != "neutral" else 0.5
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0 if missing != "neutral" else 0.5
    if np.isnan(v):
        return 0.0 if missing != "neutral" else 0.5
    return norm_clip_inv(v, good_lo, bad_hi)


def _normalized_weights(weights: dict) -> dict:
    weight_sum = float(sum(max(0.0, w) for w in weights.values()))
    if weight_sum <= 1e-12:
        weight_sum = 1.0
    return {k: float(max(0.0, v) / weight_sum) for k, v in weights.items()}


def _score_from_components(components: dict, weights: dict) -> float:
    q_value = 0.0
    for key, weight in weights.items():
        q_value += float(weight) * float(components.get(key, 0.0))
    return float(np.clip(q_value, 0.0, 1.0))


def _get_q_thresholds(cfg) -> dict:
    q_cfg = cfg.get("init_quality", {})
    return {
        "q_pre_accept_threshold": float(q_cfg.get("q_pre_accept_threshold", 0.60)),
        "q_pre_delay_threshold": float(q_cfg.get("q_pre_delay_threshold", 0.50)),
        "q_post_accept_threshold": float(
            q_cfg.get("q_post_accept_threshold", q_cfg.get("post_accept_threshold", q_cfg.get("accept_threshold", 0.55)))
        ),
        "q_post_delay_threshold": float(
            q_cfg.get("q_post_delay_threshold", q_cfg.get("post_delay_threshold", q_cfg.get("delay_threshold", 0.50)))
        ),
        "force_success_only": bool(q_cfg.get("gate_force_success_only", False)),
    }


def evaluate_post_geom_quality(cfg, geom=None) -> dict:
    sfm_cfg = cfg.get("sfm", {})
    q_cfg = cfg.get("init_quality", {})
    geom = geom or {}

    reproj_med_bad = float(q_cfg.get("reproj_med_bad_px", max(float(sfm_cfg.get("tri_reproj_thresh", 3.0)), 3.0)))
    reproj_p90_bad = float(q_cfg.get("reproj_p90_bad_px", max(2.0 * float(sfm_cfg.get("tri_reproj_thresh", 3.0)), 6.0)))
    cheirality_min = float(q_cfg.get("cheirality_min", 0.7))
    triangulation_ratio_min = float(q_cfg.get("triangulation_ratio_min", 0.25))

    reproj_med_px = geom.get("reproj_med_px")
    reproj_p90_px = geom.get("reproj_p90_px")
    cheirality_ratio = geom.get("cheirality_ratio")
    triangulation_ratio = geom.get("triangulation_ratio")

    reproj_med_ok = reproj_med_px is not None and float(reproj_med_px) <= reproj_med_bad
    reproj_p90_ok = reproj_p90_px is not None and float(reproj_p90_px) <= reproj_p90_bad
    cheirality_ok = cheirality_ratio is not None and float(cheirality_ratio) >= cheirality_min
    triangulation_ok = triangulation_ratio is not None and float(triangulation_ratio) >= triangulation_ratio_min

    checks = [
        (reproj_med_ok, "high_reprojection"),
        (reproj_p90_ok, "high_tail_reprojection"),
        (cheirality_ok, "bad_depth_sign"),
        (triangulation_ok, "weak_triangulation"),
    ]
    fail_reason = "geom_quality_ok"
    for passed, reason in checks:
        if not passed:
            fail_reason = reason
            break

    return {
        "post_geom_reproj_med_ok": bool(reproj_med_ok),
        "post_geom_reproj_p90_ok": bool(reproj_p90_ok),
        "post_geom_cheirality_ok": bool(cheirality_ok),
        "post_geom_triangulation_ratio_ok": bool(triangulation_ok),
        "post_geom_strict_ok": bool(reproj_med_ok and reproj_p90_ok and cheirality_ok and triangulation_ok),
        "post_geom_failure_reason": fail_reason,
        "post_geom_reproj_med_bad_px": reproj_med_bad,
        "post_geom_reproj_p90_bad_px": reproj_p90_bad,
        "post_geom_cheirality_min": cheirality_min,
        "post_geom_triangulation_ratio_min": triangulation_ratio_min,
    }


def build_q_pre_metrics(cfg, front, cand):
    sfm_cfg = cfg.get("sfm", {})
    q_cfg = cfg.get("init_quality", {})
    tri_min_valid = float(sfm_cfg.get("tri_min_valid", 40.0))

    p_static = float(front.get("front_p_static", 0.0))
    coverage = float(front.get("front_coverage_ratio", 0.0))
    entropy = float(front.get("front_grid_entropy", 0.0))
    band_clean = 1.0 - float(front.get("front_p_band", 0.0))
    dyn_clean = 1.0 - float(front.get("front_kept_dyn_ratio", 0.0))
    pnp_sr = float(
        cand.get("qcand_pnp_success_rate", cand.get("cand_pnp_success_rate", cand.get("pnp_success_rate", 0.0)))
    )
    parallax_raw = cand.get("qcand_parallax_px", cand.get("cand_parallax_px", cand.get("parallax_px_candidate")))
    tri_points_raw = cand.get("qcand_tri_points", cand.get("cand_tri_points", cand.get("tri_points_candidate")))
    parallax_n = safe_norm_clip(
        parallax_raw,
        float(sfm_cfg.get("pivot_min_parallax", 5.0)),
        float(sfm_cfg.get("pivot_max_parallax", 80.0)),
    )
    tri_hi = max(tri_min_valid * 4.0, tri_min_valid + 40.0)
    tri_points_n = safe_norm_clip(tri_points_raw, tri_min_valid, tri_hi)
    cand_viable_ratio = float(np.clip(cand.get("cand_viable_ratio", cand.get("cand_ok_ratio", 0.0)), 0.0, 1.0))
    cand_model_purity = float(
        np.clip(cand.get("cand_model_purity", cand.get("cand_model_consensus", 0.0)), 0.0, 1.0)
    )
    cand_geom_rank_mean = float(np.clip(cand.get("cand_geom_rank_mean", 0.0), 0.0, 1.0))
    cand_pareto_consistent = float(np.clip(cand.get("cand_pareto_consistent", 0.0), 0.0, 1.0))

    pre_weights_raw = {
        "p_static": float(q_cfg.get("w_pre_p_static", q_cfg.get("w_p_static", 0.16))),
        "coverage": float(q_cfg.get("w_pre_coverage", q_cfg.get("w_coverage", 0.10))),
        "entropy": float(q_cfg.get("w_pre_entropy", q_cfg.get("w_entropy", 0.08))),
        "band_clean": float(q_cfg.get("w_pre_band_clean", q_cfg.get("w_band_clean", 0.08))),
        "dyn_clean": float(q_cfg.get("w_pre_dyn_clean", q_cfg.get("w_dyn_clean", 0.08))),
        "pnp_success": float(q_cfg.get("w_pre_pnp_success", q_cfg.get("w_pnp_success", 0.18))),
        "parallax": float(q_cfg.get("w_pre_parallax", q_cfg.get("w_parallax", 0.12))),
        "tri_points": float(q_cfg.get("w_pre_tri_points", q_cfg.get("w_tri_points", 0.08))),
        "cand_model_purity": float(
            q_cfg.get("w_pre_cand_model_purity", q_cfg.get("w_cand_model_purity", q_cfg.get("w_cand_consensus", 0.02)))
        ),
        "cand_geom_rank_mean": float(q_cfg.get("w_pre_cand_geom_rank_mean", q_cfg.get("w_cand_geom_rank_mean", 0.04))),
        "cand_pareto_consistent": float(
            q_cfg.get(
                "w_pre_cand_pareto_consistent",
                q_cfg.get("w_cand_pareto_consistent", q_cfg.get("w_cand_gap", 0.02)),
            )
        ),
    }
    pre_weights = _normalized_weights(pre_weights_raw)
    components = {
        "p_static": p_static,
        "coverage": coverage,
        "entropy": entropy,
        "band_clean": band_clean,
        "dyn_clean": dyn_clean,
        "pnp_success": pnp_sr,
        "parallax": parallax_n,
        "tri_points": tri_points_n,
        "cand_model_purity": cand_model_purity,
        "cand_geom_rank_mean": cand_geom_rank_mean,
        "cand_pareto_consistent": cand_pareto_consistent,
    }
    q_pre = _score_from_components(components, pre_weights)
    thresholds = _get_q_thresholds(cfg)
    out = {f"q_pre_{k}": float(v) for k, v in components.items()}
    out["Q_pre"] = float(q_pre)
    out["cand_viable_ratio"] = cand_viable_ratio
    out["q_pre_accept_threshold"] = thresholds["q_pre_accept_threshold"]
    out["q_pre_delay_threshold"] = thresholds["q_pre_delay_threshold"]
    out["diag_front_p_static"] = p_static
    out["diag_front_coverage_ratio"] = coverage
    out["diag_front_grid_entropy"] = entropy
    out["diag_front_band_clean"] = band_clean
    out["diag_front_dyn_clean"] = dyn_clean
    out["diag_cand_parallax_px"] = 0.0 if parallax_raw is None else float(parallax_raw)
    out["diag_cand_tri_points"] = 0.0 if tri_points_raw is None else float(tri_points_raw)
    out["diag_cand_pnp_success_rate"] = pnp_sr
    return out


def build_q_post_metrics(cfg, q_pre_metrics, geom=None, ok=None):
    sfm_cfg = cfg.get("sfm", {})
    q_cfg = cfg.get("init_quality", {})
    geom = geom or {}

    reproj_med_good = float(q_cfg.get("reproj_med_good_px", 0.5))
    reproj_med_bad = float(q_cfg.get("reproj_med_bad_px", max(float(sfm_cfg.get("tri_reproj_thresh", 3.0)), 3.0)))
    reproj_p90_good = float(q_cfg.get("reproj_p90_good_px", 1.0))
    reproj_p90_bad = float(q_cfg.get("reproj_p90_bad_px", max(2.0 * float(sfm_cfg.get("tri_reproj_thresh", 3.0)), 6.0)))
    cheirality_min = float(q_cfg.get("cheirality_min", 0.7))
    triangulation_ratio_min = float(q_cfg.get("triangulation_ratio_min", 0.25))
    triangulation_ratio_good = float(q_cfg.get("triangulation_ratio_good", 0.75))

    reproj_med_px = geom.get("reproj_med_px")
    reproj_p90_px = geom.get("reproj_p90_px")
    cheirality_ratio = geom.get("cheirality_ratio")
    triangulation_ratio = geom.get("triangulation_ratio")

    reproj_med_n = safe_norm_clip_inv(reproj_med_px, reproj_med_good, reproj_med_bad, missing="zero")
    reproj_p90_n = safe_norm_clip_inv(reproj_p90_px, reproj_p90_good, reproj_p90_bad, missing="zero")
    cheirality_n = safe_norm_clip(cheirality_ratio, cheirality_min, 1.0, missing="zero")
    triangulation_ratio_n = safe_norm_clip(
        triangulation_ratio, triangulation_ratio_min, triangulation_ratio_good, missing="zero"
    )

    post_weights_raw = {
        "q_pre": float(q_cfg.get("w_post_q_pre", 0.45)),
        "reproj_med": float(q_cfg.get("w_post_reproj_med", q_cfg.get("w_reproj_med", 0.20))),
        "reproj_p90": float(q_cfg.get("w_post_reproj_p90", q_cfg.get("w_reproj_p90", 0.10))),
        "cheirality": float(q_cfg.get("w_post_cheirality", q_cfg.get("w_cheirality", 0.10))),
        "triangulation_ratio": float(
            q_cfg.get("w_post_triangulation_ratio", q_cfg.get("w_triangulation_ratio", 0.15))
        ),
    }
    post_weights = _normalized_weights(post_weights_raw)
    post_geom_weights = _normalized_weights({k: v for k, v in post_weights_raw.items() if k != "q_pre"})
    components = {
        "q_pre": float(q_pre_metrics.get("Q_pre", 0.0)),
        "reproj_med": reproj_med_n,
        "reproj_p90": reproj_p90_n,
        "cheirality": cheirality_n,
        "triangulation_ratio": triangulation_ratio_n,
    }
    q_post = _score_from_components(components, post_weights)
    q_post_geom_only = _score_from_components({k: v for k, v in components.items() if k != "q_pre"}, post_geom_weights)
    thresholds = _get_q_thresholds(cfg)
    geom_quality = evaluate_post_geom_quality(cfg, geom)
    out = {f"q_post_{k}": float(v) for k, v in components.items()}
    out["Q_post"] = float(q_post)
    out["Q_post_geom_only"] = float(q_post_geom_only)
    out["Q_post_semantics"] = "mixed_prepost_score"
    out["Q_post_geom_only_semantics"] = "posterior_geometry_only_score"
    out["q_post_accept_threshold"] = thresholds["q_post_accept_threshold"]
    out["q_post_delay_threshold"] = thresholds["q_post_delay_threshold"]
    out["geom_reproj_med_px"] = reproj_med_px
    out["geom_reproj_p90_px"] = reproj_p90_px
    out["geom_cheirality_ratio"] = cheirality_ratio
    out["geom_triangulation_ratio"] = triangulation_ratio
    out["geom_init_ok"] = bool(ok) if ok is not None else None
    out.update(geom_quality)
    return out


def _pick_first_reason(candidates: list[tuple[bool, str]], fallback: str) -> str:
    for cond, reason in candidates:
        if cond:
            return reason
    return fallback


def decide_gate(cfg, q_pre_metrics, q_post_metrics=None, ok=None):
    thresholds = _get_q_thresholds(cfg)
    q_pre = float(q_pre_metrics.get("Q_pre", 0.0))
    q_post = None if q_post_metrics is None else float(q_post_metrics.get("Q_post", 0.0))

    if q_pre >= thresholds["q_pre_accept_threshold"]:
        gate_pre = "pre_accept"
    elif q_pre >= thresholds["q_pre_delay_threshold"]:
        gate_pre = "pre_delay"
    else:
        gate_pre = "pre_reset"

    gate_pre_reason = _pick_first_reason(
        [
            (float(q_pre_metrics.get("q_pre_p_static", 0.0)) < 0.35, "low_static_support"),
            (float(q_pre_metrics.get("q_pre_coverage", 0.0)) < 0.35, "low_coverage"),
            (float(q_pre_metrics.get("q_pre_dyn_clean", 0.0)) < 0.35, "dynamic_contamination"),
            (float(q_pre_metrics.get("q_pre_band_clean", 0.0)) < 0.35, "boundary_contamination"),
            (float(q_pre_metrics.get("q_pre_parallax", 0.0)) < 0.25, "low_parallax"),
            (float(q_pre_metrics.get("q_pre_tri_points", 0.0)) < 0.25, "low_tri_points"),
            (float(q_pre_metrics.get("q_pre_pnp_success", 0.0)) < 0.35, "low_pnp_success"),
            (float(q_pre_metrics.get("q_pre_cand_pareto_consistent", 1.0)) < 0.5, "candidate_dominance_conflict"),
            (float(q_pre_metrics.get("q_pre_cand_model_purity", 1.0)) < 0.5, "candidate_model_ambiguity"),
            (float(q_pre_metrics.get("q_pre_cand_geom_rank_mean", 1.0)) < 0.6, "candidate_ambiguity"),
        ],
        "ready_for_init" if gate_pre == "pre_accept" else "marginal_pre_quality",
    )

    if q_post_metrics is None:
        gate_post = None
        gate_post_reason = None
    else:
        if thresholds["force_success_only"] and ok is False:
            gate_post = "reset"
        elif ok is False:
            gate_post = "delay" if q_post is not None and q_post >= thresholds["q_post_delay_threshold"] else "reset"
        elif q_post is not None and q_post >= thresholds["q_post_accept_threshold"]:
            gate_post = "accept"
        elif q_post is not None and q_post >= thresholds["q_post_delay_threshold"]:
            gate_post = "delay"
        else:
            gate_post = "reset"

        gate_post_reason = _pick_first_reason(
            [
                (ok is False, "solver_failed"),
                (float(q_post_metrics.get("q_post_reproj_med", 1.0)) < 0.35, "high_reprojection"),
                (float(q_post_metrics.get("q_post_reproj_p90", 1.0)) < 0.35, "high_tail_reprojection"),
                (float(q_post_metrics.get("q_post_cheirality", 1.0)) < 0.5, "bad_depth_sign"),
                (float(q_post_metrics.get("q_post_triangulation_ratio", 1.0)) < 0.35, "weak_triangulation"),
                (gate_pre == "pre_accept" and gate_post != "accept", "pre_good_but_post_bad"),
            ],
            "accepted_quality" if gate_post == "accept" else "marginal_post_quality",
        )

    return {
        "gate_pre": gate_pre,
        "gate_pre_reason": gate_pre_reason,
        "gate_post": gate_post,
        "gate_post_reason": gate_post_reason,
        "gate_reason": gate_post_reason if gate_post_reason is not None else gate_pre_reason,
    }


def build_q_metrics(cfg, front, cand, geom=None, ok=None):
    """
    Compatibility wrapper: merged pre/post/gate outputs.
    """
    q_pre_metrics = build_q_pre_metrics(cfg, front, cand)
    q_post_metrics = build_q_post_metrics(cfg, q_pre_metrics, geom=geom, ok=ok)
    gate_metrics = decide_gate(cfg, q_pre_metrics, q_post_metrics, ok=ok)
    out = {}
    out.update(q_pre_metrics)
    out.update(q_post_metrics)
    out.update(gate_metrics)
    out["Q"] = float(q_post_metrics.get("Q_post", 0.0))
    out["gate_decision"] = gate_metrics.get("gate_post")
    out["q_accept_threshold"] = q_post_metrics.get("q_post_accept_threshold")
    out["q_delay_threshold"] = q_post_metrics.get("q_post_delay_threshold")
    return out


def compute_binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    pred = scores >= float(threshold)
    tp = int(np.sum((pred == 1) & (labels == 1)))
    tn = int(np.sum((pred == 0) & (labels == 0)))
    fp = int(np.sum((pred == 1) & (labels == 0)))
    fn = int(np.sum((pred == 0) & (labels == 1)))
    precision = float(tp / max(1, tp + fp))
    recall = float(tp / max(1, tp + fn))
    tpr = recall
    fpr = float(fp / max(1, fp + tn))
    acc = float((tp + tn) / max(1, labels.size))
    return {
        "threshold": float(threshold),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "tpr": tpr,
        "fpr": fpr,
        "accuracy": acc,
    }


def roc_points(labels: np.ndarray, scores: np.ndarray, num_thresholds: int = 201) -> list[dict]:
    if labels.size == 0:
        return []
    thresholds = np.linspace(0.0, 1.0, num_thresholds)
    return [compute_binary_metrics(labels, scores, thr) for thr in thresholds]


def auc_from_roc(points: list[dict]) -> float:
    if not points:
        return 0.0
    ordered = sorted(points, key=lambda x: (x["fpr"], x["tpr"]))
    xs = np.asarray([p["fpr"] for p in ordered], dtype=np.float64)
    ys = np.asarray([p["tpr"] for p in ordered], dtype=np.float64)
    return float(np.trapz(ys, xs))
