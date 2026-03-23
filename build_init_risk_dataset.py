#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import random
import re


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = None
for p in [THIS_FILE.parent, *THIS_FILE.parent.parents]:
    if (p / "configs").is_dir() and ((p / "pipelines").is_dir() or (p / " pipelines").is_dir()):
        PROJECT_ROOT = p
        break
if PROJECT_ROOT is None:
    raise RuntimeError(f"Cannot locate project root from {THIS_FILE}")


def safe_float(v, default=None):
    if v is None:
        return default
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    if math.isnan(x):
        return default
    return x


def safe_int(v, default=None):
    if v is None:
        return default
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def safe_bool(v, default=None):
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y"):
        return True
    if s in ("0", "false", "f", "no", "n", ""):
        return False
    return default


def load_csv_rows(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_v2_min_labels(path: Path) -> tuple[dict[str, dict], dict]:
    rows = load_csv_rows(path)
    by_attempt = {}
    source_counts = {}
    labeled_rows = 0
    for row in rows:
        attempt_uid = str(row.get("attempt_uid", ""))
        if not attempt_uid:
            continue
        by_attempt[attempt_uid] = row
        source = str(row.get("Y_bad_v2_min_default_source", row.get("Y_bad_v2_min_source", "")))
        source_counts[source] = source_counts.get(source, 0) + 1
        if str(row.get("Y_bad_v2_min_default", row.get("Y_bad_v2_min", ""))) in ("0", "1"):
            labeled_rows += 1
    return by_attempt, {
        "input_rows": len(rows),
        "matched_attempt_rows": len(by_attempt),
        "labeled_rows": labeled_rows,
        "source_counts": source_counts,
    }


def find_companion_json(csv_path: Path, filename: str):
    candidate = csv_path.parent / filename
    return candidate if candidate.exists() else None


def load_protocol_sidecar(csv_path: Path) -> dict:
    protocol_path = find_companion_json(csv_path, "experiment_protocol.json")
    if protocol_path is None:
        return {}
    try:
        data = load_json(protocol_path)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def has_value(row: dict, key: str) -> bool:
    if key not in row:
        return False
    value = row.get(key)
    return value not in (None, "")


def infer_variant_tag(sample_type: str, csv_path: Path) -> str:
    if sample_type == "step10b":
        folder = csv_path.parent.name
        m = re.match(r"^win\d+_s\d+_(.+)$", folder)
        if m:
            return m.group(1)
        return folder
    if sample_type == "step11":
        return csv_path.parent.name
    return "unknown"


def infer_dataset_group(csv_path: Path) -> str:
    parts = list(csv_path.parts)
    for i, part in enumerate(parts):
        if part == "VIODE" and i + 1 < len(parts):
            return parts[i + 1]
    return ""


def build_ids(sample_type: str, sequence: str, variant_tag: str, window_id: int) -> dict:
    attempt_uid = f"{sequence}::{variant_tag}::{int(window_id):06d}"
    sample_uid = f"{sample_type}::{attempt_uid}"
    return {
        "sample_uid": sample_uid,
        "attempt_uid": attempt_uid,
        "sequence_group_key": str(sequence),
    }


def infer_dynamic_level(sequence: str) -> str:
    seq = str(sequence).lower()
    if "_none" in seq or seq.endswith("none"):
        return "none"
    if "_low" in seq or seq.endswith("low"):
        return "low"
    if "_mid" in seq or seq.endswith("mid"):
        return "mid"
    if "_high" in seq or seq.endswith("high"):
        return "high"
    return "unknown"


def protocol_provenance(protocol: dict) -> str:
    return "native" if isinstance(protocol, dict) and len(protocol) > 0 else "missing"


def y_bad_provenance(y_bad_source: str) -> str:
    if y_bad_source in ("success_strict", "ba_success_strict"):
        return "native"
    if y_bad_source in ("legacy_success_plus_geom", "ba_ok_plus_geom"):
        return "inferred"
    return "fallback"


def summarize_provenance(rows: list[dict], keys: list[str]) -> dict:
    out = {}
    for key in keys:
        prov_key = f"prov_{key}"
        counts = {}
        for row in rows:
            val = row.get(prov_key, "missing")
            counts[val] = counts.get(val, 0) + 1
        out[key] = counts
    return out


def summarize_provenance_with_ratio(rows: list[dict], keys: list[str]) -> dict:
    out = {}
    n = max(1, len(rows))
    counts_by_key = summarize_provenance(rows, keys)
    for key, counts in counts_by_key.items():
        out[key] = {
            "counts": counts,
            "ratios": {name: float(val / n) for name, val in counts.items()},
        }
    return out


def summarize_missing_rates(rows: list[dict], keys: list[str]) -> dict:
    out = {}
    n = max(1, len(rows))
    for key in keys:
        missing = 0
        for row in rows:
            val = row.get(key)
            if val in (None, ""):
                missing += 1
        out[key] = float(missing / n)
    return out


def assign_splits_by_sequence(rows: list[dict], seed: int, ratios: tuple[float, float, float]) -> tuple[list[dict], dict]:
    sequences = sorted({str(row.get("sequence_group_key", "")) for row in rows if str(row.get("sequence_group_key", ""))})
    rng = random.Random(int(seed))
    rng.shuffle(sequences)
    n = len(sequences)

    if n == 0:
        return rows, {"split_mode": "strict_by_sequence", "warnings": ["no_sequence_groups_found"], "sequence_to_split": {}}

    if n == 1:
        sequence_to_split = {sequences[0]: "test"}
        warnings = ["only_one_sequence_available_all_assigned_to_test"]
    elif n == 2:
        sequence_to_split = {sequences[0]: "train", sequences[1]: "test"}
        warnings = ["only_two_sequences_available_val_split_empty"]
    else:
        train_ratio, val_ratio, test_ratio = ratios
        train_n = max(1, int(round(train_ratio * n)))
        val_n = max(1, int(round(val_ratio * n)))
        if train_n + val_n >= n:
            val_n = max(1, n - train_n - 1)
        test_n = max(1, n - train_n - val_n)
        if train_n + val_n + test_n > n:
            overflow = train_n + val_n + test_n - n
            train_n = max(1, train_n - overflow)
        while train_n + val_n + test_n < n:
            test_n += 1
        train_seq = sequences[:train_n]
        val_seq = sequences[train_n:train_n + val_n]
        test_seq = sequences[train_n + val_n:]
        sequence_to_split = {s: "train" for s in train_seq}
        sequence_to_split.update({s: "val" for s in val_seq})
        sequence_to_split.update({s: "test" for s in test_seq})
        warnings = []

    for row in rows:
        row["group_split_key"] = str(row.get("sequence_group_key", ""))
        row["dataset_row_split"] = sequence_to_split.get(row["group_split_key"], "unsplit")

    split_counts = {}
    split_bad = {}
    for row in rows:
        split = row.get("dataset_row_split", "unsplit")
        split_counts[split] = split_counts.get(split, 0) + 1
        split_bad.setdefault(split, []).append(int(row.get("Y_bad_v1", 0)))

    split_stats = {}
    for split, count in split_counts.items():
        ys = split_bad.get(split, [])
        split_stats[split] = {
            "num_rows": int(count),
            "bad_ratio": float(sum(ys) / max(1, len(ys))),
        }

    return rows, {
        "split_mode": "strict_by_sequence",
        "seed": int(seed),
        "ratios": {"train": float(ratios[0]), "val": float(ratios[1]), "test": float(ratios[2])},
        "group_key": "sequence_group_key",
        "attempt_uid_not_cross_split": True,
        "sequence_not_cross_split": True,
        "sequence_to_split": sequence_to_split,
        "split_stats": split_stats,
        "warnings": warnings,
    }


def infer_step10b_schema(row: dict) -> str:
    if "success_strict" in row or "solver_ok" in row or "geom_ok" in row:
        return "step10b_v2"
    if "Q_pre" in row or "Q_post" in row or "gate_pre" in row:
        return "step10b_mid"
    return "step10b_legacy"


def infer_step11_schema(row: dict) -> str:
    if "ba_success_strict" in row or "ba_solver_ok" in row or "ba_Q_post_geom_only" in row:
        return "step11_v2"
    if "ba_Q_pre" in row or "ba_Q_post" in row or "ba_gate_pre" in row:
        return "step11_mid"
    return "step11_legacy"


def step10b_y_bad_v1(row: dict) -> tuple[int, str]:
    success_strict = safe_int(row.get("success_strict"))
    if success_strict is not None:
        return 1 - int(success_strict > 0), "success_strict"

    success = safe_int(row.get("success"))
    reproj_med = safe_float(row.get("reproj_med_px"))
    reproj_p90 = safe_float(row.get("reproj_p90_px"))
    chei = safe_float(row.get("cheirality_ratio"))
    tri_ratio = safe_float(row.get("triangulation_ratio"))

    geom_bad = False
    if reproj_med is not None and reproj_med > 3.0:
        geom_bad = True
    if reproj_p90 is not None and reproj_p90 > 6.0:
        geom_bad = True
    if chei is not None and chei < 0.7:
        geom_bad = True
    if tri_ratio is not None and tri_ratio < 0.25:
        geom_bad = True
    if success is not None:
        return int((success <= 0) or geom_bad), "legacy_success_plus_geom"
    return 1, "missing_success"


def step11_y_bad_v1(row: dict) -> tuple[int, str]:
    ba_success_strict = safe_bool(row.get("ba_success_strict"))
    if ba_success_strict is not None:
        return int(not ba_success_strict), "ba_success_strict"

    ba_ok = safe_bool(row.get("ba_ok"))
    reproj_med = safe_float(row.get("ba_reproj_med_px"))
    reproj_p90 = safe_float(row.get("ba_reproj_p90_px"))
    chei = safe_float(row.get("ba_cheirality_ratio"))
    tri_ratio = safe_float(row.get("triangulation_ratio"))
    gt_rot = safe_float(row.get("gt_rot_med_deg"))
    gt_trans = safe_float(row.get("gt_trans_dir_med_deg"))

    geom_bad = False
    if reproj_med is not None and reproj_med > 3.0:
        geom_bad = True
    if reproj_p90 is not None and reproj_p90 > 6.0:
        geom_bad = True
    if chei is not None and chei < 0.7:
        geom_bad = True
    if tri_ratio is not None and tri_ratio < 0.25:
        geom_bad = True
    if gt_rot is not None and gt_rot > 5.0:
        geom_bad = True
    if gt_trans is not None and gt_trans > 10.0:
        geom_bad = True
    if ba_ok is not None:
        return int((not ba_ok) or geom_bad), "ba_ok_plus_geom"
    return 1, "missing_ba_ok"


def normalize_step10b_row(row: dict, csv_path: Path, protocol: dict) -> dict:
    y_bad, y_bad_source = step10b_y_bad_v1(row)
    schema_version = infer_step10b_schema(row)
    seq_name = csv_path.parents[3].name if len(csv_path.parents) >= 4 else ""
    dataset_group = infer_dataset_group(csv_path)
    sequence = f"{dataset_group}/{seq_name}" if dataset_group else seq_name
    window_id = safe_int(row.get("start"), -1)
    variant_tag = infer_variant_tag("step10b", csv_path)
    ids = build_ids("step10b", sequence, variant_tag, window_id)
    q_pre = safe_float(row.get("Q_pre"))
    q_post = safe_float(row.get("Q_post", row.get("Q")))
    q_post_geom_only = safe_float(row.get("Q_post_geom_only"))
    success_strict = safe_int(row.get("success_strict"), safe_int(row.get("success"), 0))
    out = {
        **ids,
        "sample_type": "step10b",
        "schema_version": schema_version,
        "source_csv": str(csv_path),
        "source_dir": str(csv_path.parent),
        "sequence": sequence,
        "sequence_dynamic_level": infer_dynamic_level(sequence),
        "variant_tag": variant_tag,
        "window_id": window_id,
        "protocol_dataset_split": protocol.get("dataset_split", "unknown"),
        "protocol_q_threshold_mode": protocol.get("q_threshold_mode", "unknown"),
        "protocol_threshold_set_id": protocol.get("threshold_set_id", "unknown"),
        "q_threshold_source": protocol.get("q_threshold_source", "unknown"),
        "Y_bad_v1": int(y_bad),
        "Y_bad_v1_source": y_bad_source,
        "prov_Y_bad_v1": y_bad_provenance(y_bad_source),
        "solver_ok": safe_int(row.get("solver_ok"), safe_int(row.get("success"), 0)),
        "geom_ok": safe_int(row.get("geom_ok"), None),
        "success_strict": success_strict,
        "Q_pre": q_pre,
        "Q_post": q_post,
        "Q_post_geom_only": q_post_geom_only,
        "prov_Q_pre": "native" if has_value(row, "Q_pre") and q_pre is not None else "missing",
        "prov_Q_post": (
            "native"
            if has_value(row, "Q_post") and q_post is not None
            else ("inferred" if has_value(row, "Q") and q_post is not None else "missing")
        ),
        "prov_Q_post_geom_only": "native" if has_value(row, "Q_post_geom_only") and q_post_geom_only is not None else "missing",
        "prov_success_strict": (
            "native"
            if has_value(row, "success_strict") and success_strict is not None
            else ("inferred" if has_value(row, "success") and success_strict is not None else "missing")
        ),
        "prov_experiment_protocol": protocol_provenance(protocol),
        "gate_pre": row.get("gate_pre", ""),
        "gate_post": row.get("gate_post", row.get("gate_decision", "")),
        "gate_pre_reason": row.get("gate_pre_reason", ""),
        "gate_post_reason": row.get("gate_post_reason", row.get("gate_reason", "")),
        "geom_failure_reason": row.get("geom_failure_reason", row.get("post_geom_failure_reason", "")),
        "cand_summary_missing": safe_int(row.get("cand_summary_missing"), 0),
        "front_p_static": safe_float(row.get("front_p_static")),
        "front_p_band": safe_float(row.get("front_p_band")),
        "front_coverage_ratio": safe_float(row.get("front_coverage_ratio")),
        "front_grid_entropy": safe_float(row.get("front_grid_entropy")),
        "front_kept_dyn_ratio": safe_float(row.get("front_kept_dyn_ratio")),
        "parallax_px_candidate": safe_float(row.get("parallax_px_candidate")),
        "tri_points_candidate": safe_float(row.get("tri_points", row.get("tri_points_candidate"))),
        "triangulation_ratio": safe_float(row.get("triangulation_ratio")),
        "pnp_success_rate": safe_float(row.get("pnp_success_rate")),
        "pnp_median_inliers": safe_float(row.get("pnp_median_inliers")),
        "reproj_med_px": safe_float(row.get("reproj_med_px")),
        "reproj_p90_px": safe_float(row.get("reproj_p90_px")),
        "cheirality_ratio": safe_float(row.get("cheirality_ratio")),
        "runtime_s": safe_float(row.get("time_sec")),
    }
    return out


def normalize_step11_row(row: dict, csv_path: Path, protocol: dict) -> dict:
    y_bad, y_bad_source = step11_y_bad_v1(row)
    schema_version = infer_step11_schema(row)
    seq_name = csv_path.parents[1].name if len(csv_path.parents) >= 2 else ""
    dataset_group = infer_dataset_group(csv_path)
    sequence = f"{dataset_group}/{seq_name}" if dataset_group else seq_name
    window_id = safe_int(row.get("start_frame_cfg"), -1)
    variant_tag = infer_variant_tag("step11", csv_path)
    ids = build_ids("step11", sequence, variant_tag, window_id)
    q_pre = safe_float(row.get("ba_Q_pre"))
    q_post = safe_float(row.get("ba_Q_post", row.get("ba_Q")))
    q_post_geom_only = safe_float(row.get("ba_Q_post_geom_only"))
    success_strict = safe_int(row.get("ba_success_strict"), safe_int(row.get("ba_ok"), 0))
    out = {
        **ids,
        "sample_type": "step11",
        "schema_version": schema_version,
        "source_csv": str(csv_path),
        "source_dir": str(csv_path.parent),
        "sequence": sequence,
        "sequence_dynamic_level": infer_dynamic_level(sequence),
        "variant_tag": variant_tag,
        "window_id": window_id,
        "protocol_dataset_split": protocol.get("dataset_split", "unknown"),
        "protocol_q_threshold_mode": protocol.get("q_threshold_mode", "unknown"),
        "protocol_threshold_set_id": protocol.get("threshold_set_id", "unknown"),
        "q_threshold_source": protocol.get("q_threshold_source", "unknown"),
        "Y_bad_v1": int(y_bad),
        "Y_bad_v1_source": y_bad_source,
        "prov_Y_bad_v1": y_bad_provenance(y_bad_source),
        "solver_ok": safe_int(row.get("ba_solver_ok"), safe_int(row.get("ba_ok"), 0)),
        "geom_ok": safe_int(row.get("ba_geom_ok"), None),
        "success_strict": success_strict,
        "Q_pre": q_pre,
        "Q_post": q_post,
        "Q_post_geom_only": q_post_geom_only,
        "prov_Q_pre": "native" if has_value(row, "ba_Q_pre") and q_pre is not None else "missing",
        "prov_Q_post": (
            "native"
            if has_value(row, "ba_Q_post") and q_post is not None
            else ("inferred" if has_value(row, "ba_Q") and q_post is not None else "missing")
        ),
        "prov_Q_post_geom_only": "native" if has_value(row, "ba_Q_post_geom_only") and q_post_geom_only is not None else "missing",
        "prov_success_strict": (
            "native"
            if has_value(row, "ba_success_strict") and success_strict is not None
            else ("inferred" if has_value(row, "ba_ok") and success_strict is not None else "missing")
        ),
        "prov_experiment_protocol": protocol_provenance(protocol),
        "gate_pre": row.get("ba_gate_pre", ""),
        "gate_post": row.get("ba_gate_post", row.get("ba_gate_decision", "")),
        "gate_pre_reason": row.get("ba_gate_pre_reason", ""),
        "gate_post_reason": row.get("ba_gate_post_reason", ""),
        "geom_failure_reason": row.get("ba_post_geom_failure_reason", ""),
        "cand_summary_missing": safe_int(row.get("ba_cand_summary_missing"), 0),
        "front_p_static": safe_float(row.get("front_p_static")),
        "front_p_band": safe_float(row.get("front_p_band")),
        "front_coverage_ratio": safe_float(row.get("front_coverage_ratio")),
        "front_grid_entropy": safe_float(row.get("front_grid_entropy")),
        "front_kept_dyn_ratio": safe_float(row.get("front_kept_dyn_ratio")),
        "parallax_px_candidate": safe_float(row.get("parallax_px_candidate")),
        "tri_points_candidate": safe_float(row.get("tri_points_candidate")),
        "triangulation_ratio": safe_float(row.get("triangulation_ratio")),
        "pnp_success_rate": safe_float(row.get("pnp_eval_success_rate")),
        "pnp_median_inliers": safe_float(row.get("pnp_eval_median_inliers")),
        "reproj_med_px": safe_float(row.get("ba_reproj_med_px")),
        "reproj_p90_px": safe_float(row.get("ba_reproj_p90_px")),
        "cheirality_ratio": safe_float(row.get("ba_cheirality_ratio")),
        "gt_rot_med_deg": safe_float(row.get("gt_rot_med_deg")),
        "gt_trans_dir_med_deg": safe_float(row.get("gt_trans_dir_med_deg")),
        "runtime_s": safe_float(row.get("ba_runtime_s")),
    }
    return out


def collect_input_files(input_roots: list[Path]) -> tuple[list[Path], list[Path]]:
    step10b_csvs = []
    step11_csvs = []
    for root in input_roots:
        if root.is_file():
            if root.name == "init_success_summary.csv":
                step10b_csvs.append(root)
            elif root.name == "summary.csv":
                step11_csvs.append(root)
            continue
        # Prefer bounded glob patterns under the known result layout to avoid
        # recursively walking huge mask/image trees when running multi-seed rebuilds.
        bounded_step10 = sorted(root.glob("*/eval/sfm_init_success_rate/*/init_success_summary.csv"))
        bounded_step11 = sorted(root.glob("*/eval/sfm_init_quality_v2/*/*/summary.csv"))
        if bounded_step10 or bounded_step11:
            step10b_csvs.extend(bounded_step10)
            step11_csvs.extend(bounded_step11)
            continue
        # Fallback for ad-hoc roots that do not follow the usual step1_11 layout.
        step10b_csvs.extend(sorted(root.rglob("init_success_summary.csv")))
        step11_csvs.extend(sorted(root.rglob("sfm_init_quality_v2/*/*/summary.csv")))
    dedup10 = sorted({p.resolve() for p in step10b_csvs})
    dedup11 = sorted({p.resolve() for p in step11_csvs})
    return dedup10, dedup11


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = set()
    for row in rows:
        keys |= set(row.keys())
    fieldnames = sorted(keys)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def write_split_csvs(out_dir: Path, rows: list[dict]):
    by_split = {}
    for row in rows:
        split = str(row.get("dataset_row_split", "unsplit"))
        by_split.setdefault(split, []).append(row)
    for split, split_rows in by_split.items():
        write_csv(out_dir / f"risk_dataset_{split}.csv", split_rows)


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_selected_csv(path: Path, rows: list[dict], columns: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in columns})


def task_claim_policy(task_name: str) -> dict:
    base_limit_flags = [
        "small_sample",
        "single_protocol_family",
        "dynamic_level_split_coupling",
        "proxy_label_only",
        "no_system_level_closed_loop_evidence",
    ]
    if task_name == "pre":
        return {
            "task_scope": "pre_accept_risk_feasibility",
            "allowed_claims": [
                "Current protocol supports a learnable pre-risk signal under frozen manifests and strict-by-sequence split.",
                "Under the current audit dataset, candidate-stage observability provides the dominant discriminative signal for pre-risk prediction.",
                "Front-end static support can be described as upstream context or constraint, not the direct dominant predictor of pre-risk.",
            ],
            "forbidden_claims": [
                "Do not claim system-level failure prediction or system-level gate benefit from Y_bad_v1-based pre models.",
                "Do not claim cross-scene or cross-dynamic-level generalization from the current 4-sequence split.",
                "Do not claim front-end static reliability directly dominates pre-risk under the current evidence.",
                "Do not claim calibrated probability outputs are ready for deployment-level decision making.",
            ],
            "proxy_bias_notes": [
                "Y_bad_v1 emphasizes initialization-stage solver/geometry failures and may underrepresent failures that emerge only after accept over a future horizon.",
                "Y_bad_v1 is more sensitive to geometry-quality degradation than to delayed system-level tracking collapse.",
                "A stronger label such as Y_bad_v2 is required before claiming system-level gating benefit.",
            ],
            "experimental_limit_flags": base_limit_flags,
        }
    if task_name == "post":
        return {
            "task_scope": "post_accept_risk_structure_reassessment",
            "allowed_claims": [
                "Current protocol supports a learnable post-risk or acceptor-style signal under frozen manifests and strict-by-sequence split.",
                "Under the current audit dataset, the post task should be treated as a learning-based acceptor baseline whose internal feature structure is still under reassessment.",
                "Current evidence supports comparing geometry, candidate, and front blocks as competing or complementary sources rather than fixing the task as permanently geometry-dominated or candidate-dominated.",
            ],
            "forbidden_claims": [
                "Do not claim this post model is the primary innovation carrier of the whole method.",
                "Do not claim cross-scene or cross-dynamic-level generalization from the current 4-sequence split.",
                "Do not claim calibrated probability outputs are ready for deployment-level decision making.",
                "Do not claim system-level gate benefit from Y_bad_v1-based post models alone.",
                "Do not claim the current post structure has already converged to a final minimal sufficient feature set.",
                "Do not claim posterior geometry alone or candidate evidence alone is already established as the unique dominant mechanism under the current evolving evidence.",
            ],
            "proxy_bias_notes": [
                "Y_bad_v1 for post models still reflects initialization-level proxy bad events rather than a full downstream system failure horizon.",
                "Current post labels emphasize posterior geometry adequacy and may miss later failure modes beyond the initialization window.",
                "A stronger label such as Y_bad_v2 is required before claiming system-level acceptance robustness.",
            ],
            "experimental_limit_flags": base_limit_flags,
        }
    raise ValueError(f"Unsupported task_name: {task_name}")


def task_claim_policy_post_v2_default() -> dict:
    return {
        "task_scope": "post_accept_horizon_instability_feasibility",
        "allowed_claims": [
            "Current protocol supports a learnable post-accept short-horizon instability signal on accepted windows with sufficient future horizon.",
            "Y_bad_v2_min_default is closer to acceptance stability semantics than Y_bad_v1, but remains a proxy rather than a full downstream failure truth.",
            "Current post_v2 evidence supports continuing structure reassessment and minimal sufficient feature-set analysis rather than fixing a final dominant block narrative.",
        ],
        "forbidden_claims": [
            "Do not claim system-level reset/reinit/tracking-failure prediction from Y_bad_v2_min_default alone.",
            "Do not expose Y_bad_v2_min_default as a validated pre-task label.",
            "Do not claim cross-scene or cross-dynamic-level generalization from the current 4-sequence split.",
            "Do not claim calibrated probability outputs are ready for deployment-level decision making.",
            "Do not claim the current post_v2 full feature mixing is already the preferred stable formulation.",
            "Do not claim the current post_v2 task has already converged to a purely geometry-dominated or purely candidate-dominated mechanism.",
        ],
        "proxy_bias_notes": [
            "Y_bad_v2_min_default is defined only on post-accept rows with sufficient short future horizon.",
            "Y_bad_v2_min_default emphasizes short-horizon instability after acceptance and may still miss delayed downstream failures beyond the chosen horizon.",
            "Current proxy remains closer to post-accept instability semantics than to full system-level closed-loop failure truth.",
        ],
        "experimental_limit_flags": [
            "small_sample",
            "single_protocol_family",
            "dynamic_level_split_coupling",
            "accepted_window_subset_only",
            "proxy_label_only",
            "no_system_level_closed_loop_evidence",
        ],
    }


def main():
    ap = argparse.ArgumentParser(description="Build a unified initialization-risk dataset from step10b/step11 outputs.")
    ap.add_argument(
        "--input_root",
        action="append",
        default=[],
        help="Repeatable input root. Can be a result directory or a specific csv file.",
    )
    ap.add_argument(
        "--out_dir",
        default=str(PROJECT_ROOT / "results" / "risk_dataset_v1"),
        help="Output directory for risk_dataset.csv and dataset_manifest.json",
    )
    ap.add_argument(
        "--split_mode",
        default="strict_by_sequence",
        choices=["strict_by_sequence"],
        help="Split strategy for formal dataset protocol.",
    )
    ap.add_argument(
        "--split_ratios",
        default="0.70,0.15,0.15",
        help="Comma-separated train,val,test ratios for group-aware split.",
    )
    ap.add_argument("--split_seed", type=int, default=20260312)
    ap.add_argument(
        "--write_split_csvs",
        action="store_true",
        help="Write risk_dataset_train/val/test.csv after audit. Disabled by default for audit-first workflow.",
    )
    ap.add_argument(
        "--write_task_split_csvs",
        action="store_true",
        help="Write task-specific pre/post train/val/test csv files and manifests.",
    )
    ap.add_argument(
        "--y_bad_v2_min_labels_csv",
        default="",
        help="Optional Y_bad_v2_min_default labels csv. If provided, export a post-only task package with this label.",
    )
    args = ap.parse_args()

    input_roots = [Path(p).expanduser().resolve() for p in args.input_root] if args.input_root else [
        Path("/mnt/g/Result/VIODE/city_day/stageB_full_compare/step1_11_cam0").resolve()
    ]
    step10b_csvs, step11_csvs = collect_input_files(input_roots)

    rows = []
    for csv_path in step10b_csvs:
        protocol = load_protocol_sidecar(csv_path)
        for row in load_csv_rows(csv_path):
            rows.append(normalize_step10b_row(row, csv_path, protocol))

    for csv_path in step11_csvs:
        protocol = load_protocol_sidecar(csv_path)
        for row in load_csv_rows(csv_path):
            rows.append(normalize_step11_row(row, csv_path, protocol))

    v2_labels_by_attempt = {}
    v2_coverage_summary = {}
    if args.y_bad_v2_min_labels_csv:
        v2_labels_path = Path(args.y_bad_v2_min_labels_csv).expanduser().resolve()
        v2_labels_by_attempt, v2_coverage_summary = load_v2_min_labels(v2_labels_path)
        for row in rows:
            row["Y_bad_v2_min_default"] = ""
            row["Y_bad_v2_min_default_source"] = ""
            row["prov_Y_bad_v2_min_default"] = "missing"
            row["Y_bad_v2_min_default_trigger"] = ""
            row["Y_bad_v2_min_default_trigger_window_id"] = ""
            row["Y_bad_v2_min_default_horizon_windows"] = ""
            row["Y_bad_v2_min_default_label_version"] = "Y_bad_v2_min_default"
            row["Y_bad_v2_min_default_label_scope"] = "post_accept_short_horizon_instability_proxy"
            if row.get("sample_type") != "step11":
                continue
            v2 = v2_labels_by_attempt.get(str(row.get("attempt_uid", "")))
            if not v2:
                continue
            row["Y_bad_v2_min_default"] = v2.get("Y_bad_v2_min_default", v2.get("Y_bad_v2_min", ""))
            row["Y_bad_v2_min_default_source"] = v2.get("Y_bad_v2_min_default_source", v2.get("Y_bad_v2_min_source", ""))
            row["prov_Y_bad_v2_min_default"] = v2.get("prov_Y_bad_v2_min_default", v2.get("prov_Y_bad_v2_min", ""))
            row["Y_bad_v2_min_default_trigger"] = v2.get("Y_bad_v2_min_default_trigger", v2.get("Y_bad_v2_min_trigger", ""))
            row["Y_bad_v2_min_default_trigger_window_id"] = v2.get(
                "Y_bad_v2_min_default_trigger_window_id",
                v2.get("Y_bad_v2_min_trigger_window_id", ""),
            )
            row["Y_bad_v2_min_default_horizon_windows"] = v2.get(
                "Y_bad_v2_min_default_horizon_windows",
                v2.get("Y_bad_v2_min_horizon_windows", ""),
            )

    try:
        ratio_vals = [float(x.strip()) for x in str(args.split_ratios).split(",")]
        if len(ratio_vals) != 3:
            raise ValueError
    except ValueError as exc:
        raise ValueError(f"Invalid --split_ratios: {args.split_ratios}") from exc
    split_info = {}
    if args.split_mode == "strict_by_sequence":
        rows, split_info = assign_splits_by_sequence(rows, seed=args.split_seed, ratios=tuple(ratio_vals))

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_csv = out_dir / "risk_dataset.csv"
    write_csv(dataset_csv, rows)
    if args.write_split_csvs:
        write_split_csvs(out_dir, rows)

    schema_versions = {}
    sample_types = {}
    ybad_sources = {}
    for row in rows:
        schema_versions[row["schema_version"]] = schema_versions.get(row["schema_version"], 0) + 1
        sample_types[row["sample_type"]] = sample_types.get(row["sample_type"], 0) + 1
        ybad_sources[row["Y_bad_v1_source"]] = ybad_sources.get(row["Y_bad_v1_source"], 0) + 1

    provenance_keys = ["Q_pre", "Q_post", "Q_post_geom_only", "success_strict", "experiment_protocol", "Y_bad_v1"]
    missing_rate_keys = [
        "Q_pre",
        "Q_post",
        "Q_post_geom_only",
        "success_strict",
        "solver_ok",
        "geom_ok",
        "front_p_static",
        "parallax_px_candidate",
        "pnp_success_rate",
        "reproj_med_px",
        "cheirality_ratio",
        "protocol_dataset_split",
        "protocol_threshold_set_id",
    ]
    provenance_summary = summarize_provenance(rows, provenance_keys)
    provenance_audit = summarize_provenance_with_ratio(rows, provenance_keys)
    audit = {
        "num_rows": len(rows),
        "sample_types": sample_types,
        "schema_versions": schema_versions,
        "schema_version_counts": schema_versions,
        "Y_bad_v1_ratio": float(sum(int(row["Y_bad_v1"]) for row in rows) / max(1, len(rows))),
        "Y_bad_v1_sources": ybad_sources,
        "missing_rates": summarize_missing_rates(rows, missing_rate_keys),
        "provenance": provenance_summary,
        "field_provenance_summary": provenance_audit,
        "protocol_distribution": {
            "protocol_dataset_split": {},
            "protocol_q_threshold_mode": {},
            "protocol_threshold_set_id": {},
        },
        "unknown_protocol_ratio": float(
            sum(1 for row in rows if row.get("prov_experiment_protocol") != "native") / max(1, len(rows))
        ),
        "dynamic_level_distribution": {},
        "split_dynamic_level_distribution": {},
        "dynamic_level_to_split": {},
        "dynamic_level_split_coupling": {},
    }
    for row in rows:
        for key in ("protocol_dataset_split", "protocol_q_threshold_mode", "protocol_threshold_set_id"):
            val = str(row.get(key, "unknown"))
            audit["protocol_distribution"][key][val] = audit["protocol_distribution"][key].get(val, 0) + 1
        level = str(row.get("sequence_dynamic_level", "unknown"))
        split = str(row.get("dataset_row_split", "unsplit"))
        audit["dynamic_level_distribution"][level] = audit["dynamic_level_distribution"].get(level, 0) + 1
        audit["split_dynamic_level_distribution"].setdefault(split, {})
        audit["split_dynamic_level_distribution"][split][level] = (
            audit["split_dynamic_level_distribution"][split].get(level, 0) + 1
        )
        audit["dynamic_level_to_split"].setdefault(level, set()).add(split)
    audit["dynamic_level_to_split"] = {
        level: sorted(list(splits)) for level, splits in audit["dynamic_level_to_split"].items()
    }
    audit["dynamic_level_split_coupling"] = {
        level: {
            "num_splits": len(splits),
            "splits": splits,
            "is_strongly_coupled": len(splits) <= 1,
        }
        for level, splits in audit["dynamic_level_to_split"].items()
    }

    forbidden_feature_columns = [
        "Y_bad_v1",
        "Y_bad_v1_source",
        "prov_Y_bad_v1",
        "success_strict",
        "solver_ok",
        "geom_ok",
        "Q_pre",
        "Q_post",
        "Q_post_geom_only",
        "gate_pre",
        "gate_post",
        "gate_pre_reason",
        "gate_post_reason",
        "gt_rot_med_deg",
        "gt_trans_dir_med_deg",
    ]
    label_columns = ["Y_bad_v1"]
    posthoc_analysis_columns = [
        "Q_pre",
        "Q_post",
        "Q_post_geom_only",
        "gate_pre",
        "gate_post",
        "gate_pre_reason",
        "gate_post_reason",
        "geom_failure_reason",
        "gt_rot_med_deg",
        "gt_trans_dir_med_deg",
        "runtime_s",
    ]
    protocol_columns = [
        "protocol_dataset_split",
        "protocol_q_threshold_mode",
        "protocol_threshold_set_id",
        "q_threshold_source",
    ]
    data_quality_columns = [
        "cand_summary_missing",
        "prov_Q_pre",
        "prov_Q_post",
        "prov_Q_post_geom_only",
        "prov_success_strict",
        "prov_experiment_protocol",
        "prov_Y_bad_v1",
    ]
    if args.y_bad_v2_min_labels_csv:
        data_quality_columns.extend(["prov_Y_bad_v2_min_default"])
    metadata_columns = [
        "sample_uid",
        "attempt_uid",
        "group_split_key",
        "sequence_group_key",
        "sample_type",
        "schema_version",
        "source_csv",
        "source_dir",
        "sequence",
        "variant_tag",
        "window_id",
        "dataset_row_split",
    ]
    all_columns = sorted({k for row in rows for k in row.keys()}) if rows else []
    posthoc_analysis_columns = sorted(set(posthoc_analysis_columns + [
        "Y_bad_v2_min_default_source",
        "Y_bad_v2_min_default_trigger",
        "Y_bad_v2_min_default_trigger_window_id",
        "Y_bad_v2_min_default_horizon_windows",
    ]))
    protocol_columns = sorted(set(protocol_columns))
    data_quality_columns = sorted(set(data_quality_columns))
    metadata_columns = sorted(set(metadata_columns))

    pre_feature_columns = [
        "front_p_static",
        "front_p_band",
        "front_coverage_ratio",
        "front_grid_entropy",
        "front_kept_dyn_ratio",
        "parallax_px_candidate",
        "tri_points_candidate",
        "triangulation_ratio",
        "pnp_success_rate",
        "pnp_median_inliers",
    ]
    pre_feature_columns = [k for k in pre_feature_columns if k in all_columns]

    post_only_feature_columns = [
        "reproj_med_px",
        "reproj_p90_px",
        "cheirality_ratio",
    ]
    post_only_feature_columns = [k for k in post_only_feature_columns if k in all_columns]
    post_feature_columns = sorted(set(pre_feature_columns + post_only_feature_columns))

    categorical_feature_columns = []
    numeric_feature_columns = sorted(set(pre_feature_columns + post_only_feature_columns))

    forbidden_feature_columns = sorted(
        set(forbidden_feature_columns + metadata_columns + protocol_columns + posthoc_analysis_columns + data_quality_columns)
    )
    feature_columns = sorted(set(pre_feature_columns + post_only_feature_columns))
    pre_forbidden_columns = sorted(k for k in all_columns if k not in set(pre_feature_columns))
    post_forbidden_columns = sorted(k for k in all_columns if k not in set(post_feature_columns))

    keep_pre_feature_columns = [
        "front_p_static",
        "front_p_band",
        "front_coverage_ratio",
        "front_kept_dyn_ratio",
        "parallax_px_candidate",
        "tri_points_candidate",
        "pnp_success_rate",
    ]
    keep_pre_feature_columns = [k for k in keep_pre_feature_columns if k in pre_feature_columns]
    ablation_pre_feature_columns = [
        "front_grid_entropy",
        "triangulation_ratio",
        "pnp_median_inliers",
    ]
    ablation_pre_feature_columns = [k for k in ablation_pre_feature_columns if k in pre_feature_columns]
    keep_post_only_feature_columns = [k for k in post_only_feature_columns if k in all_columns]
    keep_post_feature_columns = sorted(set(keep_pre_feature_columns + keep_post_only_feature_columns))
    ablation_post_feature_columns = sorted(set(ablation_pre_feature_columns))

    feature_columns_path = out_dir / "feature_columns.json"
    forbidden_columns_path = out_dir / "forbidden_feature_columns.json"
    legality_board_path = out_dir / "feature_legality_board.json"
    write_json(
        feature_columns_path,
        {
            "feature_columns": feature_columns,
            "pre_feature_columns": pre_feature_columns,
            "post_feature_columns": post_feature_columns,
            "metadata_columns": metadata_columns,
            "protocol_columns": protocol_columns,
            "data_quality_columns": data_quality_columns,
            "posthoc_analysis_columns": posthoc_analysis_columns,
            "categorical_feature_columns": categorical_feature_columns,
            "numeric_feature_columns": numeric_feature_columns,
            "forbidden_feature_columns": forbidden_feature_columns,
            "pre_forbidden_columns": pre_forbidden_columns,
            "post_forbidden_columns": post_forbidden_columns,
        },
    )
    write_json(
        forbidden_columns_path,
        {
            "forbidden_feature_columns": forbidden_feature_columns,
            "pre_forbidden_columns": pre_forbidden_columns,
            "post_forbidden_columns": post_forbidden_columns,
        },
    )
    write_json(
        legality_board_path,
        {
            "keep_for_pre_model": keep_pre_feature_columns,
            "keep_for_post_model": keep_post_feature_columns,
            "suggest_ablation_for_pre_model": ablation_pre_feature_columns,
            "suggest_ablation_for_post_model": ablation_post_feature_columns,
            "forbidden_for_training": forbidden_feature_columns,
            "feature_source_timing_notes": {
                "tri_points_candidate": "Candidate-stage quantity derived from the candidate generation/screening phase, not a post-accept final geometry summary.",
                "pnp_success_rate": "Candidate-stage quantity derived from current candidate evaluation/screening, without using future information or final gate outcome.",
            },
            "notes": [
                "pre model may only use candidate-stage and front-end support features.",
                "post model may extend pre features with reprojection/depth-sign geometry features.",
                "metadata/protocol/provenance/runtime/gate/Q/GT columns are audit or posthoc only.",
                "front_grid_entropy, triangulation_ratio, and pnp_median_inliers are retained as ablation candidates, not baseline-required features.",
            ],
        },
    )
    legality_board_hash = hashlib.sha256(legality_board_path.read_bytes()).hexdigest()

    manifest = {
        "script_name": THIS_FILE.name,
        "input_roots": [str(p) for p in input_roots],
        "step10b_csv_count": len(step10b_csvs),
        "step11_csv_count": len(step11_csvs),
        "num_rows": len(rows),
        "output_csv": str(dataset_csv),
        "output_split_csvs": (
            {
                split: str(out_dir / f"risk_dataset_{split}.csv")
                for split in sorted({str(row.get('dataset_row_split', 'unsplit')) for row in rows})
            }
            if args.write_split_csvs else {}
        ),
        "schema_versions": schema_versions,
        "sample_types": sample_types,
        "Y_bad_v1_sources": ybad_sources,
        "feature_columns_path": str(feature_columns_path),
        "forbidden_feature_columns_path": str(forbidden_columns_path),
        "feature_legality_board_path": str(legality_board_path),
        "feature_legality_board_hash": legality_board_hash,
        "feature_columns_count": len(feature_columns),
        "pre_feature_columns_count": len(pre_feature_columns),
        "post_feature_columns_count": len(post_feature_columns),
        "forbidden_feature_columns": forbidden_feature_columns,
        "label_columns": label_columns,
        "posthoc_analysis_columns": posthoc_analysis_columns,
        "metadata_columns": metadata_columns,
        "split_mode": args.split_mode,
        "protocol_columns": protocol_columns,
        "data_quality_columns": data_quality_columns,
        "label_version": "Y_bad_v1",
        "label_scope": "init_level_bad_event_proxy",
        "claim_guardrails": {
            "allowed_claims": [
                "Current results establish a feasible initialization-risk modeling chain under frozen protocol and strict task-specific feature constraints.",
                "Current results support task-level feasibility claims for pre-risk and post-risk, not system-level robustness claims.",
            ],
            "forbidden_claims": [
                "Do not claim system-level failure prediction or system-level gate benefit from Y_bad_v1-based models.",
                "Do not claim cross-scene or cross-dynamic-level generalization from the current 4-sequence split.",
                "Do not claim monotonic constrained modeling from direction-aligned baselines.",
            ],
            "proxy_bias_notes": [
                "Y_bad_v1 is an initialization-level proxy bad-event label and is not yet a downstream system failure horizon label.",
                "Current proxy labels are more sensitive to initialization geometry/solver failure than to delayed tracking collapse after acceptance.",
            ],
            "experimental_limit_flags": [
                "small_sample",
                "single_protocol_family",
                "dynamic_level_split_coupling",
                "proxy_label_only",
                "no_system_level_closed_loop_evidence",
            ],
        },
        "notes": [
            "Y_bad_v1 is initialization-level bad event, not yet a system-level failure horizon label.",
            "Old and new step10b/step11 schemas are normalized into one table.",
            "protocol_* columns describe experiment protocol; dataset_row_split is the builder-assigned train/val/test split.",
            "Missing protocol sidecars are recorded as unknown protocol metadata.",
            "Strict split is group-aware and sequence-level by default.",
            "sequence_dynamic_level is derived from sequence naming only and should be treated as coarse metadata.",
        ],
    }
    if args.y_bad_v2_min_labels_csv:
        manifest["available_label_versions"] = ["Y_bad_v1", "Y_bad_v2_min_default"]
        manifest["y_bad_v2_min_default"] = {
            "labels_csv": str(Path(args.y_bad_v2_min_labels_csv).expanduser().resolve()),
            "label_scope": "post_accept_short_horizon_instability_proxy",
            "task_open": ["post"],
            "task_closed": ["pre"],
            "label_population": "accepted_step11_rows_with_sufficient_future_horizon",
            "coverage_summary": v2_coverage_summary,
            "forbidden_claims": [
                "Do not expose Y_bad_v2_min_default as a validated pre-task label.",
                "Do not claim system-level reset/reinit/tracking-failure prediction from Y_bad_v2_min_default alone.",
            ],
        }
    else:
        manifest["available_label_versions"] = ["Y_bad_v1"]
    manifest_path = out_dir / "dataset_manifest.json"
    write_json(manifest_path, manifest)

    audit_path = out_dir / "dataset_audit.json"
    write_json(audit_path, audit)

    split_manifest_path = out_dir / "dataset_split_manifest.json"
    write_json(split_manifest_path, split_info)

    if args.write_task_split_csvs:
        task_metadata_columns = [
            "sample_uid",
            "attempt_uid",
            "sequence",
            "variant_tag",
            "window_id",
            "dataset_row_split",
            "sample_type",
            "schema_version",
        ]
        task_specs = {
            "pre": {
                "allowed_feature_columns": keep_pre_feature_columns,
                "ablation_feature_columns": ablation_pre_feature_columns,
                "forbidden_columns": pre_forbidden_columns,
            },
            "post": {
                "allowed_feature_columns": keep_post_feature_columns,
                "ablation_feature_columns": ablation_post_feature_columns,
                "forbidden_columns": post_forbidden_columns,
            },
        }
        schema_version_counts = {}
        for row in rows:
            schema_version_counts[row["schema_version"]] = schema_version_counts.get(row["schema_version"], 0) + 1
        for task_name, spec in task_specs.items():
            policy = task_claim_policy(task_name)
            allowed_columns = list(task_metadata_columns) + list(spec["allowed_feature_columns"]) + label_columns
            for split in sorted({str(row.get("dataset_row_split", "unsplit")) for row in rows}):
                split_rows = [row for row in rows if str(row.get("dataset_row_split", "unsplit")) == split]
                out_csv = out_dir / f"risk_dataset_{task_name}_{split}.csv"
                write_selected_csv(out_csv, split_rows, allowed_columns)
            task_manifest = {
                "task_name": task_name,
                "allowed_feature_columns": spec["allowed_feature_columns"],
                "ablation_feature_columns": spec["ablation_feature_columns"],
                "label_columns": label_columns,
                "task_metadata_columns": task_metadata_columns,
                "split_policy": args.split_mode,
                "label_version": "Y_bad_v1",
                "label_scope": "init_level_bad_event_proxy",
                "source_schema_versions": schema_version_counts,
                "legality_board_path": str(legality_board_path),
                "legality_board_hash": legality_board_hash,
                "protocol_columns_excluded": protocol_columns,
                "posthoc_columns_excluded": posthoc_analysis_columns,
                "data_quality_columns_excluded": data_quality_columns,
                "forbidden_columns": spec["forbidden_columns"],
                "task_scope": policy["task_scope"],
                "allowed_claims": policy["allowed_claims"],
                "forbidden_claims": policy["forbidden_claims"],
                "proxy_bias_notes": policy["proxy_bias_notes"],
                "experimental_limit_flags": policy["experimental_limit_flags"],
                "output_split_csvs": {
                    split: str(out_dir / f"risk_dataset_{task_name}_{split}.csv")
                    for split in sorted({str(row.get('dataset_row_split', 'unsplit')) for row in rows})
                },
            }
            write_json(out_dir / f"risk_dataset_{task_name}_manifest.json", task_manifest)

        if args.y_bad_v2_min_labels_csv:
            policy = task_claim_policy_post_v2_default()
            task_name = "post_v2_min_default"
            label_columns_v2 = ["Y_bad_v2_min_default"]
            labeled_rows = [row for row in rows if str(row.get("Y_bad_v2_min_default", "")) in ("0", "1")]
            allowed_columns = list(task_metadata_columns) + list(keep_post_feature_columns) + label_columns_v2
            split_values = sorted({str(row.get("dataset_row_split", "unsplit")) for row in labeled_rows})
            split_files = {}
            for split in split_values:
                split_rows = [row for row in labeled_rows if str(row.get("dataset_row_split", "unsplit")) == split]
                out_csv = out_dir / f"risk_dataset_{task_name}_{split}.csv"
                write_selected_csv(out_csv, split_rows, allowed_columns)
                split_files[split] = str(out_csv)
            task_manifest = {
                "task_name": task_name,
                "allowed_feature_columns": keep_post_feature_columns,
                "ablation_feature_columns": ablation_post_feature_columns,
                "label_columns": label_columns_v2,
                "task_metadata_columns": task_metadata_columns,
                "split_policy": args.split_mode,
                "label_version": "Y_bad_v2_min_default",
                "label_scope": "post_accept_short_horizon_instability_proxy",
                "label_population": "accepted_step11_rows_with_sufficient_future_horizon",
                "coverage_summary": {
                    "matched_step11_rows": int(sum(1 for row in rows if row.get("sample_type") == "step11" and row.get("attempt_uid") in v2_labels_by_attempt)),
                    "labeled_rows": int(len(labeled_rows)),
                    "bad_ratio_labeled": (
                        float(sum(int(row["Y_bad_v2_min_default"]) for row in labeled_rows) / max(1, len(labeled_rows)))
                        if labeled_rows else None
                    ),
                    "nonempty_splits": split_values,
                    "source_counts": v2_coverage_summary.get("source_counts", {}),
                },
                "source_schema_versions": schema_version_counts,
                "legality_board_path": str(legality_board_path),
                "legality_board_hash": legality_board_hash,
                "protocol_columns_excluded": protocol_columns,
                "posthoc_columns_excluded": posthoc_analysis_columns,
                "data_quality_columns_excluded": data_quality_columns,
                "forbidden_columns": post_forbidden_columns,
                "task_scope": policy["task_scope"],
                "allowed_claims": policy["allowed_claims"],
                "forbidden_claims": policy["forbidden_claims"],
                "proxy_bias_notes": policy["proxy_bias_notes"],
                "experimental_limit_flags": policy["experimental_limit_flags"],
                "output_split_csvs": split_files,
            }
            write_json(out_dir / f"risk_dataset_{task_name}_manifest.json", task_manifest)

    print(f"[RiskDataset] step10b_csv_count={len(step10b_csvs)}")
    print(f"[RiskDataset] step11_csv_count={len(step11_csvs)}")
    print(f"[RiskDataset] num_rows={len(rows)}")
    print(f"[RiskDataset] saved csv: {dataset_csv}")
    print(f"[RiskDataset] saved manifest: {manifest_path}")
    print(f"[RiskDataset] saved audit: {audit_path}")
    print(f"[RiskDataset] saved split manifest: {split_manifest_path}")
    print(f"[RiskDataset] saved feature columns: {feature_columns_path}")
    print(f"[RiskDataset] saved forbidden columns: {forbidden_columns_path}")
    print(f"[RiskDataset] saved legality board: {legality_board_path}")


if __name__ == "__main__":
    main()
