#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import math
import re


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = None
for p in [THIS_FILE.parent, *THIS_FILE.parent.parents]:
    if (p / "configs").is_dir() and ((p / "pipelines").is_dir() or (p / " pipelines").is_dir()):
        PROJECT_ROOT = p
        break
if PROJECT_ROOT is None:
    raise RuntimeError(f"Cannot locate project root from {THIS_FILE}")


LABEL_VERSION_DEFAULT = "Y_bad_v2_min_default"
LABEL_SCOPE_DEFAULT = "post_accept_short_horizon_instability_proxy"


def load_csv_rows(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


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


def infer_variant_tag(csv_path: Path) -> str:
    return csv_path.parent.name


def infer_dataset_group(csv_path: Path) -> str:
    parts = list(csv_path.parts)
    for i, part in enumerate(parts):
        if part == "VIODE" and i + 1 < len(parts):
            return parts[i + 1]
    return ""


def load_current_context_by_attempt(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    rows = load_csv_rows(path)
    out = {}
    for row in rows:
        attempt_uid = str(row.get("attempt_uid", "")).strip()
        if not attempt_uid:
            continue
        out[attempt_uid] = row
    return out


def build_ids(sequence: str, variant_tag: str, window_id: int) -> dict:
    attempt_uid = f"{sequence}::{variant_tag}::{int(window_id):06d}"
    sample_uid = f"step11::{attempt_uid}"
    return {
        "sample_uid": sample_uid,
        "attempt_uid": attempt_uid,
    }


def collect_step11_csvs(input_roots: list[Path]) -> list[Path]:
    csvs = []
    for root in input_roots:
        if root.is_file():
            if root.name == "summary.csv":
                csvs.append(root)
            continue
        csvs.extend(sorted(root.rglob("sfm_init_quality_v2/*/*/summary.csv")))
    return sorted({p.resolve() for p in csvs})


def normalize_row(row: dict, csv_path: Path, current_context: dict[str, dict] | None = None) -> dict:
    seq_name = csv_path.parents[1].name if len(csv_path.parents) >= 2 else ""
    dataset_group = infer_dataset_group(csv_path)
    sequence = f"{dataset_group}/{seq_name}" if dataset_group else seq_name
    variant_tag = infer_variant_tag(csv_path)
    window_id = safe_int(row.get("start_frame_cfg"), -1)
    ids = build_ids(sequence, variant_tag, window_id)
    ctx = (current_context or {}).get(ids["attempt_uid"], {})
    return {
        **ids,
        "sequence": sequence,
        "variant_tag": variant_tag,
        "window_id": window_id,
        "window_W": safe_int(row.get("window_W")),
        "start_frame_global": safe_int(row.get("start_frame_global")),
        "gate_post": str(row.get("ba_gate_post", row.get("ba_gate_decision", ""))),
        "gate_post_reason": str(row.get("ba_gate_post_reason", "")),
        "success_strict": safe_bool(row.get("ba_success_strict"), None),
        "solver_ok": safe_bool(row.get("ba_solver_ok"), None),
        "geom_ok": safe_bool(row.get("ba_geom_ok"), None),
        "accepted_but_bad_geom": safe_bool(row.get("ba_accepted_but_bad_geom"), False),
        "reproj_med_px": safe_float(row.get("ba_reproj_med_px")),
        "gt_rot_med_deg": safe_float(row.get("gt_rot_med_deg")),
        "gt_trans_dir_med_deg": safe_float(row.get("gt_trans_dir_med_deg")),
        "current_parallax_px_candidate": safe_float(
            ctx.get("parallax_px_candidate", ctx.get("current_parallax_px_candidate"))
        ),
        "source_csv": str(csv_path),
    }


def evaluate_future_trigger(
    current_row: dict,
    future_row: dict,
    tau_reproj: float,
    tau_rot: float,
    tau_trans: float,
    future_high_gt_rot_current_parallax_min: float | None,
    enabled_triggers: set[str],
):
    if "future_reset" in enabled_triggers and str(future_row.get("gate_post", "")) == "reset":
        return "future_reset"
    if "future_solver_fail" in enabled_triggers and (
        future_row.get("success_strict") is False or future_row.get("solver_ok") is False
    ):
        return "future_solver_fail"
    if "future_geom_fail" in enabled_triggers and (
        future_row.get("geom_ok") is False or future_row.get("accepted_but_bad_geom") is True
    ):
        return "future_geom_fail"
    reproj = safe_float(future_row.get("reproj_med_px"))
    if "future_high_reproj" in enabled_triggers and reproj is not None and reproj > float(tau_reproj):
        return "future_high_reproj"
    gt_rot = safe_float(future_row.get("gt_rot_med_deg"))
    if "future_high_gt_rot" in enabled_triggers and gt_rot is not None and gt_rot > float(tau_rot):
        if future_high_gt_rot_current_parallax_min is not None:
            current_parallax = safe_float(current_row.get("current_parallax_px_candidate"))
            if current_parallax is None or current_parallax < float(future_high_gt_rot_current_parallax_min):
                pass
            else:
                return "future_high_gt_rot"
        else:
            return "future_high_gt_rot"
    gt_trans = safe_float(future_row.get("gt_trans_dir_med_deg"))
    if "future_high_gt_trans" in enabled_triggers and gt_trans is not None and gt_trans > float(tau_trans):
        return "future_high_gt_trans"
    return None


def build_labels(
    rows: list[dict],
    horizon_windows: int,
    tau_reproj: float,
    tau_rot: float,
    tau_trans: float,
    future_high_gt_rot_current_parallax_min: float | None,
    enabled_triggers: set[str],
):
    by_group = {}
    for row in rows:
        key = (row["sequence"], row["variant_tag"])
        by_group.setdefault(key, []).append(row)
    for group_rows in by_group.values():
        group_rows.sort(key=lambda r: (int(r.get("window_id", -1)), int(r.get("start_frame_global") or -1)))

    labeled_rows = []
    trigger_counts = {}
    source_counts = {}
    for key, group_rows in by_group.items():
        for i, row in enumerate(group_rows):
            out = dict(row)
            out["Y_bad_v2_min_horizon_windows"] = int(horizon_windows)
            out["Y_bad_v2_min_label_version"] = LABEL_VERSION_DEFAULT
            out["Y_bad_v2_min_label_scope"] = LABEL_SCOPE_DEFAULT
            out["Y_bad_v2_min"] = ""
            out["Y_bad_v2_min_source"] = ""
            out["prov_Y_bad_v2_min"] = "missing"
            out["Y_bad_v2_min_trigger"] = ""
            out["Y_bad_v2_min_trigger_window_id"] = ""
            out[LABEL_VERSION_DEFAULT] = ""
            out[f"{LABEL_VERSION_DEFAULT}_source"] = ""
            out[f"prov_{LABEL_VERSION_DEFAULT}"] = "missing"
            out[f"{LABEL_VERSION_DEFAULT}_trigger"] = ""
            out[f"{LABEL_VERSION_DEFAULT}_trigger_window_id"] = ""
            out[f"{LABEL_VERSION_DEFAULT}_label_version"] = LABEL_VERSION_DEFAULT
            out[f"{LABEL_VERSION_DEFAULT}_label_scope"] = LABEL_SCOPE_DEFAULT
            out[f"{LABEL_VERSION_DEFAULT}_horizon_windows"] = int(horizon_windows)

            gate_post = str(row.get("gate_post", ""))
            if gate_post != "accept":
                out["Y_bad_v2_min_source"] = "not_applicable_non_accept"
                out["prov_Y_bad_v2_min"] = "not_applicable"
                out[f"{LABEL_VERSION_DEFAULT}_source"] = "not_applicable_non_accept"
                out[f"prov_{LABEL_VERSION_DEFAULT}"] = "not_applicable"
                source_counts[out["Y_bad_v2_min_source"]] = source_counts.get(out["Y_bad_v2_min_source"], 0) + 1
                labeled_rows.append(out)
                continue

            future_rows = group_rows[i + 1:i + 1 + int(horizon_windows)]
            if len(future_rows) < int(horizon_windows):
                out["Y_bad_v2_min_source"] = "insufficient_future_horizon"
                out["prov_Y_bad_v2_min"] = "missing"
                out[f"{LABEL_VERSION_DEFAULT}_source"] = "insufficient_future_horizon"
                out[f"prov_{LABEL_VERSION_DEFAULT}"] = "missing"
                source_counts[out["Y_bad_v2_min_source"]] = source_counts.get(out["Y_bad_v2_min_source"], 0) + 1
                labeled_rows.append(out)
                continue

            trigger = None
            trigger_window = None
            for future_row in future_rows:
                trigger = evaluate_future_trigger(
                    row,
                    future_row,
                    tau_reproj,
                    tau_rot,
                    tau_trans,
                    future_high_gt_rot_current_parallax_min,
                    enabled_triggers,
                )
                if trigger is not None:
                    trigger_window = future_row["window_id"]
                    break

            if trigger is None:
                out["Y_bad_v2_min"] = 0
                out["Y_bad_v2_min_source"] = "accepted_horizon_stable"
                out["prov_Y_bad_v2_min"] = "native"
                out["Y_bad_v2_min_trigger"] = "stable_horizon"
                out[LABEL_VERSION_DEFAULT] = 0
                out[f"{LABEL_VERSION_DEFAULT}_source"] = "accepted_horizon_stable"
                out[f"prov_{LABEL_VERSION_DEFAULT}"] = "native"
                out[f"{LABEL_VERSION_DEFAULT}_trigger"] = "stable_horizon"
            else:
                out["Y_bad_v2_min"] = 1
                out["Y_bad_v2_min_source"] = "accepted_horizon_unstable"
                out["prov_Y_bad_v2_min"] = "native"
                out["Y_bad_v2_min_trigger"] = trigger
                out["Y_bad_v2_min_trigger_window_id"] = trigger_window
                out[LABEL_VERSION_DEFAULT] = 1
                out[f"{LABEL_VERSION_DEFAULT}_source"] = "accepted_horizon_unstable"
                out[f"prov_{LABEL_VERSION_DEFAULT}"] = "native"
                out[f"{LABEL_VERSION_DEFAULT}_trigger"] = trigger
                out[f"{LABEL_VERSION_DEFAULT}_trigger_window_id"] = trigger_window
                trigger_counts[trigger] = trigger_counts.get(trigger, 0) + 1

            source_counts[out["Y_bad_v2_min_source"]] = source_counts.get(out["Y_bad_v2_min_source"], 0) + 1
            labeled_rows.append(out)
    return labeled_rows, source_counts, trigger_counts


def build_audit(
    rows: list[dict],
    source_counts: dict,
    trigger_counts: dict,
    horizon_windows: int,
    tau_reproj: float,
    tau_rot: float,
    tau_trans: float,
    future_high_gt_rot_current_parallax_min: float | None,
    enabled_triggers: set[str],
):
    applicable = [r for r in rows if r.get("Y_bad_v2_min_source") not in ("not_applicable_non_accept", "insufficient_future_horizon")]
    labeled = [r for r in applicable if r.get("Y_bad_v2_min") in (0, 1)]
    bad_vals = [int(r["Y_bad_v2_min"]) for r in labeled]
    by_sequence = {}
    for row in rows:
        seq = str(row.get("sequence", ""))
        by_sequence.setdefault(seq, {"num_rows": 0, "accept_rows": 0, "labeled_rows": 0, "bad_rows": 0})
        by_sequence[seq]["num_rows"] += 1
        if str(row.get("gate_post", "")) == "accept":
            by_sequence[seq]["accept_rows"] += 1
        if row.get("Y_bad_v2_min") in (0, 1):
            by_sequence[seq]["labeled_rows"] += 1
            by_sequence[seq]["bad_rows"] += int(row["Y_bad_v2_min"])

    return {
        "num_rows": len(rows),
        "accept_rows": int(sum(1 for r in rows if str(r.get("gate_post", "")) == "accept")),
        "applicable_rows": len(applicable),
        "labeled_rows": len(labeled),
        "bad_ratio_labeled": float(sum(bad_vals) / max(1, len(bad_vals))) if labeled else None,
        "source_counts": source_counts,
        "trigger_counts": trigger_counts,
        "by_sequence": by_sequence,
        "label_version": LABEL_VERSION_DEFAULT,
        "label_scope": LABEL_SCOPE_DEFAULT,
        "label_population": "accepted_step11_rows_with_sufficient_future_horizon",
        "coverage_summary": {
            "total_rows": len(rows),
            "accept_rows": int(sum(1 for r in rows if str(r.get("gate_post", "")) == "accept")),
            "applicable_rows": len(applicable),
            "labeled_rows": len(labeled),
            "coverage_over_total_ratio": float(len(labeled) / max(1, len(rows))),
            "coverage_over_accept_ratio": float(len(labeled) / max(1, sum(1 for r in rows if str(r.get("gate_post", "")) == "accept"))),
        },
        "forbidden_claims": [
            "Do not claim Y_bad_v2_min_default is a full system-level reset/reinit/tracking-failure ground-truth label.",
            "Do not extend Y_bad_v2_min_default to pre-task training without separate validation.",
            "Do not claim cross-scene or cross-dynamic-level generalization from the current 4-sequence audit.",
        ],
        "horizon_windows": int(horizon_windows),
        "thresholds": {
            "tau_reproj_med_px": float(tau_reproj),
            "tau_gt_rot_med_deg": float(tau_rot),
            "tau_gt_trans_dir_med_deg": float(tau_trans),
            "future_high_gt_rot_current_parallax_min": (
                None if future_high_gt_rot_current_parallax_min is None else float(future_high_gt_rot_current_parallax_min)
            ),
        },
        "enabled_triggers": sorted(enabled_triggers),
        "notes": [
            "Y_bad_v2_min_default is a short-horizon instability proxy defined only for gate_post == accept rows with sufficient future windows.",
            "It is not yet a system-level reset/reinit/tracking-failure ground-truth label.",
            "Current triggers are configurable subsets of future reset, solver fail, geometry fail, or GT/reprojection degradation within K future evaluation windows.",
        ],
    }


def main():
    ap = argparse.ArgumentParser(description="Build Y_bad_v2_min labels from step11 future-horizon instability.")
    ap.add_argument("--input_root", action="append", default=[], help="Repeatable input root. Can be a step11 summary.csv or a result dir.")
    ap.add_argument("--out_dir", default=str(PROJECT_ROOT / "results" / "y_bad_v2_min"))
    ap.add_argument("--horizon_windows", type=int, default=1)
    ap.add_argument("--tau_reproj_med_px", type=float, default=3.0)
    ap.add_argument("--tau_gt_rot_med_deg", type=float, default=5.0)
    ap.add_argument("--tau_gt_trans_dir_med_deg", type=float, default=10.0)
    ap.add_argument(
        "--future_high_gt_rot_current_parallax_min",
        type=float,
        default=None,
        help="Optional current-window parallax guard for future_high_gt_rot trigger.",
    )
    ap.add_argument(
        "--current_context_csv",
        default=None,
        help="Optional CSV with attempt_uid keyed current-window context such as parallax_px_candidate.",
    )
    ap.add_argument(
        "--enabled_triggers",
        default="future_reset,future_solver_fail,future_high_gt_rot",
        help="Comma-separated trigger set for Y_bad_v2_min.",
    )
    args = ap.parse_args()

    input_roots = [Path(p).expanduser().resolve() for p in args.input_root] if args.input_root else [
        Path("/mnt/g/Result/VIODE/city_day/stageB_full_compare/step1_11_cam0").resolve()
    ]
    csvs = collect_step11_csvs(input_roots)
    enabled_triggers = {s.strip() for s in str(args.enabled_triggers).split(",") if s.strip()}
    valid_triggers = {
        "future_reset",
        "future_solver_fail",
        "future_geom_fail",
        "future_high_reproj",
        "future_high_gt_rot",
        "future_high_gt_trans",
    }
    unknown = sorted(enabled_triggers - valid_triggers)
    if unknown:
        raise ValueError(f"Unknown enabled triggers: {unknown}")
    current_context = load_current_context_by_attempt(
        Path(args.current_context_csv).expanduser().resolve() if args.current_context_csv else None
    )
    rows = []
    for csv_path in csvs:
        for row in load_csv_rows(csv_path):
            rows.append(normalize_row(row, csv_path, current_context=current_context))

    labeled_rows, source_counts, trigger_counts = build_labels(
        rows,
        horizon_windows=int(args.horizon_windows),
        tau_reproj=float(args.tau_reproj_med_px),
        tau_rot=float(args.tau_gt_rot_med_deg),
        tau_trans=float(args.tau_gt_trans_dir_med_deg),
        future_high_gt_rot_current_parallax_min=args.future_high_gt_rot_current_parallax_min,
        enabled_triggers=enabled_triggers,
    )
    audit = build_audit(
        labeled_rows,
        source_counts=source_counts,
        trigger_counts=trigger_counts,
        horizon_windows=int(args.horizon_windows),
        tau_reproj=float(args.tau_reproj_med_px),
        tau_rot=float(args.tau_gt_rot_med_deg),
        tau_trans=float(args.tau_gt_trans_dir_med_deg),
        future_high_gt_rot_current_parallax_min=args.future_high_gt_rot_current_parallax_min,
        enabled_triggers=enabled_triggers,
    )

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_csv = out_dir / "y_bad_v2_min_labels.csv"
    audit_json = out_dir / "y_bad_v2_min_audit.json"
    manifest_json = out_dir / "y_bad_v2_min_manifest.json"
    write_csv(labels_csv, labeled_rows)
    write_json(audit_json, audit)
    write_json(
        manifest_json,
        {
            "script_name": THIS_FILE.name,
            "input_roots": [str(p) for p in input_roots],
            "step11_csv_count": len(csvs),
            "num_rows": len(labeled_rows),
            "current_context_csv": None if not args.current_context_csv else str(Path(args.current_context_csv).expanduser().resolve()),
            "label_version": LABEL_VERSION_DEFAULT,
            "label_scope": LABEL_SCOPE_DEFAULT,
            "label_population": audit["label_population"],
            "coverage_summary": audit["coverage_summary"],
            "forbidden_claims": audit["forbidden_claims"],
            "horizon_windows": int(args.horizon_windows),
            "enabled_triggers": sorted(enabled_triggers),
            "thresholds": {
                "tau_reproj_med_px": float(args.tau_reproj_med_px),
                "tau_gt_rot_med_deg": float(args.tau_gt_rot_med_deg),
                "tau_gt_trans_dir_med_deg": float(args.tau_gt_trans_dir_med_deg),
                "future_high_gt_rot_current_parallax_min": (
                    None if args.future_high_gt_rot_current_parallax_min is None else float(args.future_high_gt_rot_current_parallax_min)
                ),
            },
            "outputs": {
                "labels_csv": str(labels_csv),
                "audit_json": str(audit_json),
                "manifest_json": str(manifest_json),
            },
            "notes": audit["notes"],
        },
    )

    print(f"[Y_bad_v2_min] step11_csv_count={len(csvs)}")
    print(f"[Y_bad_v2_min] num_rows={len(labeled_rows)}")
    print(f"[Y_bad_v2_min] labeled_rows={audit['labeled_rows']} bad_ratio={audit['bad_ratio_labeled']}")
    print(f"[Y_bad_v2_min] saved labels: {labels_csv}")
    print(f"[Y_bad_v2_min] saved audit: {audit_json}")
    print(f"[Y_bad_v2_min] saved manifest: {manifest_json}")


if __name__ == "__main__":
    main()
