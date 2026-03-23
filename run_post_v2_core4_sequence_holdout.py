#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import subprocess
import sys


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = None
for p in [THIS_FILE.parent, *THIS_FILE.parent.parents]:
    if (p / "configs").is_dir() and ((p / "pipelines").is_dir() or (p / " pipelines").is_dir()):
        PROJECT_ROOT = p
        break
if PROJECT_ROOT is None:
    raise RuntimeError(f"Cannot locate project root from {THIS_FILE}")


SOURCE_COL = "parallax_px_candidate"


def gated_feature_name(threshold: float) -> str:
    return f"parallax_px_candidate_gated_ge_{int(round(float(threshold)))}"


def build_core4(gated_col: str) -> list[tuple[str, list[str]]]:
    return [
        ("geometry_only", ["reproj_med_px", "reproj_p90_px", "cheirality_ratio"]),
        ("geometry_plus_gated_parallax", ["reproj_med_px", "reproj_p90_px", "cheirality_ratio", gated_col]),
        ("drop_front", [SOURCE_COL, "tri_points_candidate", "pnp_success_rate", "reproj_med_px", "reproj_p90_px", "cheirality_ratio"]),
        ("full_anchor", ["front_p_static", "front_p_band", "front_coverage_ratio", "front_kept_dyn_ratio", SOURCE_COL, "tri_points_candidate", "pnp_success_rate", "reproj_med_px", "reproj_p90_px", "cheirality_ratio"]),
    ]


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_csv_rows(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def read_header(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return next(csv.reader(f))


def safe_float(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def run_baseline(result_root: Path, feature_cols: list[str], out_dir: Path) -> dict:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "sfm_init" / "train_init_risk_baseline.py"),
        "--task", "post",
        "--model", "logistic",
        "--train_csv", str(result_root / "risk_dataset_post_v2_min_default_train.csv"),
        "--val_csv", str(result_root / "risk_dataset_post_v2_min_default_val.csv"),
        "--test_csv", str(result_root / "risk_dataset_post_v2_min_default_test.csv"),
        "--manifest", str(result_root / "risk_dataset_post_v2_min_default_manifest.json"),
        "--label_col", "Y_bad_v2_min_default",
        "--out_dir", str(out_dir),
        "--override_feature_columns", ",".join(feature_cols),
    ]
    subprocess.run(cmd, check=True)
    return load_json(out_dir / "metrics.json")


def main():
    ap = argparse.ArgumentParser(description="Run explicit sequence-holdout core4 evaluation for post_v2.")
    ap.add_argument("--base_root", required=True)
    ap.add_argument("--test_sequence", required=True)
    ap.add_argument("--val_sequences", required=True, help="Comma-separated sequence names for validation split.")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--gated_parallax_threshold", type=float, default=60.0)
    args = ap.parse_args()

    base_root = Path(args.base_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    test_sequence = str(args.test_sequence)
    val_sequences = [s.strip() for s in str(args.val_sequences).split(",") if s.strip()]
    val_set = set(val_sequences)
    gated_col = gated_feature_name(float(args.gated_parallax_threshold))

    all_rows = load_csv_rows(base_root / "risk_dataset.csv")
    task_manifest = load_json(base_root / "risk_dataset_post_v2_min_default_manifest.json")
    dataset_manifest = load_json(base_root / "dataset_manifest.json")
    feature_columns = load_json(base_root / "feature_columns.json")
    legality_board = load_json(base_root / "feature_legality_board.json")
    forbidden_columns = load_json(base_root / "forbidden_feature_columns.json")

    sequences = sorted({str(r["sequence"]) for r in all_rows})
    if test_sequence not in sequences:
        raise ValueError(f"test_sequence not found: {test_sequence}")
    missing_val = [s for s in val_sequences if s not in sequences]
    if missing_val:
        raise ValueError(f"val_sequences not found: {missing_val}")
    if test_sequence in val_set:
        raise ValueError("test_sequence must not appear in val_sequences")

    sequence_to_split = {}
    for seq in sequences:
        if seq == test_sequence:
            sequence_to_split[seq] = "test"
        elif seq in val_set:
            sequence_to_split[seq] = "val"
        else:
            sequence_to_split[seq] = "train"

    for row in all_rows:
        row["dataset_row_split"] = sequence_to_split[str(row["sequence"])]
        v = safe_float(row.get(SOURCE_COL), 0.0)
        row[gated_col] = v if v >= float(args.gated_parallax_threshold) else 0.0

    write_csv(out_dir / "risk_dataset.csv", list(all_rows[0].keys()), all_rows)
    write_json(
        out_dir / "dataset_split_manifest.json",
        {
            "split_mode": "explicit_sequence_holdout",
            "test_sequence": test_sequence,
            "val_sequences": val_sequences,
            "sequence_to_split": sequence_to_split,
        },
    )

    dataset_manifest["split_mode"] = "explicit_sequence_holdout"
    dataset_manifest["split_seed"] = None
    dataset_manifest["explicit_holdout"] = {
        "test_sequence": test_sequence,
        "val_sequences": val_sequences,
    }
    write_json(out_dir / "dataset_manifest.json", dataset_manifest)
    write_json(out_dir / "feature_columns.json", feature_columns)
    write_json(out_dir / "feature_legality_board.json", legality_board)
    write_json(out_dir / "forbidden_feature_columns.json", forbidden_columns)

    task_header = read_header(base_root / "risk_dataset_post_v2_min_default_train.csv")
    if gated_col not in task_header:
        task_header.append(gated_col)
    labeled = [
        row for row in all_rows
        if str(row.get("sample_type", "")) == "step11" and str(row.get("Y_bad_v2_min_default", "")) in ("0", "1")
    ]
    split_rows = {
        "train": [row for row in labeled if row["dataset_row_split"] == "train"],
        "val": [row for row in labeled if row["dataset_row_split"] == "val"],
        "test": [row for row in labeled if row["dataset_row_split"] == "test"],
    }
    for split_name, rows in split_rows.items():
        write_csv(
            out_dir / f"risk_dataset_post_v2_min_default_{split_name}.csv",
            task_header,
            [{k: row.get(k, "") for k in task_header} for row in rows],
        )
    task_manifest["split_policy"] = "explicit_sequence_holdout"
    task_manifest["explicit_holdout"] = {
        "test_sequence": test_sequence,
        "val_sequences": val_sequences,
    }
    allowed = list(task_manifest.get("allowed_feature_columns", []))
    if gated_col not in allowed:
        allowed.append(gated_col)
    task_manifest["allowed_feature_columns"] = allowed
    task_manifest["gated_parallax_feature"] = {
        "name": gated_col,
        "source_column": SOURCE_COL,
        "activation_rule": f"{SOURCE_COL} >= {float(args.gated_parallax_threshold):.1f}",
        "inactive_value": 0.0,
    }
    write_json(out_dir / "risk_dataset_post_v2_min_default_manifest.json", task_manifest)

    summary_rows = []
    for model_name, cols in build_core4(gated_col):
        model_dir = out_dir / "core4_runs" / model_name
        metrics = run_baseline(out_dir, cols, model_dir)
        test_metrics = metrics["splits"]["test"]
        summary_rows.append(
            {
                "model": model_name,
                "features": ",".join(cols),
                "num_rows": test_metrics["num_rows"],
                "positive_ratio": test_metrics["positive_ratio"],
                "auroc": test_metrics["auroc"],
                "auprc": test_metrics["auprc"],
                "brier": test_metrics["brier"],
                "ece": test_metrics["ece"],
            }
        )
    summary_rows.sort(key=lambda r: float(r["auroc"]), reverse=True)
    write_csv(
        out_dir / "post_v2_core4_holdout_summary.csv",
        list(summary_rows[0].keys()),
        summary_rows,
    )
    write_json(
        out_dir / "post_v2_core4_holdout_summary.json",
        {
            "test_sequence": test_sequence,
            "val_sequences": val_sequences,
            "rows": summary_rows,
        },
    )


if __name__ == "__main__":
    main()
