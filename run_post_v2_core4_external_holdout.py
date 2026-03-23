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


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_csv_rows(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_header(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return next(csv.reader(f))


def safe_float(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def run_baseline(package_dir: Path, feature_cols: list[str], out_dir: Path) -> dict:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "sfm_init" / "train_init_risk_baseline.py"),
        "--task", "post",
        "--model", "logistic",
        "--train_csv", str(package_dir / "risk_dataset_post_v2_min_default_train.csv"),
        "--val_csv", str(package_dir / "risk_dataset_post_v2_min_default_val.csv"),
        "--test_csv", str(package_dir / "risk_dataset_post_v2_min_default_test.csv"),
        "--manifest", str(package_dir / "risk_dataset_post_v2_min_default_manifest.json"),
        "--label_col", "Y_bad_v2_min_default",
        "--out_dir", str(out_dir),
        "--override_feature_columns", ",".join(feature_cols),
    ]
    subprocess.run(cmd, check=True)
    return load_json(out_dir / "metrics.json")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run fixed core4 external held-out evaluation on a labeled external package.")
    ap.add_argument("--base_root", required=True, help="Base VIODE post_v2 package with train/val splits.")
    ap.add_argument("--external_root", required=True, help="External package root containing risk_dataset.csv with Y_bad_v2_min_default.")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--external_name", default="", help="Display name for the external test set.")
    ap.add_argument("--gated_parallax_threshold", type=float, default=60.0)
    args = ap.parse_args()

    base_root = Path(args.base_root).expanduser().resolve()
    external_root = Path(args.external_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    base_manifest = load_json(base_root / "risk_dataset_post_v2_min_default_manifest.json")
    train_header = read_header(base_root / "risk_dataset_post_v2_min_default_train.csv")
    train_rows = load_csv_rows(base_root / "risk_dataset_post_v2_min_default_train.csv")
    val_rows = load_csv_rows(base_root / "risk_dataset_post_v2_min_default_val.csv")
    external_rows_all = load_csv_rows(external_root / "risk_dataset.csv")

    gated_col = gated_feature_name(float(args.gated_parallax_threshold))
    external_rows = []
    for row in external_rows_all:
        if str(row.get("sample_type", "")) != "step11":
            continue
        if str(row.get("Y_bad_v2_min_default", "")) not in ("0", "1"):
            continue
        row = dict(row)
        v = safe_float(row.get(SOURCE_COL), 0.0)
        row[gated_col] = v if v >= float(args.gated_parallax_threshold) else 0.0
        row["dataset_row_split"] = "test"
        external_rows.append(row)

    if not external_rows:
        raise RuntimeError(f"No labeled step11 external rows found in {external_root / 'risk_dataset.csv'}")

    task_header = list(train_header)
    if gated_col not in task_header:
        task_header.append(gated_col)

    package_dir = out_dir / "package"
    write_csv(package_dir / "risk_dataset_post_v2_min_default_train.csv", task_header, [{k: r.get(k, "") for k in task_header} for r in train_rows])
    write_csv(package_dir / "risk_dataset_post_v2_min_default_val.csv", task_header, [{k: r.get(k, "") for k in task_header} for r in val_rows])
    write_csv(package_dir / "risk_dataset_post_v2_min_default_test.csv", task_header, [{k: r.get(k, "") for k in task_header} for r in external_rows])
    # train_init_risk_baseline.py also expects a package-local audit board named risk_dataset.csv.
    write_csv(
        package_dir / "risk_dataset.csv",
        task_header,
        [{k: r.get(k, "") for k in task_header} for r in (train_rows + val_rows + external_rows)],
    )

    manifest = dict(base_manifest)
    allowed = list(manifest.get("allowed_feature_columns", []))
    if gated_col not in allowed:
        allowed.append(gated_col)
    manifest["allowed_feature_columns"] = allowed
    manifest["split_policy"] = "external_holdout"
    manifest["external_holdout"] = {
        "name": str(args.external_name or external_root.name),
        "base_root": str(base_root),
        "external_root": str(external_root),
        "test_rows": int(len(external_rows)),
    }
    manifest["gated_parallax_feature"] = {
        "name": gated_col,
        "source_column": SOURCE_COL,
        "activation_rule": f"{SOURCE_COL} >= {float(args.gated_parallax_threshold):.1f}",
        "inactive_value": 0.0,
    }
    write_json(package_dir / "risk_dataset_post_v2_min_default_manifest.json", manifest)
    write_json(
        package_dir / "external_holdout_manifest.json",
        {
            "base_root": str(base_root),
            "external_root": str(external_root),
            "external_name": str(args.external_name or external_root.name),
            "gated_parallax_threshold": float(args.gated_parallax_threshold),
            "num_train_rows": int(len(train_rows)),
            "num_val_rows": int(len(val_rows)),
            "num_test_rows": int(len(external_rows)),
        },
    )

    summary_rows = []
    for model_name, cols in build_core4(gated_col):
        model_dir = out_dir / "core4_runs" / model_name
        metrics = run_baseline(package_dir, cols, model_dir)
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
    summary_rows.sort(key=lambda r: float(r["auroc"]) if r["auroc"] is not None else -1.0, reverse=True)
    write_csv(out_dir / "post_v2_core4_external_holdout_summary.csv", list(summary_rows[0].keys()), summary_rows)
    write_json(
        out_dir / "post_v2_core4_external_holdout_summary.json",
        {
            "external_name": str(args.external_name or external_root.name),
            "base_root": str(base_root),
            "external_root": str(external_root),
            "rows": summary_rows,
        },
    )


if __name__ == "__main__":
    main()
