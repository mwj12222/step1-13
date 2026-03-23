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


def load_csv_rows(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_header(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return next(csv.reader(f))


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def safe_float(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def gated_feature_name(threshold: float) -> str:
    return f"parallax_px_candidate_gated_ge_{int(round(float(threshold)))}"


def build_core4(gated_col: str) -> list[tuple[str, list[str]]]:
    return [
        ("geometry_only", ["reproj_med_px", "reproj_p90_px", "cheirality_ratio"]),
        ("geometry_plus_gated_parallax", ["reproj_med_px", "reproj_p90_px", "cheirality_ratio", gated_col]),
        ("drop_front", [SOURCE_COL, "tri_points_candidate", "pnp_success_rate", "reproj_med_px", "reproj_p90_px", "cheirality_ratio"]),
        ("full_anchor", ["front_p_static", "front_p_band", "front_coverage_ratio", "front_kept_dyn_ratio", SOURCE_COL, "tri_points_candidate", "pnp_success_rate", "reproj_med_px", "reproj_p90_px", "cheirality_ratio"]),
    ]


def run_cmd(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def run_builder(template_root: Path, seed: int, out_dir: Path) -> Path:
    manifest = load_json(template_root / "dataset_manifest.json")
    input_roots = list(manifest["input_roots"])
    y_bad_v2 = manifest["y_bad_v2_min_default"]["labels_csv"]
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "sfm_init" / "build_init_risk_dataset.py"),
        "--out_dir", str(out_dir),
        "--split_mode", "strict_by_sequence",
        "--split_seed", str(seed),
        "--split_ratios", "0.70,0.15,0.15",
        "--write_task_split_csvs",
        "--y_bad_v2_min_labels_csv", str(y_bad_v2),
    ]
    for root in input_roots:
        cmd.extend(["--input_root", str(root)])
    run_cmd(cmd)
    return out_dir


def augment_with_gated_parallax(result_root: Path, threshold: float) -> str:
    feature_name = gated_feature_name(threshold)
    for name in [
        "risk_dataset.csv",
        "risk_dataset_post_v2_min_default_train.csv",
        "risk_dataset_post_v2_min_default_val.csv",
        "risk_dataset_post_v2_min_default_test.csv",
    ]:
        path = result_root / name
        rows = load_csv_rows(path)
        header = read_header(path)
        if feature_name not in header:
            header.append(feature_name)
        for row in rows:
            v = safe_float(row.get(SOURCE_COL), 0.0)
            row[feature_name] = v if v >= float(threshold) else 0.0
        write_csv(path, header, rows)

    manifest_path = result_root / "risk_dataset_post_v2_min_default_manifest.json"
    manifest = load_json(manifest_path)
    allowed = list(manifest.get("allowed_feature_columns", []))
    if feature_name not in allowed:
        allowed.append(feature_name)
    manifest["allowed_feature_columns"] = allowed
    manifest["gated_parallax_feature"] = {
        "name": feature_name,
        "source_column": SOURCE_COL,
        "activation_rule": f"{SOURCE_COL} >= {float(threshold):.1f}",
        "inactive_value": 0.0,
    }
    write_json(manifest_path, manifest)
    return feature_name


def run_baseline(result_root: Path, feature_cols: list[str], out_dir: Path) -> dict:
    manifest_path = result_root / "risk_dataset_post_v2_min_default_manifest.json"
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "sfm_init" / "train_init_risk_baseline.py"),
        "--task", "post",
        "--model", "logistic",
        "--train_csv", str(result_root / "risk_dataset_post_v2_min_default_train.csv"),
        "--val_csv", str(result_root / "risk_dataset_post_v2_min_default_val.csv"),
        "--test_csv", str(result_root / "risk_dataset_post_v2_min_default_test.csv"),
        "--manifest", str(manifest_path),
        "--label_col", "Y_bad_v2_min_default",
        "--out_dir", str(out_dir),
        "--override_feature_columns", ",".join(feature_cols),
    ]
    run_cmd(cmd)
    return load_json(out_dir / "metrics.json")


def mean(xs: list[float]) -> float | None:
    return (sum(xs) / len(xs)) if xs else None


def main():
    ap = argparse.ArgumentParser(description="Run core-4 post_v2 comparisons across multiple split seeds.")
    ap.add_argument("--template_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seeds", default="20260432,20260433,20260434")
    ap.add_argument("--gated_parallax_threshold", type=float, default=60.0)
    args = ap.parse_args()

    template_root = Path(args.template_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]
    all_rows = []
    seed_summary = []

    for seed in seeds:
        rebuild_root = out_dir / "rebuilds" / f"seed_{seed}"
        status = {"seed": seed, "status": "ok", "message": ""}
        try:
            run_builder(template_root=template_root, seed=seed, out_dir=rebuild_root)
            gated_col = augment_with_gated_parallax(rebuild_root, float(args.gated_parallax_threshold))
            core4 = build_core4(gated_col)
            test_csv = rebuild_root / "risk_dataset_post_v2_min_default_test.csv"
            val_csv = rebuild_root / "risk_dataset_post_v2_min_default_val.csv"
            train_csv = rebuild_root / "risk_dataset_post_v2_min_default_train.csv"
            if not (train_csv.exists() and val_csv.exists() and test_csv.exists()):
                raise RuntimeError("task-specific post_v2 split csvs are incomplete")

            for model_name, cols in core4:
                model_dir = out_dir / "runs" / f"seed_{seed}" / model_name
                metrics = run_baseline(rebuild_root, cols, model_dir)
                test_metrics = metrics["splits"]["test"]
                all_rows.append(
                    {
                        "seed": seed,
                        "model": model_name,
                        "num_features": len(cols),
                        "features": ",".join(cols),
                        "num_rows": test_metrics["num_rows"],
                        "positive_ratio": test_metrics["positive_ratio"],
                        "auroc": test_metrics["auroc"],
                        "auprc": test_metrics["auprc"],
                        "brier": test_metrics["brier"],
                        "ece": test_metrics["ece"],
                    }
                )
        except Exception as exc:
            status["status"] = "failed"
            status["message"] = str(exc)
        seed_summary.append(status)

    by_model = {}
    for row in all_rows:
        by_model.setdefault(row["model"], []).append(row)

    aggregate_rows = []
    for model, rows in sorted(by_model.items()):
        aurocs = [float(r["auroc"]) for r in rows]
        auprcs = [float(r["auprc"]) for r in rows]
        aggregate_rows.append(
            {
                "model": model,
                "num_successful_seeds": len(rows),
                "mean_auroc": mean(aurocs),
                "min_auroc": min(aurocs) if aurocs else None,
                "max_auroc": max(aurocs) if aurocs else None,
                "mean_auprc": mean(auprcs),
            }
        )
    aggregate_rows.sort(key=lambda r: (r["mean_auroc"] is None, -(r["mean_auroc"] or -1)))

    summary = {
        "template_root": str(template_root),
        "seeds": seeds,
        "seed_status": seed_summary,
        "aggregate_rows": aggregate_rows,
        "takeaways": [
            "This is a light split-seed stability check for the four fixed post_v2 core comparisons.",
            "The main question is whether geometry_plus_gated_parallax stays competitive with geometry_only and clearly ahead of full across rebuilt splits.",
            "Seeds that fail to produce a valid post_v2 test split are recorded rather than silently dropped.",
        ],
    }
    write_json(out_dir / "post_v2_core4_multiseed_summary.json", summary)
    if all_rows:
        write_csv(
            out_dir / "post_v2_core4_multiseed_runs.csv",
            list(all_rows[0].keys()),
            all_rows,
        )
    write_csv(
        out_dir / "post_v2_core4_multiseed_aggregate.csv",
        ["model", "num_successful_seeds", "mean_auroc", "min_auroc", "max_auroc", "mean_auprc"],
        aggregate_rows,
    )


if __name__ == "__main__":
    main()
