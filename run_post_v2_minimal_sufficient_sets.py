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


FRONT_SINGLE_COLS = [
    "front_p_static",
    "front_coverage_ratio",
]

CANDIDATE_SINGLE_COLS = [
    "parallax_px_candidate",
    "tri_points_candidate",
    "pnp_success_rate",
]

GEOMETRY_COLS = [
    "reproj_med_px",
    "reproj_p90_px",
    "cheirality_ratio",
]

ANCHOR_FULL_COLS = [
    "front_p_static",
    "front_p_band",
    "front_coverage_ratio",
    "front_kept_dyn_ratio",
    "parallax_px_candidate",
    "tri_points_candidate",
    "pnp_success_rate",
    "reproj_med_px",
    "reproj_p90_px",
    "cheirality_ratio",
]

ANCHOR_DROP_FRONT_COLS = [
    "parallax_px_candidate",
    "tri_points_candidate",
    "pnp_success_rate",
    "reproj_med_px",
    "reproj_p90_px",
    "cheirality_ratio",
]


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


def run_baseline(
    train_csv: Path,
    val_csv: Path,
    test_csv: Path,
    manifest: Path,
    label_col: str,
    out_dir: Path,
    override_cols: list[str],
):
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "sfm_init" / "train_init_risk_baseline.py"),
        "--task", "post",
        "--model", "logistic",
        "--train_csv", str(train_csv),
        "--val_csv", str(val_csv),
        "--test_csv", str(test_csv),
        "--manifest", str(manifest),
        "--label_col", label_col,
        "--out_dir", str(out_dir),
        "--override_feature_columns", ",".join(override_cols),
    ]
    subprocess.run(cmd, check=True)
    return load_json(out_dir / "metrics.json")


def main():
    ap = argparse.ArgumentParser(
        description="Run minimal-sufficient post_v2 feature-set comparisons around geometry + single add-on."
    )
    ap.add_argument("--result_root", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    root = Path(args.result_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = root / "risk_dataset_post_v2_min_default_manifest.json"
    manifest = load_json(manifest_path)
    allowed = set(manifest["allowed_feature_columns"])
    train_csv = root / "risk_dataset_post_v2_min_default_train.csv"
    val_csv = root / "risk_dataset_post_v2_min_default_val.csv"
    test_csv = root / "risk_dataset_post_v2_min_default_test.csv"
    label_col = "Y_bad_v2_min_default"

    geometry = [c for c in GEOMETRY_COLS if c in allowed]
    if len(geometry) != len(GEOMETRY_COLS):
        raise ValueError(f"Missing geometry columns from manifest: expected {GEOMETRY_COLS}, got {geometry}")

    experiments = [
        ("post_v2_geometry_only", geometry, "Geometry-only anchor."),
        ("post_v2_geometry_plus_parallax", geometry + [c for c in ["parallax_px_candidate"] if c in allowed], "Geometry plus best candidate separation signal by current test analysis."),
        ("post_v2_geometry_plus_tri_points", geometry + [c for c in ["tri_points_candidate"] if c in allowed], "Geometry plus triangulation-count style candidate support."),
        ("post_v2_geometry_plus_pnp_success", geometry + [c for c in ["pnp_success_rate"] if c in allowed], "Geometry plus candidate-stage PnP support signal."),
        ("post_v2_geometry_plus_front_p_static", geometry + [c for c in ["front_p_static"] if c in allowed], "Geometry plus strongest current front single feature."),
        ("post_v2_geometry_plus_front_coverage", geometry + [c for c in ["front_coverage_ratio"] if c in allowed], "Geometry plus front coverage support."),
        ("post_v2_drop_front_anchor", [c for c in ANCHOR_DROP_FRONT_COLS if c in allowed], "Current best anchor from block ablations."),
        ("post_v2_full_anchor", [c for c in ANCHOR_FULL_COLS if c in allowed], "Current full feature-mixing anchor."),
    ]

    rows = []
    for name, cols, note in experiments:
        cols = list(dict.fromkeys(cols))
        model_dir = out_dir / name
        metrics = run_baseline(
            train_csv=train_csv,
            val_csv=val_csv,
            test_csv=test_csv,
            manifest=manifest_path,
            label_col=label_col,
            out_dir=model_dir,
            override_cols=cols,
        )
        test_metrics = metrics["splits"]["test"]
        rows.append(
            {
                "name": name,
                "num_features": len(cols),
                "features": ",".join(cols),
                "auroc": test_metrics["auroc"],
                "auprc": test_metrics["auprc"],
                "brier": test_metrics["brier"],
                "ece": test_metrics["ece"],
                "positive_ratio": test_metrics["positive_ratio"],
                "num_rows": test_metrics["num_rows"],
                "note": note,
            }
        )

    rows.sort(key=lambda r: float(r["auroc"]), reverse=True)
    summary = {
        "task": "post_v2_min_default_minimal_sufficient_sets",
        "label_version": label_col,
        "label_scope": manifest.get("label_scope"),
        "rows": rows,
        "takeaways": [
            "Start from geometry-only, then test whether a single candidate or single front feature gives a stable additive gain.",
            "Keep full and drop-front only as anchors, not as default preferred formulations.",
            "Use this suite to search for a minimal sufficient acceptor recipe before revisiting richer block mixing.",
        ],
    }
    write_json(out_dir / "post_v2_minimal_sufficient_summary.json", summary)
    write_csv(
        out_dir / "post_v2_minimal_sufficient_summary.csv",
        ["name", "num_features", "features", "auroc", "auprc", "brier", "ece", "positive_ratio", "num_rows", "note"],
        rows,
    )
    print(f"[PostV2MinimalSets] saved -> {out_dir}")


if __name__ == "__main__":
    main()
