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


FRONT_COLS = [
    "front_p_static",
    "front_p_band",
    "front_coverage_ratio",
    "front_kept_dyn_ratio",
]

CANDIDATE_COLS = [
    "parallax_px_candidate",
    "tri_points_candidate",
    "pnp_success_rate",
]

GEOMETRY_COLS = [
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


def run_baseline(train_csv: Path, val_csv: Path, test_csv: Path, manifest: Path, label_col: str, out_dir: Path, override_cols: list[str]):
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
    ap = argparse.ArgumentParser(description="Run minimal post_v2_min_default ablations by feature block.")
    ap.add_argument("--result_root", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    root = Path(args.result_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_json(root / "risk_dataset_post_v2_min_default_manifest.json")
    allowed = set(manifest["allowed_feature_columns"])
    train_csv = root / "risk_dataset_post_v2_min_default_train.csv"
    val_csv = root / "risk_dataset_post_v2_min_default_val.csv"
    test_csv = root / "risk_dataset_post_v2_min_default_test.csv"
    label_col = "Y_bad_v2_min_default"

    front = [c for c in FRONT_COLS if c in allowed]
    candidate = [c for c in CANDIDATE_COLS if c in allowed]
    geometry = [c for c in GEOMETRY_COLS if c in allowed]

    ablations = [
        ("post_v2_full", sorted(front + candidate + geometry), "Full post_v2 baseline feature set."),
        ("post_v2_geometry_only", sorted(geometry), "Only posterior geometry features."),
        ("post_v2_candidate_only", sorted(candidate), "Only candidate-stage observability features."),
        ("post_v2_front_only", sorted(front), "Only front-end static support features."),
        ("post_v2_drop_front", sorted(candidate + geometry), "Drop front block; keep candidate + geometry."),
        ("post_v2_drop_candidate", sorted(front + geometry), "Drop candidate block; keep front + geometry."),
        ("post_v2_drop_geometry", sorted(front + candidate), "Drop geometry block; keep front + candidate."),
    ]

    rows = []
    for name, cols, note in ablations:
        model_dir = out_dir / name
        metrics = run_baseline(train_csv, val_csv, test_csv, root / "risk_dataset_post_v2_min_default_manifest.json", label_col, model_dir, cols)
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

    summary = {
        "task": "post_v2_min_default",
        "label_version": label_col,
        "label_scope": manifest.get("label_scope"),
        "rows": rows,
        "takeaways": [
            "Compare geometry-only against full to test whether upgraded labels still reduce to a pure geometry acceptor.",
            "Compare drop-front and drop-candidate to estimate whether front or candidate blocks still contribute under the stronger label.",
            "Compare drop-geometry and candidate/front-only runs to test whether post_v2_min_default remains learnable without posterior geometry.",
        ],
    }
    write_json(out_dir / "post_v2_ablation_summary.json", summary)
    write_csv(
        out_dir / "post_v2_ablation_summary.csv",
        ["name", "num_features", "features", "auroc", "auprc", "brier", "ece", "positive_ratio", "num_rows", "note"],
        rows,
    )
    print(f"[PostV2Ablation] saved -> {out_dir}")


if __name__ == "__main__":
    main()
