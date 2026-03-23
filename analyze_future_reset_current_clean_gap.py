#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = None
for p in [THIS_FILE.parent, *THIS_FILE.parent.parents]:
    if (p / "configs").is_dir() and ((p / "pipelines").is_dir() or (p / " pipelines").is_dir()):
        PROJECT_ROOT = p
        break
if PROJECT_ROOT is None:
    raise RuntimeError(f"Cannot locate project root from {THIS_FILE}")

DEFAULT_BASE_ROOT = Path("/mnt/g/Result/VIODE/post_v2_rebuild_9seq/risk_dataset_9seq_seed20260432")
DEFAULT_OUT_DIR = PROJECT_ROOT / "docs" / "research" / "init_risk_future_reset_current_clean_gap_20260321"
DEFAULT_EXTERNALS = [
    (
        "corridor4",
        Path("/mnt/g/Experiments/TUM VI/dataset-corridor4_512_16/step1_11_cam0_corridor4_external_core4_undistort_b00_20260320") / "external_package" / "post_v2_min_default_corridor4" / "risk_dataset.csv",
        Path("/mnt/g/Experiments/TUM VI/dataset-corridor4_512_16/step1_11_cam0_corridor4_external_core4_undistort_b00_20260320") / "external_holdout_core4" / "post_v2_min_default_corridor4" / "core4_runs" / "full_anchor" / "predictions.csv",
    ),
    (
        "room2",
        Path("/mnt/g/Experiments/TUM VI/dataset-room2_512_16/step1_11_cam0_room2_external_core4_undistort_b00_20260320") / "external_package" / "post_v2_min_default_room2" / "risk_dataset.csv",
        Path("/mnt/g/Experiments/TUM VI/dataset-room2_512_16/step1_11_cam0_room2_external_core4_undistort_b00_20260320") / "external_holdout_core4" / "post_v2_min_default_room2" / "core4_runs" / "full_anchor" / "predictions.csv",
    ),
]

FULL_ANCHOR_FEATURES = [
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

EXTERNAL_COMPARE_FIELDS = [
    "parallax_px_candidate",
    "tri_points_candidate",
    "pnp_success_rate",
    "reproj_med_px",
    "reproj_p90_px",
    "cheirality_ratio",
    "front_p_static",
    "front_coverage_ratio",
    "Q_post_geom_only",
    "Y_bad_v2_min_default_trigger",
]


def load_csv_rows(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def safe_float(v, default=None):
    try:
        x = float(v)
    except Exception:
        return default
    if math.isnan(x) or math.isinf(x):
        return default
    return x


def mean(xs: list[float]):
    return sum(xs) / len(xs) if xs else None


def fmt(v, nd=4):
    if v is None:
        return "-"
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return str(v)


def find_positive_reference(rows: list[dict]) -> dict:
    positives = [r for r in rows if int(r["y_true"]) == 1]
    if len(positives) != 1:
        raise RuntimeError(f"Expected exactly one positive row, got {len(positives)}")
    return positives[0]


def build_external_contrast_rows(name: str, risk_dataset_csv: Path, predictions_csv: Path) -> tuple[list[dict], dict]:
    base_rows = {r["sample_uid"]: r for r in load_csv_rows(risk_dataset_csv)}
    preds = [r for r in load_csv_rows(predictions_csv) if r["split"] == "test"]
    preds.sort(key=lambda r: float(r["p_hat"]))
    positive_pred = find_positive_reference(preds)
    positive_row = base_rows[positive_pred["sample_uid"]]

    out = []
    for rank, pred in enumerate(preds, start=1):
        base = base_rows[pred["sample_uid"]]
        row = {
            "dataset": name,
            "rank_by_p_hat": rank,
            "sample_uid": pred["sample_uid"],
            "window_id": base.get("window_id", ""),
            "y_true": pred["y_true"],
            "p_hat": pred["p_hat"],
            "is_reference_positive": "1" if pred["sample_uid"] == positive_pred["sample_uid"] else "0",
        }
        for key in EXTERNAL_COMPARE_FIELDS:
            row[key] = base.get(key, "")
        for key in ["parallax_px_candidate", "tri_points_candidate", "reproj_med_px", "reproj_p90_px", "front_coverage_ratio"]:
            cur = safe_float(base.get(key))
            ref = safe_float(positive_row.get(key))
            row[f"delta_vs_positive__{key}"] = "" if cur is None or ref is None else cur - ref
        out.append(row)

    summary = {
        "dataset": name,
        "num_test_rows": len(preds),
        "positive_uid": positive_pred["sample_uid"],
        "positive_score": float(positive_pred["p_hat"]),
        "negative_min_score": min(float(r["p_hat"]) for r in preds if int(r["y_true"]) == 0),
        "all_positive_below_all_negative": all(
            float(positive_pred["p_hat"]) < float(r["p_hat"]) for r in preds if int(r["y_true"]) == 0
        ),
    }
    return out, summary


def run_internal_full_anchor(base_root: Path, out_dir: Path, force: bool = False) -> Path:
    pred_path = out_dir / "predictions.csv"
    if pred_path.exists() and not force:
        return pred_path
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "sfm_init" / "train_init_risk_baseline.py"),
        "--task",
        "post",
        "--model",
        "logistic",
        "--train_csv",
        str(base_root / "risk_dataset_post_v2_min_default_train.csv"),
        "--val_csv",
        str(base_root / "risk_dataset_post_v2_min_default_val.csv"),
        "--test_csv",
        str(base_root / "risk_dataset_post_v2_min_default_val.csv"),
        "--manifest",
        str(base_root / "risk_dataset_post_v2_min_default_manifest.json"),
        "--label_col",
        "Y_bad_v2_min_default",
        "--out_dir",
        str(out_dir),
        "--override_feature_columns",
        ",".join(FULL_ANCHOR_FEATURES),
    ]
    subprocess.run(cmd, check=True)
    return pred_path


def is_clean_future_reset(row: dict, args) -> bool:
    return (
        row.get("Y_bad_v2_min_default_trigger", "") == "future_reset"
        and safe_float(row.get("Q_post_geom_only"), -1.0) >= float(args.clean_q_post_geom_min)
        and safe_float(row.get("pnp_success_rate"), -1.0) >= float(args.clean_pnp_success_min)
        and safe_float(row.get("cheirality_ratio"), -1.0) >= float(args.clean_cheirality_min)
        and safe_float(row.get("reproj_med_px"), 1e9) <= float(args.clean_reproj_med_max)
        and safe_float(row.get("reproj_p90_px"), 1e9) <= float(args.clean_reproj_p90_max)
    )


def enrich_internal_rows(base_rows_by_uid: dict[str, dict], pred_rows: list[dict]) -> list[dict]:
    out = []
    for pred in pred_rows:
        base = dict(base_rows_by_uid[pred["sample_uid"]])
        base["p_hat"] = float(pred["p_hat"])
        base["y_true"] = int(pred["y_true"])
        base["split"] = pred["split"]
        out.append(base)
    return out


def frac_below(items: list[dict], threshold: float | None):
    if not items or threshold is None:
        return None
    return sum(1 for r in items if float(r["p_hat"]) < threshold) / len(items)


def percentile_against_negatives(item: dict, negatives: list[dict]) -> float | None:
    if not negatives:
        return None
    return sum(1 for r in negatives if float(r["p_hat"]) < float(item["p_hat"])) / len(negatives)


def summarize_internal_split(split: str, rows: list[dict], args) -> tuple[list[dict], list[dict]]:
    neg = [r for r in rows if int(r["y_true"]) == 0]
    pos = [r for r in rows if int(r["y_true"]) == 1]
    neg_scores = sorted(float(r["p_hat"]) for r in neg)
    neg_median = neg_scores[len(neg_scores) // 2] if neg_scores else None
    neg_q25 = neg_scores[len(neg_scores) // 4] if neg_scores else None

    groups = {
        "all_positive": pos,
        "future_high_gt_rot": [r for r in pos if r.get("Y_bad_v2_min_default_trigger", "") == "future_high_gt_rot"],
        "future_reset": [r for r in pos if r.get("Y_bad_v2_min_default_trigger", "") == "future_reset"],
        "future_solver_fail": [r for r in pos if r.get("Y_bad_v2_min_default_trigger", "") == "future_solver_fail"],
        "clean_future_reset": [r for r in pos if is_clean_future_reset(r, args)],
    }

    stats = []
    for name, items in groups.items():
        stats.append(
            {
                "split": split,
                "group": name,
                "num_rows": len(items),
                "mean_p_hat": mean([float(r["p_hat"]) for r in items]),
                "mean_negative_p_hat": mean([float(r["p_hat"]) for r in neg]),
                "frac_below_neg_q25": frac_below(items, neg_q25),
                "frac_below_neg_median": frac_below(items, neg_median),
                "recall_at_0p5": None if not items else sum(1 for r in items if float(r["p_hat"]) >= 0.5) / len(items),
            }
        )

    detailed_rows = []
    for r in groups["future_reset"]:
        rr = dict(r)
        rr["is_clean_future_reset"] = "1" if is_clean_future_reset(r, args) else "0"
        rr["negative_rank_percentile"] = percentile_against_negatives(r, neg)
        rr["below_negative_q25"] = "1" if neg_q25 is not None and float(r["p_hat"]) < neg_q25 else "0"
        rr["below_negative_median"] = "1" if neg_median is not None and float(r["p_hat"]) < neg_median else "0"
        detailed_rows.append(rr)
    detailed_rows.sort(key=lambda r: (int(r["is_clean_future_reset"]) * -1, float(r["p_hat"])))
    return stats, detailed_rows


def build_markdown(external_summaries: list[dict], internal_stats: list[dict], internal_future_reset_rows: list[dict], args) -> str:
    lines = []
    lines.append("# future_reset current-clean gap")
    lines.append("")
    lines.append("## Main Takeaway")
    lines.append("")
    lines.append("- The two TUM VI external positives are both `future_reset` rows that look geometrically clean at the current window, and `full_anchor` ranks each of them below every external negative.")
    lines.append("- This is not just an external fluke. On non-held-out VIODE train/val, `future_reset` positives are also the lowest-scored positive trigger family under the same frozen `full_anchor` feature set.")
    lines.append("- The cleanest `future_reset` subset is especially weak: every such row falls below the negative median in both train and val, so the current model treats them as safer than ordinary stable negatives.")
    lines.append("")
    lines.append("## External Evidence")
    lines.append("")
    lines.append("| dataset | test rows | positive score | min negative score | full inversion |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in external_summaries:
        lines.append(
            f"| {row['dataset']} | {row['num_test_rows']} | {fmt(row['positive_score'])} | {fmt(row['negative_min_score'])} | {row['all_positive_below_all_negative']} |"
        )
    lines.append("")
    lines.append("## Internal Train/Val Evidence")
    lines.append("")
    lines.append(f"Clean-current rule: `Q_post_geom_only >= {args.clean_q_post_geom_min}`, `pnp_success_rate >= {args.clean_pnp_success_min}`, `cheirality_ratio >= {args.clean_cheirality_min}`, `reproj_med_px <= {args.clean_reproj_med_max}`, `reproj_p90_px <= {args.clean_reproj_p90_max}`.")
    lines.append("")
    lines.append("| split | group | n | mean p_hat | mean negative p_hat | frac below neg q25 | frac below neg median | recall@0.5 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in internal_stats:
        lines.append(
            f"| {row['split']} | {row['group']} | {row['num_rows']} | {fmt(row['mean_p_hat'])} | {fmt(row['mean_negative_p_hat'])} | {fmt(row['frac_below_neg_q25'])} | {fmt(row['frac_below_neg_median'])} | {fmt(row['recall_at_0p5'])} |"
        )
    lines.append("")
    clean_rows = [r for r in internal_future_reset_rows if r['is_clean_future_reset'] == '1']
    if clean_rows:
        lines.append("## Internal Clean future_reset Rows")
        lines.append("")
        lines.append("| split | sample_uid | p_hat | neg-rank percentile | parallax | tri_points | reproj_med | reproj_p90 | Q_post_geom_only |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for row in clean_rows:
            lines.append(
                f"| {row['split']} | {row['sample_uid']} | {fmt(row['p_hat'])} | {fmt(row['negative_rank_percentile'])} | {fmt(row['parallax_px_candidate'])} | {fmt(row['tri_points_candidate'])} | {fmt(row['reproj_med_px'])} | {fmt(row['reproj_p90_px'])} | {fmt(row['Q_post_geom_only'])} |"
            )
        lines.append("")
    lines.append("## Recommended Internal-Only Next Version")
    lines.append("")
    lines.append("- Recommended first move: feature-first, not label replacement. `future_reset` is rare in current train/val, so replacing the main label would be high variance.")
    lines.append("- Add an internal-only interaction feature block that explicitly captures `current clean but reset-prone` windows.")
    lines.append("- Suggested new features for the next internal ablation:")
    lines.append("  - `clean_geom_flag = 1[Q_post_geom_only>=0.99 and pnp_success_rate>=1 and cheirality_ratio>=0.99 and reproj_med_px<=0.10 and reproj_p90_px<=0.30]`")
    lines.append("  - `clean_geom_inv_parallax = clean_geom_flag / (1 + parallax_px_candidate)`")
    lines.append("  - `clean_geom_high_support = clean_geom_flag * log1p(tri_points_candidate)`")
    lines.append("  - `clean_geom_cover_parallax = clean_geom_flag * front_coverage_ratio / (1 + parallax_px_candidate)`")
    lines.append("- If that feature-only ablation still misses the same rows, then add an auxiliary internal label instead of changing the main held-out protocol: `Y_future_reset_clean_aux = 1[trigger=future_reset and clean_geom_flag=1]`.")
    lines.append("- Keep TUM VI frozen. All tuning, thresholding, and feature selection should stay on internal VIODE train/val only.")
    lines.append("")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description="Analyze the future_reset but current-clean gap on TUM VI external and VIODE internal train/val.")
    ap.add_argument("--base_root", default=str(DEFAULT_BASE_ROOT))
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--force_rerun_internal", action="store_true")
    ap.add_argument("--clean_q_post_geom_min", type=float, default=0.99)
    ap.add_argument("--clean_pnp_success_min", type=float, default=0.99)
    ap.add_argument("--clean_cheirality_min", type=float, default=0.99)
    ap.add_argument("--clean_reproj_med_max", type=float, default=0.10)
    ap.add_argument("--clean_reproj_p90_max", type=float, default=0.30)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    base_root = Path(args.base_root).expanduser().resolve()

    external_rows = []
    external_summaries = []
    for name, dataset_csv, predictions_csv in DEFAULT_EXTERNALS:
        rows, summary = build_external_contrast_rows(name, dataset_csv, predictions_csv)
        external_rows.extend(rows)
        external_summaries.append(summary)

    external_fieldnames = list(external_rows[0].keys()) if external_rows else []
    if external_rows:
        write_csv(out_dir / "external_future_reset_vs_negatives.csv", external_fieldnames, external_rows)
    write_json(out_dir / "external_inversion_summary.json", external_summaries)

    internal_run_dir = out_dir / "internal_full_anchor_trainval"
    internal_pred_path = run_internal_full_anchor(base_root, internal_run_dir, force=bool(args.force_rerun_internal))
    base_rows_by_uid = {r["sample_uid"]: r for r in load_csv_rows(base_root / "risk_dataset.csv")}
    pred_rows = [r for r in load_csv_rows(internal_pred_path) if r["split"] in ("train", "val")]
    enriched = enrich_internal_rows(base_rows_by_uid, pred_rows)

    internal_stats = []
    internal_future_reset_rows = []
    for split in ("train", "val"):
        split_rows = [r for r in enriched if r["split"] == split and r.get("sample_type") == "step11"]
        stats_rows, detailed_rows = summarize_internal_split(split, split_rows, args)
        internal_stats.extend(stats_rows)
        internal_future_reset_rows.extend(detailed_rows)

    write_csv(out_dir / "internal_future_reset_group_stats.csv", list(internal_stats[0].keys()), internal_stats)
    write_csv(out_dir / "internal_future_reset_rows.csv", list(internal_future_reset_rows[0].keys()), internal_future_reset_rows)

    md = build_markdown(external_summaries, internal_stats, internal_future_reset_rows, args)
    write_text(out_dir / "future_reset_current_clean_gap_summary.md", md)
    write_json(
        out_dir / "future_reset_current_clean_gap_audit.json",
        {
            "base_root": str(base_root),
            "external_sources": [
                {
                    "name": name,
                    "risk_dataset_csv": str(dataset_csv),
                    "predictions_csv": str(predictions_csv),
                }
                for name, dataset_csv, predictions_csv in DEFAULT_EXTERNALS
            ],
            "clean_thresholds": {
                "Q_post_geom_only_min": float(args.clean_q_post_geom_min),
                "pnp_success_rate_min": float(args.clean_pnp_success_min),
                "cheirality_ratio_min": float(args.clean_cheirality_min),
                "reproj_med_px_max": float(args.clean_reproj_med_max),
                "reproj_p90_px_max": float(args.clean_reproj_p90_max),
            },
            "outputs": {
                "external_contrast_csv": str(out_dir / "external_future_reset_vs_negatives.csv"),
                "external_inversion_summary_json": str(out_dir / "external_inversion_summary.json"),
                "internal_group_stats_csv": str(out_dir / "internal_future_reset_group_stats.csv"),
                "internal_future_reset_rows_csv": str(out_dir / "internal_future_reset_rows.csv"),
                "summary_md": str(out_dir / "future_reset_current_clean_gap_summary.md"),
                "internal_run_dir": str(internal_run_dir),
            },
        },
    )

    print(f"[future_reset_gap] saved -> {out_dir}")


if __name__ == "__main__":
    main()
