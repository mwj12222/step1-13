#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import math
import random


def load_csv_rows(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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


def safe_float(v, default=None):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    if math.isnan(x):
        return default
    return x


def compute_auroc(y_true: list[int], y_score: list[float]):
    pos = sum(1 for y in y_true if y == 1)
    neg = sum(1 for y in y_true if y == 0)
    if pos == 0 or neg == 0:
        return None
    pairs = sorted(zip(y_score, y_true))
    rank_sum = 0.0
    i = 0
    n = len(pairs)
    while i < n:
        j = i + 1
        while j < n and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) * 0.5
        rank_sum += avg_rank * sum(y for _, y in pairs[i:j])
        i = j
    return float((rank_sum - pos * (pos + 1) * 0.5) / (pos * neg))


def bootstrap_auroc_ci(pred_rows: list[dict], n_boot=2000, seed=20260313):
    y = [int(float(r["y_true"])) for r in pred_rows]
    p = [float(r["p_hat"]) for r in pred_rows]
    if len(set(y)) < 2:
        return {"n_boot": int(n_boot), "used_boot": 0, "auroc_p50": None, "auroc_p05": None, "auroc_p95": None}
    rng = random.Random(int(seed))
    vals = []
    n = len(pred_rows)
    for _ in range(int(n_boot)):
        idx = [rng.randrange(n) for _ in range(n)]
        y_b = [y[i] for i in idx]
        p_b = [p[i] for i in idx]
        auc = compute_auroc(y_b, p_b)
        if auc is not None:
            vals.append(auc)
    vals.sort()
    if not vals:
        return {"n_boot": int(n_boot), "used_boot": 0, "auroc_p50": None, "auroc_p05": None, "auroc_p95": None}
    def pct(q):
        k = min(len(vals) - 1, max(0, int(round(q * (len(vals) - 1)))))
        return float(vals[k])
    return {
        "n_boot": int(n_boot),
        "used_boot": len(vals),
        "auroc_p50": pct(0.50),
        "auroc_p05": pct(0.05),
        "auroc_p95": pct(0.95),
    }


def build_v2_only_rows(risk_rows: list[dict]) -> list[dict]:
    out = []
    for row in risk_rows:
        if row.get("sample_type") != "step11":
            continue
        if str(row.get("Y_bad_v2_min_default", "")) not in ("0", "1"):
            continue
        y1 = int(float(row.get("Y_bad_v1", 0)))
        y2 = int(float(row.get("Y_bad_v2_min_default", 0)))
        if y1 == 0 and y2 == 1:
            out.append(row)
    return out


def join_prediction_table(v2_only_rows: list[dict], model_prediction_paths: dict[str, Path]) -> list[dict]:
    pred_maps = {}
    for model_name, path in model_prediction_paths.items():
        pred_maps[model_name] = {row["sample_uid"]: row for row in load_csv_rows(path)}
    out = []
    for row in v2_only_rows:
        sample_uid = row["sample_uid"]
        joined = {
            "sample_uid": sample_uid,
            "sequence": row.get("sequence", ""),
            "window_id": row.get("window_id", ""),
            "dataset_row_split": row.get("dataset_row_split", ""),
            "Y_bad_v1": row.get("Y_bad_v1", ""),
            "Y_bad_v2_min_default": row.get("Y_bad_v2_min_default", ""),
            "Y_bad_v2_min_default_trigger": row.get("Y_bad_v2_min_default_trigger", ""),
        }
        for model_name, pred_map in pred_maps.items():
            pred = pred_map.get(sample_uid, {})
            prob = safe_float(pred.get("p_hat"))
            y_true = pred.get("y_true", "")
            joined[f"{model_name}_p_hat"] = prob
            joined[f"{model_name}_split"] = pred.get("split", "")
            joined[f"{model_name}_correct_at_0p5"] = (
                int((prob >= 0.5) == (int(float(y_true)) == 1)) if prob is not None and str(y_true) not in ("", None) else ""
            )
        out.append(joined)
    return out


def build_v2_only_summary(rows: list[dict], model_names: list[str]) -> list[dict]:
    out = []
    for model_name in model_names:
        correct = [int(r[f"{model_name}_correct_at_0p5"]) for r in rows if str(r.get(f"{model_name}_correct_at_0p5", "")) in ("0", "1")]
        out.append(
            {
                "model": model_name,
                "num_v2_only_rows": len(rows),
                "num_correct_at_0p5": sum(correct),
                "num_wrong_at_0p5": len(correct) - sum(correct),
                "accuracy_at_0p5": float(sum(correct) / max(1, len(correct))) if correct else None,
            }
        )
    return out


def main():
    ap = argparse.ArgumentParser(description="Analyze post_v2_min_default ablations with v2-only decomposition and bootstrap AUROC intervals.")
    ap.add_argument("--result_root", required=True)
    ap.add_argument("--ablation_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--bootstrap_runs", type=int, default=2000)
    ap.add_argument("--bootstrap_seed", type=int, default=20260313)
    args = ap.parse_args()

    result_root = Path(args.result_root).expanduser().resolve()
    ablation_root = Path(args.ablation_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    risk_rows = load_csv_rows(result_root / "risk_dataset.csv")
    v2_only_rows = build_v2_only_rows(risk_rows)

    tracked_models = {
        "full": ablation_root / "post_v2_full" / "predictions.csv",
        "geometry_only": ablation_root / "post_v2_geometry_only" / "predictions.csv",
        "drop_candidate": ablation_root / "post_v2_drop_candidate" / "predictions.csv",
    }
    v2_only_table = join_prediction_table(v2_only_rows, tracked_models)
    v2_only_summary = build_v2_only_summary(v2_only_table, list(tracked_models.keys()))

    ablation_summary = load_json(ablation_root / "post_v2_ablation_summary.json")
    ci_rows = []
    for row in ablation_summary["rows"]:
        pred_rows = [r for r in load_csv_rows(ablation_root / row["name"] / "predictions.csv") if r.get("split") == "test"]
        ci = bootstrap_auroc_ci(pred_rows, n_boot=args.bootstrap_runs, seed=args.bootstrap_seed)
        ci_rows.append(
            {
                "name": row["name"],
                "auroc": row["auroc"],
                "auroc_p05": ci["auroc_p05"],
                "auroc_p50": ci["auroc_p50"],
                "auroc_p95": ci["auroc_p95"],
                "used_boot": ci["used_boot"],
                "num_test_rows": len(pred_rows),
            }
        )

    summary = {
        "v2_only_count": len(v2_only_rows),
        "v2_only_trigger_distribution": {},
        "v2_only_summary": v2_only_summary,
        "bootstrap_auroc_ci": ci_rows,
        "takeaways": [
            "The v2-only subset captures samples newly introduced by the stronger post-accept short-horizon instability label.",
            "Comparing full, geometry_only, and drop_candidate on v2-only rows helps determine whether candidate block adds useful corrections beyond posterior geometry alone.",
            "Bootstrap AUROC intervals provide a minimal uncertainty view over the 7 ablation runs under the current tiny test split.",
        ],
    }
    trig = {}
    for row in v2_only_rows:
        t = str(row.get("Y_bad_v2_min_default_trigger", ""))
        trig[t] = trig.get(t, 0) + 1
    summary["v2_only_trigger_distribution"] = trig

    write_csv(
        out_dir / "post_v2_only_decomposition.csv",
        sorted({k for row in v2_only_table for k in row.keys()}),
        v2_only_table,
    )
    write_csv(
        out_dir / "post_v2_only_model_summary.csv",
        ["model", "num_v2_only_rows", "num_correct_at_0p5", "num_wrong_at_0p5", "accuracy_at_0p5"],
        v2_only_summary,
    )
    write_csv(
        out_dir / "post_v2_ablation_bootstrap_ci.csv",
        ["name", "auroc", "auroc_p05", "auroc_p50", "auroc_p95", "used_boot", "num_test_rows"],
        ci_rows,
    )
    write_json(out_dir / "post_v2_ablation_analysis.json", summary)
    print(f"[PostV2AblationAnalysis] saved -> {out_dir}")


if __name__ == "__main__":
    main()
