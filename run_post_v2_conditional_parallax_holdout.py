#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import math
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


GEOMETRY_COLS = [
    "reproj_med_px",
    "reproj_p90_px",
    "cheirality_ratio",
]

SOURCE_COL = "parallax_px_candidate"


def load_csv_rows(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_csv_header(path: Path) -> list[str]:
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


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def safe_float(v, default=0.0):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float(default)
    if math.isnan(x) or math.isinf(x):
        return float(default)
    return float(x)


def mean(xs: list[float]):
    return sum(xs) / len(xs) if xs else None


def fmt(v, nd=4):
    if v is None:
        return "-"
    return f"{float(v):.{nd}f}"


def aligned_score(y_true: int, p_hat: float) -> float:
    return float(p_hat) if int(y_true) == 1 else float(1.0 - float(p_hat))


def prediction_map(path: Path) -> dict[str, dict]:
    rows = load_csv_rows(path)
    return {str(r["sample_uid"]): r for r in rows if str(r.get("split", "")) == "test"}


def contiguous_segments(rows: list[dict], flag_key: str) -> list[dict]:
    flagged = [r for r in rows if int(r[flag_key]) == 1]
    flagged.sort(key=lambda r: int(r["window_id"]))
    segments = []
    current = []
    for row in flagged:
        if not current:
            current = [row]
            continue
        if int(row["window_id"]) - int(current[-1]["window_id"]) == 10:
            current.append(row)
        else:
            segments.append(current)
            current = [row]
    if current:
        segments.append(current)
    out = []
    for seg in segments:
        out.append(
            {
                "start_window_id": int(seg[0]["window_id"]),
                "end_window_id": int(seg[-1]["window_id"]),
                "num_rows": len(seg),
                "num_positive_labels": sum(int(r["y_true"]) for r in seg),
                "mean_parallax": mean([safe_float(r.get(SOURCE_COL)) for r in seg]),
                "mean_geo_p": mean([safe_float(r.get("p_geometry_only")) for r in seg]),
                "mean_gp_p": mean([safe_float(r.get("p_geometry_plus_parallax")) for r in seg]),
                "mean_gcp_p": mean([safe_float(r.get("p_conditional_parallax")) for r in seg]),
                "mean_cond_vs_geo_delta": mean([safe_float(r.get("cond_vs_geo_aligned_delta")) for r in seg]),
                "mean_cond_vs_gp_delta": mean([safe_float(r.get("cond_vs_gp_aligned_delta")) for r in seg]),
            }
        )
    out.sort(key=lambda r: abs(float(r["mean_cond_vs_gp_delta"])), reverse=True)
    return out


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


def materialize_package(base_root: Path, threshold: float, out_dir: Path) -> tuple[Path, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_name = f"parallax_px_candidate_cond_ge_{int(round(threshold))}"
    all_src = base_root / "risk_dataset.csv"
    all_rows = load_csv_rows(all_src)
    all_header = load_csv_header(all_src)
    if feature_name not in all_header:
        all_header = all_header + [feature_name]
    for row in all_rows:
        v = safe_float(row.get(SOURCE_COL), 0.0)
        row[feature_name] = v if v >= float(threshold) else 0.0
    write_csv(out_dir / "risk_dataset.csv", all_header, all_rows)

    for split in ["train", "val", "test"]:
        src = base_root / f"risk_dataset_post_v2_min_default_{split}.csv"
        rows = load_csv_rows(src)
        header = load_csv_header(src)
        if feature_name not in header:
            header = header + [feature_name]
        for row in rows:
            v = safe_float(row.get(SOURCE_COL), 0.0)
            row[feature_name] = v if v >= float(threshold) else 0.0
        write_csv(out_dir / src.name, header, rows)

    manifest = load_json(base_root / "risk_dataset_post_v2_min_default_manifest.json")
    allowed = list(manifest.get("allowed_feature_columns", []))
    if feature_name not in allowed:
        allowed.append(feature_name)
    manifest["allowed_feature_columns"] = allowed
    manifest["conditional_parallax_feature"] = {
        "name": feature_name,
        "source_column": SOURCE_COL,
        "activation_rule": f"{SOURCE_COL} >= {float(threshold):.1f}",
        "inactive_value": 0.0,
    }
    write_json(out_dir / "risk_dataset_post_v2_min_default_manifest.json", manifest)
    return out_dir, feature_name


def analyze_threshold(base_root: Path, package_dir: Path, model_dir: Path, feature_name: str) -> dict:
    test_rows = load_csv_rows(package_dir / "risk_dataset_post_v2_min_default_test.csv")
    by_uid = {str(r["sample_uid"]): r for r in test_rows}

    geo_map = prediction_map(base_root / "core4_runs" / "geometry_only" / "predictions.csv")
    gp_map = prediction_map(base_root / "core4_runs" / "geometry_plus_parallax" / "predictions.csv")
    cond_map = prediction_map(model_dir / "predictions.csv")

    merged = []
    sample_ids = sorted(
        set(geo_map.keys()) & set(gp_map.keys()) & set(cond_map.keys()),
        key=lambda uid: int(by_uid[uid]["window_id"]),
    )
    for uid in sample_ids:
        row = by_uid[uid]
        y = int(float(row["Y_bad_v2_min_default"]))
        p_geo = float(geo_map[uid]["p_hat"])
        p_gp = float(gp_map[uid]["p_hat"])
        p_cond = float(cond_map[uid]["p_hat"])
        geo_align = aligned_score(y, p_geo)
        gp_align = aligned_score(y, p_gp)
        cond_align = aligned_score(y, p_cond)
        merged.append(
            {
                "sample_uid": uid,
                "sequence": row["sequence"],
                "window_id": int(row["window_id"]),
                "y_true": y,
                SOURCE_COL: safe_float(row.get(SOURCE_COL), 0.0),
                feature_name: safe_float(row.get(feature_name), 0.0),
                "p_geometry_only": p_geo,
                "p_geometry_plus_parallax": p_gp,
                "p_conditional_parallax": p_cond,
                "geo_aligned": geo_align,
                "gp_aligned": gp_align,
                "cond_aligned": cond_align,
                "cond_vs_geo_aligned_delta": cond_align - geo_align,
                "cond_vs_gp_aligned_delta": cond_align - gp_align,
                "help_preserved": int((gp_align > geo_align) and (cond_align > geo_align)),
                "help_lost": int((gp_align > geo_align) and not (cond_align > geo_align)),
                "hurt_avoided": int((gp_align < geo_align) and (cond_align >= geo_align)),
                "hurt_persisted": int((gp_align < geo_align) and (cond_align < geo_align)),
                "new_hurt_vs_geo": int((gp_align >= geo_align) and (cond_align < geo_align)),
            }
        )

    return {
        "merged_rows": merged,
        "help_preserved_segments": contiguous_segments(merged, "help_preserved"),
        "hurt_avoided_segments": contiguous_segments(merged, "hurt_avoided"),
        "summary": {
            "num_rows": len(merged),
            "help_total": sum(1 for r in merged if r["gp_aligned"] > r["geo_aligned"]),
            "hurt_total": sum(1 for r in merged if r["gp_aligned"] < r["geo_aligned"]),
            "help_preserved_count": sum(int(r["help_preserved"]) for r in merged),
            "help_lost_count": sum(int(r["help_lost"]) for r in merged),
            "hurt_avoided_count": sum(int(r["hurt_avoided"]) for r in merged),
            "hurt_persisted_count": sum(int(r["hurt_persisted"]) for r in merged),
            "new_hurt_vs_geo_count": sum(int(r["new_hurt_vs_geo"]) for r in merged),
            "mean_cond_vs_geo_aligned_delta": mean([float(r["cond_vs_geo_aligned_delta"]) for r in merged]),
            "mean_cond_vs_gp_aligned_delta": mean([float(r["cond_vs_gp_aligned_delta"]) for r in merged]),
        },
    }


def build_markdown(payload: dict) -> str:
    lines = []
    lines.append("# conditional parallax holdout validation")
    lines.append("")
    lines.append("## 结论先行")
    lines.append("")
    for item in payload["headline"]:
        lines.append(f"- {item}")
    for seq in payload["sequences"]:
        lines.append("")
        lines.append(f"## {seq['test_sequence']}")
        lines.append("")
        lines.append(f"- 选中的条件阈值: `parallax >= {seq['selected_threshold']}`")
        lines.append(f"- val 最优条件模型 AUROC: `{fmt(seq['selected_row']['val_auroc'])}`")
        lines.append(f"- test 条件模型 AUROC: `{fmt(seq['selected_row']['test_auroc'])}`")
        lines.append(f"- geometry_only / geometry+parallax / geometry+conditional: `{fmt(seq['geometry_only_test_auroc'])}` / `{fmt(seq['geometry_plus_parallax_test_auroc'])}` / `{fmt(seq['selected_row']['test_auroc'])}`")
        lines.append("")
        lines.append("| metric | value |")
        lines.append("| --- | --- |")
        lines.append(f"| help total | {seq['analysis']['summary']['help_total']} |")
        lines.append(f"| help preserved | {seq['analysis']['summary']['help_preserved_count']} |")
        lines.append(f"| help lost | {seq['analysis']['summary']['help_lost_count']} |")
        lines.append(f"| hurt total | {seq['analysis']['summary']['hurt_total']} |")
        lines.append(f"| hurt avoided | {seq['analysis']['summary']['hurt_avoided_count']} |")
        lines.append(f"| hurt persisted | {seq['analysis']['summary']['hurt_persisted_count']} |")
        lines.append(f"| new hurt vs geo | {seq['analysis']['summary']['new_hurt_vs_geo_count']} |")
        lines.append(f"| mean cond-vs-geo aligned delta | {fmt(seq['analysis']['summary']['mean_cond_vs_geo_aligned_delta'])} |")
        lines.append(f"| mean cond-vs-gp aligned delta | {fmt(seq['analysis']['summary']['mean_cond_vs_gp_aligned_delta'])} |")
        lines.append("")
        lines.append("保住帮助段")
        lines.append("")
        lines.append("| segment | rows | y+ | mean parallax | cond-vs-gp delta |")
        lines.append("| --- | --- | --- | --- | --- |")
        for row in seq["analysis"]["help_preserved_segments"][:5]:
            lines.append(
                f"| {row['start_window_id']}-{row['end_window_id']} | {row['num_rows']} | {row['num_positive_labels']} | {fmt(row['mean_parallax'])} | {fmt(row['mean_cond_vs_gp_delta'])} |"
            )
        lines.append("")
        lines.append("避开伤害段")
        lines.append("")
        lines.append("| segment | rows | y+ | mean parallax | cond-vs-gp delta |")
        lines.append("| --- | --- | --- | --- | --- |")
        for row in seq["analysis"]["hurt_avoided_segments"][:5]:
            lines.append(
                f"| {row['start_window_id']}-{row['end_window_id']} | {row['num_rows']} | {row['num_positive_labels']} | {fmt(row['mean_parallax'])} | {fmt(row['mean_cond_vs_gp_delta'])} |"
            )
    lines.append("")
    lines.append("## 收口")
    lines.append("")
    for item in payload["takeaways"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Validate conditional activation of parallax on explicit held-out post_v2 packages.")
    ap.add_argument("--holdout_root", action="append", required=True)
    ap.add_argument("--thresholds", default="45,50,55,60,65,70")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    holdout_roots = [Path(p).expanduser().resolve() for p in args.holdout_root]
    thresholds = [float(x.strip()) for x in str(args.thresholds).split(",") if x.strip()]
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_sequences = []
    headline = []
    for holdout_root in holdout_roots:
        holdout_name = holdout_root.name
        seq_out_dir = out_dir / holdout_name
        base_summary = load_json(holdout_root / "post_v2_core4_holdout_summary.json")
        row_map = {r["model"]: r for r in base_summary["rows"]}
        seq_rows = []
        for threshold in thresholds:
            threshold_tag = f"cond_ge_{int(round(threshold))}"
            package_dir, feature_name = materialize_package(
                holdout_root,
                threshold=threshold,
                out_dir=seq_out_dir / "threshold_runs" / threshold_tag / "package",
            )
            model_dir = seq_out_dir / "threshold_runs" / threshold_tag / "geometry_plus_conditional_parallax"
            metrics = run_baseline(package_dir, GEOMETRY_COLS + [feature_name], model_dir)
            analysis = analyze_threshold(holdout_root, package_dir, model_dir, feature_name)
            val_metrics = metrics["splits"]["val"]
            test_metrics = metrics["splits"]["test"]
            row = {
                "threshold": threshold,
                "feature_name": feature_name,
                "val_auroc": val_metrics["auroc"],
                "test_auroc": test_metrics["auroc"],
                "test_auprc": test_metrics["auprc"],
                "test_brier": test_metrics["brier"],
                "test_ece": test_metrics["ece"],
                **analysis["summary"],
            }
            seq_rows.append(row)
            write_csv(
                seq_out_dir / "threshold_runs" / threshold_tag / "merged_rows.csv",
                list(analysis["merged_rows"][0].keys()) if analysis["merged_rows"] else [],
                analysis["merged_rows"],
            )
            write_csv(
                seq_out_dir / "threshold_runs" / threshold_tag / "help_preserved_segments.csv",
                list(analysis["help_preserved_segments"][0].keys()) if analysis["help_preserved_segments"] else [],
                analysis["help_preserved_segments"],
            )
            write_csv(
                seq_out_dir / "threshold_runs" / threshold_tag / "hurt_avoided_segments.csv",
                list(analysis["hurt_avoided_segments"][0].keys()) if analysis["hurt_avoided_segments"] else [],
                analysis["hurt_avoided_segments"],
            )
            write_json(seq_out_dir / "threshold_runs" / threshold_tag / "analysis_summary.json", analysis["summary"])

        seq_rows.sort(key=lambda r: (float(r["val_auroc"]), float(r["test_auroc"])), reverse=True)
        selected = seq_rows[0]
        selected_tag = f"cond_ge_{int(round(float(selected['threshold'])))}"
        selected_analysis = {
            "summary": load_json(seq_out_dir / "threshold_runs" / selected_tag / "analysis_summary.json"),
            "help_preserved_segments": load_csv_rows(seq_out_dir / "threshold_runs" / selected_tag / "help_preserved_segments.csv"),
            "hurt_avoided_segments": load_csv_rows(seq_out_dir / "threshold_runs" / selected_tag / "hurt_avoided_segments.csv"),
        }
        write_csv(
            seq_out_dir / "conditional_parallax_threshold_sweep.csv",
            list(seq_rows[0].keys()),
            seq_rows,
        )
        write_json(
            seq_out_dir / "conditional_parallax_threshold_sweep.json",
            {
                "holdout_root": str(holdout_root),
                "test_sequence": base_summary["test_sequence"],
                "selected_threshold": selected["threshold"],
                "rows": seq_rows,
            },
        )
        all_sequences.append(
            {
                "holdout_name": holdout_name,
                "test_sequence": base_summary["test_sequence"],
                "geometry_only_test_auroc": row_map["geometry_only"]["auroc"],
                "geometry_plus_parallax_test_auroc": row_map["geometry_plus_parallax"]["auroc"],
                "selected_threshold": selected["threshold"],
                "selected_row": selected,
                "analysis": selected_analysis,
            }
        )
        headline.append(
            f"{base_summary['test_sequence']} 上，条件化阈值 `{int(round(float(selected['threshold'])))} px` 的 test AUROC={fmt(selected['test_auroc'])}，相对 geometry+parallax={fmt(row_map['geometry_plus_parallax']['auroc'])}。"
        )

    takeaways = [
        "如果条件化版本能在 held-out 上稳定提升 geometry+parallax，同时减少 hurt_avoided 之外的新 hurt，就说明高视差门控在吸收有用局部增益。",
        "如果条件化版本仍然追不上 geometry_only，则当前 parallax 的问题不只是低视差污染，还包括 sequence-level 机制差异。",
        "这一步只验证“是否值得把 parallax 从总是加入改成条件加入”，不把它直接写成最终正式配方。",
    ]
    payload = {"headline": headline, "sequences": all_sequences, "takeaways": takeaways}
    write_json(out_dir / "conditional_parallax_holdout_summary.json", payload)
    write_text(out_dir / "conditional_parallax_holdout_summary.md", build_markdown(payload))
    print(f"[ConditionalParallax] saved -> {out_dir}")


if __name__ == "__main__":
    main()
