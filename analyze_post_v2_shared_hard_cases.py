#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import math


FEATURES = [
    "parallax_px_candidate",
    "tri_points_candidate",
    "pnp_success_rate",
    "reproj_med_px",
    "reproj_p90_px",
    "cheirality_ratio",
    "front_p_static",
    "front_coverage_ratio",
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


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def safe_float(v, default=None):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    if math.isnan(x) or math.isinf(x):
        return default
    return x


def mean(xs: list[float]):
    return sum(xs) / len(xs) if xs else None


def fmt(v, nd=4):
    if v is None:
        return "-"
    return f"{float(v):.{nd}f}"


def pred_map(path: Path) -> dict[str, dict]:
    rows = load_csv_rows(path)
    return {str(r["sample_uid"]): r for r in rows if str(r.get("split", "")) == "test"}


def contiguous_segments(rows: list[dict], flag_key: str) -> list[dict]:
    flagged = [r for r in rows if int(r[flag_key]) == 1]
    flagged.sort(key=lambda r: int(r["window_id"]))
    segments = []
    cur = []
    for row in flagged:
        if not cur:
            cur = [row]
            continue
        if int(row["window_id"]) - int(cur[-1]["window_id"]) == 10:
            cur.append(row)
        else:
            segments.append(cur)
            cur = [row]
    if cur:
        segments.append(cur)
    out = []
    for seg in segments:
        item = {
            "sequence": seg[0]["sequence"],
            "start_window_id": int(seg[0]["window_id"]),
            "end_window_id": int(seg[-1]["window_id"]),
            "num_rows": len(seg),
            "num_positive_labels": sum(int(r["y_true"]) for r in seg),
            "mean_p_geo": mean([float(r["p_geometry_only"]) for r in seg]),
            "mean_p_gated": mean([float(r["p_gated"]) for r in seg]),
        }
        for feat in FEATURES:
            vals = [safe_float(r.get(feat)) for r in seg]
            vals = [v for v in vals if v is not None]
            item[f"{feat}_mean"] = mean(vals)
        out.append(item)
    out.sort(key=lambda r: r["num_rows"], reverse=True)
    return out


def analyze_one_core4_holdout(root: Path) -> dict:
    test_rows = load_csv_rows(root / "risk_dataset_post_v2_min_default_test.csv")
    by_uid = {str(r["sample_uid"]): r for r in test_rows}
    geo = pred_map(root / "core4_runs" / "geometry_only" / "predictions.csv")
    gated = pred_map(root / "core4_runs" / "geometry_plus_gated_parallax" / "predictions.csv")
    seq = test_rows[0]["sequence"] if test_rows else root.name
    merged = []
    for uid in sorted(set(geo) & set(gated), key=lambda k: int(by_uid[k]["window_id"])):
        row = by_uid[uid]
        y = int(float(row["Y_bad_v2_min_default"]))
        p_geo = float(geo[uid]["p_hat"])
        p_gated = float(gated[uid]["p_hat"])
        wrong_geo = int((p_geo >= 0.5) != bool(y))
        wrong_gated = int((p_gated >= 0.5) != bool(y))
        item = {
            "source_type": "core4_gated_holdout",
            "sequence": seq,
            "sample_uid": uid,
            "window_id": int(row["window_id"]),
            "y_true": y,
            "p_geometry_only": p_geo,
            "p_gated": p_gated,
            "wrong_geometry_only": wrong_geo,
            "wrong_gated": wrong_gated,
            "shared_wrong": int(wrong_geo == 1 and wrong_gated == 1),
            "gated_fixes_geo": int(wrong_geo == 1 and wrong_gated == 0),
            "geo_beats_gated": int(wrong_geo == 0 and wrong_gated == 1),
        }
        for feat in FEATURES:
            item[feat] = row.get(feat, "")
        merged.append(item)
    return {"sequence": seq, "rows": merged}


def analyze_one_conditional_holdout(cond_root: Path, base_holdout_root: Path, cond_summary_entry: dict) -> dict:
    holdout_name = cond_summary_entry["holdout_name"]
    sequence = cond_summary_entry["test_sequence"]
    threshold = int(round(float(cond_summary_entry["selected_threshold"])))
    pkg_dir = cond_root / holdout_name / "threshold_runs" / f"cond_ge_{threshold}" / "package"
    model_dir = cond_root / holdout_name / "threshold_runs" / f"cond_ge_{threshold}" / "geometry_plus_conditional_parallax"
    test_rows = load_csv_rows(pkg_dir / "risk_dataset_post_v2_min_default_test.csv")
    by_uid = {str(r["sample_uid"]): r for r in test_rows}
    geo = pred_map(base_holdout_root / holdout_name / "core4_runs" / "geometry_only" / "predictions.csv")
    gated = pred_map(model_dir / "predictions.csv")

    merged = []
    for uid in sorted(set(geo) & set(gated), key=lambda k: int(by_uid[k]["window_id"])):
        row = by_uid[uid]
        y = int(float(row["Y_bad_v2_min_default"]))
        p_geo = float(geo[uid]["p_hat"])
        p_gated = float(gated[uid]["p_hat"])
        wrong_geo = int((p_geo >= 0.5) != bool(y))
        wrong_gated = int((p_gated >= 0.5) != bool(y))
        item = {
            "source_type": "conditional_holdout",
            "sequence": sequence,
            "sample_uid": uid,
            "window_id": int(row["window_id"]),
            "y_true": y,
            "p_geometry_only": p_geo,
            "p_gated": p_gated,
            "wrong_geometry_only": wrong_geo,
            "wrong_gated": wrong_gated,
            "shared_wrong": int(wrong_geo == 1 and wrong_gated == 1),
            "gated_fixes_geo": int(wrong_geo == 1 and wrong_gated == 0),
            "geo_beats_gated": int(wrong_geo == 0 and wrong_gated == 1),
        }
        for feat in FEATURES:
            item[feat] = row.get(feat, "")
        merged.append(item)
    return {"sequence": sequence, "rows": merged}


def feature_profile(rows: list[dict], flag_key: str) -> list[dict]:
    selected = [r for r in rows if int(r[flag_key]) == 1]
    baseline = rows
    out = []
    for feat in FEATURES:
        a = [safe_float(r.get(feat)) for r in selected]
        a = [v for v in a if v is not None]
        b = [safe_float(r.get(feat)) for r in baseline]
        b = [v for v in b if v is not None]
        out.append(
            {
                "feature": feat,
                "selected_mean": mean(a),
                "all_mean": mean(b),
                "delta": (mean(a) - mean(b)) if a and b else None,
            }
        )
    return out


def build_markdown(payload: dict) -> str:
    lines = []
    lines.append("# post_v2 shared hard-case profile")
    lines.append("")
    lines.append("## 结论先行")
    lines.append("")
    for item in payload["headline"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## 按 held-out 汇总")
    lines.append("")
    lines.append("| sequence | rows | both wrong | gated fixes geo | geo beats gated | both correct |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for row in payload["sequence_summary"]:
        lines.append(
            f"| {row['sequence']} | {row['num_rows']} | {row['shared_wrong_count']} | {row['gated_fixes_geo_count']} | {row['geo_beats_gated_count']} | {row['both_correct_count']} |"
        )
    lines.append("")
    lines.append("## shared hard cases 的特征画像")
    lines.append("")
    lines.append("| feature | shared_wrong mean | all_test mean | delta |")
    lines.append("| --- | --- | --- | --- |")
    for row in payload["shared_wrong_profile"]:
        lines.append(
            f"| {row['feature']} | {fmt(row['selected_mean'])} | {fmt(row['all_mean'])} | {fmt(row['delta'])} |"
        )
    lines.append("")
    lines.append("## gated 修掉 geometry 的样本画像")
    lines.append("")
    lines.append("| feature | gated_fixes_geo mean | all_test mean | delta |")
    lines.append("| --- | --- | --- | --- |")
    for row in payload["gated_fix_profile"]:
        lines.append(
            f"| {row['feature']} | {fmt(row['selected_mean'])} | {fmt(row['all_mean'])} | {fmt(row['delta'])} |"
        )
    lines.append("")
    lines.append("## geometry 胜过 gated 的样本画像")
    lines.append("")
    lines.append("| feature | geo_beats_gated mean | all_test mean | delta |")
    lines.append("| --- | --- | --- | --- |")
    for row in payload["geo_beats_profile"]:
        lines.append(
            f"| {row['feature']} | {fmt(row['selected_mean'])} | {fmt(row['all_mean'])} | {fmt(row['delta'])} |"
        )
    lines.append("")
    lines.append("## 典型共享难段")
    lines.append("")
    lines.append("| sequence | segment | rows | y+ | parallax | reproj_p90 | tri_points |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in payload["top_shared_segments"][:12]:
        lines.append(
            f"| {row['sequence']} | {row['start_window_id']}-{row['end_window_id']} | {row['num_rows']} | {row['num_positive_labels']} | {fmt(row['parallax_px_candidate_mean'])} | {fmt(row['reproj_p90_px_mean'])} | {fmt(row['tri_points_candidate_mean'])} |"
        )
    lines.append("")
    lines.append("## 收口")
    lines.append("")
    for item in payload["takeaways"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Profile shared hard cases between geometry_only and geometry+gated_parallax across explicit held-outs.")
    ap.add_argument(
        "--conditional_root",
        default="/mnt/g/Result/VIODE/post_v2_rebuild_9seq/post_v2_conditional_parallax_holdout_20260319",
    )
    ap.add_argument(
        "--conditional_summary_json",
        default="/mnt/g/Result/VIODE/post_v2_rebuild_9seq/post_v2_conditional_parallax_holdout_20260319/conditional_parallax_holdout_summary.json",
    )
    ap.add_argument(
        "--base_plain_holdout_root",
        default="/mnt/g/Result/VIODE/post_v2_rebuild_9seq",
    )
    ap.add_argument("--gated_holdout_root", action="append", default=[])
    ap.add_argument(
        "--out_dir",
        default=str(Path(__file__).resolve().parents[2] / "docs" / "research" / "init_risk_post_v2_shared_hard_cases_20260319"),
    )
    args = ap.parse_args()

    conditional_root = Path(args.conditional_root).expanduser().resolve()
    conditional_summary = load_json(Path(args.conditional_summary_json).expanduser().resolve())
    base_plain_holdout_root = Path(args.base_plain_holdout_root).expanduser().resolve()
    gated_roots = [Path(p).expanduser().resolve() for p in args.gated_holdout_root]
    if not gated_roots:
        gated_roots = [
            Path("/mnt/g/Result/VIODE/post_v2_rebuild_9seq/post_v2_holdout_city_night_1_low_gated_20260319"),
            Path("/mnt/g/Result/VIODE/post_v2_rebuild_9seq/post_v2_holdout_city_day_0_none_gated_20260319"),
            Path("/mnt/g/Result/VIODE/post_v2_rebuild_9seq/post_v2_holdout_parking_lot_3_high_gated_20260319"),
        ]

    analyses = []
    for seq_entry in conditional_summary["sequences"]:
        analyses.append(analyze_one_conditional_holdout(conditional_root, base_plain_holdout_root, seq_entry))
    for root in gated_roots:
        analyses.append(analyze_one_core4_holdout(root))

    all_rows = []
    seq_summary = []
    all_shared_segments = []
    for item in analyses:
        rows = item["rows"]
        all_rows.extend(rows)
        seq_summary.append(
            {
                "sequence": item["sequence"],
                "num_rows": len(rows),
                "shared_wrong_count": sum(int(r["shared_wrong"]) for r in rows),
                "gated_fixes_geo_count": sum(int(r["gated_fixes_geo"]) for r in rows),
                "geo_beats_gated_count": sum(int(r["geo_beats_gated"]) for r in rows),
                "both_correct_count": sum(int(r["wrong_geometry_only"] == 0 and r["wrong_gated"] == 0) for r in rows),
            }
        )
        all_shared_segments.extend(contiguous_segments(rows, "shared_wrong"))

    seq_summary.sort(key=lambda r: r["shared_wrong_count"], reverse=True)
    all_shared_segments.sort(key=lambda r: (r["num_rows"], r["num_positive_labels"]), reverse=True)

    shared_wrong_profile = feature_profile(all_rows, "shared_wrong")
    gated_fix_profile = feature_profile(all_rows, "gated_fixes_geo")
    geo_beats_profile = feature_profile(all_rows, "geo_beats_gated")

    total_rows = len(all_rows)
    total_shared = sum(int(r["shared_wrong"]) for r in all_rows)
    total_fix = sum(int(r["gated_fixes_geo"]) for r in all_rows)
    total_geo_beats = sum(int(r["geo_beats_gated"]) for r in all_rows)
    headline = [
        f"跨 {len(seq_summary)} 条 explicit held-out test 序列，geometry_only 与 geometry+gated_parallax 共有 {total_rows} 条测试样本，其中 both-wrong 共有 {total_shared} 条。",
        f"gated 修掉 geometry 的样本有 {total_fix} 条，而 geometry 胜过 gated 的样本有 {total_geo_beats} 条。",
        "当前共享难例仍占显著比例，说明后续瓶颈已经开始转向更细结构或标签边界，而不只是继续替换配方。",
    ]
    takeaways = [
        "如果 both-wrong 仍然很多，说明 geometry 与 gated_parallax 都在同一类难例上失效，这更像标签边界或缺特征问题。",
        "如果 gated_fixes_geo 主要集中在高视差正样本，而 geo_beats_gated 主要落在中低视差或混合段，那么 gated_parallax 的作用更像条件性补充，而不是全局替代几何。",
        "下一步最值得做的不是继续扩 full，而是盯住这些 both-wrong 难例，看它们是否共享同样的 horizon / trigger / observability 缺口。",
    ]

    payload = {
        "headline": headline,
        "sequence_summary": seq_summary,
        "shared_wrong_profile": shared_wrong_profile,
        "gated_fix_profile": gated_fix_profile,
        "geo_beats_profile": geo_beats_profile,
        "top_shared_segments": all_shared_segments,
        "takeaways": takeaways,
    }

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "post_v2_shared_hard_cases_summary.json", payload)
    write_text(out_dir / "post_v2_shared_hard_cases_summary.md", build_markdown(payload))
    if all_rows:
        write_csv(out_dir / "post_v2_shared_hard_cases_rows.csv", list(all_rows[0].keys()), all_rows)
    write_csv(out_dir / "post_v2_shared_hard_cases_by_sequence.csv", list(seq_summary[0].keys()), seq_summary)
    write_csv(out_dir / "post_v2_shared_hard_cases_feature_profile_shared_wrong.csv", list(shared_wrong_profile[0].keys()), shared_wrong_profile)
    write_csv(out_dir / "post_v2_shared_hard_cases_feature_profile_gated_fix.csv", list(gated_fix_profile[0].keys()), gated_fix_profile)
    write_csv(out_dir / "post_v2_shared_hard_cases_feature_profile_geo_beats.csv", list(geo_beats_profile[0].keys()), geo_beats_profile)
    if all_shared_segments:
        write_csv(out_dir / "post_v2_shared_hard_cases_segments.csv", list(all_shared_segments[0].keys()), all_shared_segments)
    print(f"[PostV2SharedHardCases] saved -> {out_dir}")


if __name__ == "__main__":
    main()
