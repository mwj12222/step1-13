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


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def safe_float(v):
    try:
        x = float(v)
    except Exception:
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def fmt(v, nd=4):
    if v in (None, ""):
        return "-"
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return str(v)


def prediction_map(path: Path) -> dict[str, dict]:
    rows = load_csv_rows(path)
    out = {}
    for row in rows:
        if row.get("split") != "test":
            continue
        out[str(row["sample_uid"])] = row
    return out


def aligned_score(y_true: int, p_hat: float) -> float:
    return p_hat if y_true == 1 else (1.0 - p_hat)


def group_segments(rows: list[dict], key: str, positive: bool) -> list[dict]:
    if positive:
        filtered = [r for r in rows if float(r[key]) > 0.0]
        filtered.sort(key=lambda r: int(r["window_id"]))
    else:
        filtered = [r for r in rows if float(r[key]) < 0.0]
        filtered.sort(key=lambda r: int(r["window_id"]))
    segments = []
    current = []
    for row in filtered:
        if not current:
            current = [row]
            continue
        prev = current[-1]
        if int(row["window_id"]) - int(prev["window_id"]) == 10:
            current.append(row)
        else:
            segments.append(current)
            current = [row]
    if current:
        segments.append(current)
    out = []
    for seg in segments:
        item = {
            "start_window_id": int(seg[0]["window_id"]),
            "end_window_id": int(seg[-1]["window_id"]),
            "num_rows": len(seg),
            "num_positive_labels": sum(int(r["y_true"]) for r in seg),
            "mean_geo_p": mean([float(r["p_geometry_only"]) for r in seg]),
            "mean_gp_p": mean([float(r["p_geometry_plus_parallax"]) for r in seg]),
            "mean_aligned_delta": mean([float(r["aligned_delta"]) for r in seg]),
            "threshold_help_count": sum(int(r["threshold_help"]) for r in seg),
            "threshold_hurt_count": sum(int(r["threshold_hurt"]) for r in seg),
        }
        for feat in FEATURES:
            vals = [safe_float(r.get(feat)) for r in seg]
            vals = [v for v in vals if v is not None]
            item[f"{feat}_mean"] = mean(vals)
        out.append(item)
    out.sort(key=lambda r: abs(float(r["mean_aligned_delta"])), reverse=True)
    return out


def build_markdown(payload: dict) -> str:
    lines = []
    lines.append("# post_v2 parallax holdout diff")
    lines.append("")
    lines.append("## 结论先行")
    lines.append("")
    for item in payload["headline_judgement"]:
        lines.append(f"- {item}")
    for seq in payload["sequences"]:
        lines.append("")
        lines.append(f"## {seq['test_sequence']}")
        lines.append("")
        lines.append("| metric | value |")
        lines.append("| --- | --- |")
        lines.append(f"| geometry_only AUROC | {fmt(seq['summary']['geometry_only_auroc'])} |")
        lines.append(f"| geometry+parallax AUROC | {fmt(seq['summary']['geometry_plus_parallax_auroc'])} |")
        lines.append(f"| mean aligned delta | {fmt(seq['summary']['mean_aligned_delta'])} |")
        lines.append(f"| threshold help count | {seq['summary']['threshold_help_count']} |")
        lines.append(f"| threshold hurt count | {seq['summary']['threshold_hurt_count']} |")
        lines.append("")
        lines.append("帮助最大的样本段")
        lines.append("")
        lines.append("| window segment | rows | y+ | aligned delta | threshold help | parallax | reproj_p90 | tri_points |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for row in seq["top_help_segments"][:5]:
            lines.append(
                f"| {row['start_window_id']}-{row['end_window_id']} | {row['num_rows']} | {row['num_positive_labels']} | {fmt(row['mean_aligned_delta'])} | {row['threshold_help_count']} | {fmt(row['parallax_px_candidate_mean'])} | {fmt(row['reproj_p90_px_mean'])} | {fmt(row['tri_points_candidate_mean'])} |"
            )
        lines.append("")
        lines.append("拖后腿最大的样本段")
        lines.append("")
        lines.append("| window segment | rows | y+ | aligned delta | threshold hurt | parallax | reproj_p90 | tri_points |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for row in seq["top_hurt_segments"][:5]:
            lines.append(
                f"| {row['start_window_id']}-{row['end_window_id']} | {row['num_rows']} | {row['num_positive_labels']} | {fmt(row['mean_aligned_delta'])} | {row['threshold_hurt_count']} | {fmt(row['parallax_px_candidate_mean'])} | {fmt(row['reproj_p90_px_mean'])} | {fmt(row['tri_points_candidate_mean'])} |"
            )
    lines.append("")
    lines.append("## 收口")
    lines.append("")
    for item in payload["critical_takeaways"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def analyze_holdout(root: Path) -> dict:
    test_rows = load_csv_rows(root / "risk_dataset_post_v2_min_default_test.csv")
    geo_map = prediction_map(root / "core4_runs" / "geometry_only" / "predictions.csv")
    gp_map = prediction_map(root / "core4_runs" / "geometry_plus_parallax" / "predictions.csv")
    sample_ids = sorted(
        set(geo_map.keys()) & set(gp_map.keys()),
        key=lambda uid: int(next(r["window_id"] for r in test_rows if r["sample_uid"] == uid)),
    )
    data_rows = {str(r["sample_uid"]): r for r in test_rows}
    merged = []
    for uid in sample_ids:
        base = data_rows[uid]
        y = int(base["Y_bad_v2_min_default"])
        p_geo = float(geo_map[uid]["p_hat"])
        p_gp = float(gp_map[uid]["p_hat"])
        row = {
            "sample_uid": uid,
            "sequence": base["sequence"],
            "window_id": int(base["window_id"]),
            "y_true": y,
            "p_geometry_only": p_geo,
            "p_geometry_plus_parallax": p_gp,
            "aligned_geo": aligned_score(y, p_geo),
            "aligned_gp": aligned_score(y, p_gp),
            "aligned_delta": aligned_score(y, p_gp) - aligned_score(y, p_geo),
            "threshold_help": int((p_geo >= 0.5) != bool(y) and (p_gp >= 0.5) == bool(y)),
            "threshold_hurt": int((p_geo >= 0.5) == bool(y) and (p_gp >= 0.5) != bool(y)),
        }
        for feat in FEATURES:
            row[feat] = base.get(feat, "")
        merged.append(row)

    help_segments = group_segments(merged, "aligned_delta", positive=True)
    hurt_segments = group_segments(merged, "aligned_delta", positive=False)
    summary = {
        "test_sequence": merged[0]["sequence"] if merged else "<none>",
        "num_rows": len(merged),
        "geometry_only_auroc": next(iter(load_json(root / "post_v2_core4_holdout_summary.json")["rows"]))["auroc"] if False else None,
    }
    holdout_rows = load_json(root / "post_v2_core4_holdout_summary.json")["rows"]
    row_map = {r["model"]: r for r in holdout_rows}
    summary = {
        "test_sequence": merged[0]["sequence"] if merged else "<none>",
        "num_rows": len(merged),
        "geometry_only_auroc": row_map["geometry_only"]["auroc"],
        "geometry_plus_parallax_auroc": row_map["geometry_plus_parallax"]["auroc"],
        "mean_aligned_delta": mean([float(r["aligned_delta"]) for r in merged]),
        "threshold_help_count": sum(int(r["threshold_help"]) for r in merged),
        "threshold_hurt_count": sum(int(r["threshold_hurt"]) for r in merged),
    }
    return {
        "test_sequence": summary["test_sequence"],
        "summary": summary,
        "merged_rows": merged,
        "top_help_segments": help_segments,
        "top_hurt_segments": hurt_segments,
    }


def main():
    ap = argparse.ArgumentParser(description="Analyze where parallax helps or hurts relative to geometry_only on held-out sequences.")
    ap.add_argument("--holdout_root", action="append", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    roots = [Path(p).expanduser().resolve() for p in args.holdout_root]
    out_dir = Path(args.out_dir).expanduser().resolve()

    payload_sequences = [analyze_holdout(root) for root in roots]
    headline = []
    for seq in payload_sequences:
        s = seq["summary"]
        headline.append(
            f"{seq['test_sequence']} 上，geometry+parallax 相对 geometry_only 的 mean aligned delta={fmt(s['mean_aligned_delta'])}，threshold help={s['threshold_help_count']}，threshold hurt={s['threshold_hurt_count']}。"
        )
    critical = [
        "如果同一 held-out 上 threshold hurt 多于 threshold help，且 mean aligned delta 为负，说明 parallax 不只是没带来额外排序收益，而是在更多窗口上把正确方向分数推错了。",
        "当前两条新增 held-out 的共同点，是 geometry+parallax 仍优于 full，但帮助主要集中在局部窗口段，拖后腿则更广泛地分布在该 sequence 内。",
        "因此现在最该解释的不是“parallax 为什么有时有效”，而是“什么样的窗口段会让 parallax 从增益项变成干扰项”。",
    ]
    payload = {
        "holdout_roots": [str(r) for r in roots],
        "headline_judgement": headline,
        "sequences": payload_sequences,
        "critical_takeaways": critical,
    }

    write_json(out_dir / "post_v2_parallax_holdout_diff.json", payload)
    for seq in payload_sequences:
        seq_name = seq["test_sequence"].replace("/", "__")
        write_csv(
            out_dir / f"{seq_name}_merged_rows.csv",
            list(seq["merged_rows"][0].keys()) if seq["merged_rows"] else [],
            seq["merged_rows"],
        )
        write_csv(
            out_dir / f"{seq_name}_help_segments.csv",
            list(seq["top_help_segments"][0].keys()) if seq["top_help_segments"] else [],
            seq["top_help_segments"],
        )
        write_csv(
            out_dir / f"{seq_name}_hurt_segments.csv",
            list(seq["top_hurt_segments"][0].keys()) if seq["top_hurt_segments"] else [],
            seq["top_hurt_segments"],
        )
    write_text(out_dir / "post_v2_parallax_holdout_diff.md", build_markdown(payload))


if __name__ == "__main__":
    main()
