#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json


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


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def fmt(v, nd=4):
    if v in (None, ""):
        return "-"
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return str(v)


def build_markdown(payload: dict) -> str:
    lines = []
    lines.append(f"# post_v2 explicit holdout: {payload['test_sequence']}")
    lines.append("")
    lines.append("## 结论先行")
    lines.append("")
    for item in payload["headline_judgement"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## core4 结果")
    lines.append("")
    lines.append("| model | test AUROC | test AUPRC | test Brier | test ECE |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in payload["rows"]:
        lines.append(
            f"| {row['model']} | {fmt(row['auroc'])} | {fmt(row['auprc'])} | {fmt(row['brier'])} | {fmt(row['ece'])} |"
        )
    lines.append("")
    lines.append("## 当前该怎么收口")
    lines.append("")
    for item in payload["critical_takeaways"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Summarize one explicit post_v2 sequence holdout result.")
    ap.add_argument("--holdout_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    holdout_dir = Path(args.holdout_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    summary = load_json(holdout_dir / "post_v2_core4_holdout_summary.json")
    rows = load_csv_rows(holdout_dir / "post_v2_core4_holdout_summary.csv")
    rows.sort(key=lambda r: float(r["auroc"]), reverse=True)

    best = rows[0]
    candidate_model = "geometry_plus_gated_parallax" if any(r["model"] == "geometry_plus_gated_parallax" for r in rows) else "geometry_plus_parallax"
    candidate_label = "geometry + gated_parallax" if candidate_model == "geometry_plus_gated_parallax" else "geometry + parallax"
    gp = next(r for r in rows if r["model"] == candidate_model)
    go = next(r for r in rows if r["model"] == "geometry_only")
    full = next(r for r in rows if r["model"] == "full_anchor")

    payload = {
        "holdout_dir": str(holdout_dir),
        "test_sequence": summary["test_sequence"],
        "val_sequences": summary["val_sequences"],
        "rows": rows,
        "headline_judgement": [
            f"在新的 explicit held-out test 端点 `{summary['test_sequence']}` 上，最优模型不是 `{candidate_label}`，而是 `geometry_only`，其 test AUROC={fmt(go['auroc'])}。"
            if float(go["auroc"]) >= float(gp["auroc"])
            else f"在新的 explicit held-out test 端点 `{summary['test_sequence']}` 上，最优模型已经变成 `{candidate_label}`，其 test AUROC={fmt(gp['auroc'])}，高于 `geometry_only`={fmt(go['auroc'])}。",
            f"`{candidate_label}` 仍然优于 `full`（{fmt(gp['auroc'])} vs {fmt(full['auroc'])}），但{'不再优于' if float(gp['auroc']) < float(go['auroc']) else '也优于'} `geometry_only`（{fmt(gp['auroc'])} vs {fmt(go['auroc'])}）。",
            "这说明前一轮 multi-seed 给出的“首选候选配方”判断，在同一 9seq 数据池内成立，但一旦切到新的 held-out sequence，优势还不够稳。",
        ],
        "critical_takeaways": [
            f"现在更稳的说法应该是：{candidate_label} 是当前最有希望的 post_v2 候选配方，但还没有通过新增 held-out sequence 的稳定性检验。",
            "geometry_only 在部分新 test 端点上重新拿回第一，说明 parallax 的补益带有 sequence dependence。",
            f"因此下一阶段不能把 {candidate_label} 直接写成已升级完成的最终配方，而应改写成：current preferred candidate pending broader held-out validation.",
        ],
    }

    write_json(out_dir / "post_v2_sequence_holdout_summary.json", payload)
    write_text(out_dir / "post_v2_sequence_holdout_summary.md", build_markdown(payload))


if __name__ == "__main__":
    main()
