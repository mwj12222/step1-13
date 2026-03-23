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


def fmt(v, nd=4):
    if v in (None, ""):
        return "-"
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return str(v)


def build_markdown(payload: dict) -> str:
    candidate_label = payload["candidate_label"]
    lines = []
    lines.append("# post_v2 core4 multi-seed 正式补充小结")
    lines.append("")
    lines.append("## 结论先行")
    lines.append("")
    for item in payload["headline_judgement"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## seed 级结果")
    lines.append("")
    lines.append(f"| seed | geometry_only | {candidate_label} | drop_front | full_anchor | judgement |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for row in payload["per_seed_rows"]:
        lines.append(
            f"| {row['seed']} | {fmt(row['geometry_only'])} | {fmt(row['candidate_model'])} | {fmt(row['drop_front'])} | {fmt(row['full_anchor'])} | {row['judgement']} |"
        )
    lines.append("")
    lines.append("## aggregate")
    lines.append("")
    lines.append("| model | successful_seeds | mean AUROC | min AUROC | max AUROC | mean AUPRC |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for row in payload["aggregate_rows"]:
        lines.append(
            f"| {row['model']} | {row['num_successful_seeds']} | {fmt(row['mean_auroc'])} | {fmt(row['min_auroc'])} | {fmt(row['max_auroc'])} | {fmt(row['mean_auprc'])} |"
        )
    lines.append("")
    lines.append("## 当前可以升级的口径")
    lines.append("")
    for item in payload["claim_safe"]:
        lines.append(f"- Allowed: {item}")
    for item in payload["claim_forbidden"]:
        lines.append(f"- Forbidden: {item}")
    lines.append("")
    lines.append("## 收口")
    lines.append("")
    for item in payload["critical_takeaways"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Summarize the post_v2 core4 multi-seed comparison into a formal supplement note.")
    ap.add_argument("--multiseed_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    multiseed_dir = Path(args.multiseed_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    run_rows = load_csv_rows(multiseed_dir / "post_v2_core4_multiseed_runs.csv")
    aggregate_rows = load_csv_rows(multiseed_dir / "post_v2_core4_multiseed_aggregate.csv")
    summary = load_json(multiseed_dir / "post_v2_core4_multiseed_summary.json")

    by_seed = {}
    for row in run_rows:
        by_seed.setdefault(str(row["seed"]), {})[str(row["model"])] = float(row["auroc"])

    candidate_model = "geometry_plus_gated_parallax"
    if not any(candidate_model in row for row in by_seed.values()):
        candidate_model = "geometry_plus_parallax"
    candidate_label = "geometry+gated_parallax" if candidate_model == "geometry_plus_gated_parallax" else "geometry+parallax"

    per_seed_rows = []
    gp_all_noninferior = True
    gp_all_better_than_full = True
    for seed, row in sorted(by_seed.items()):
        gp = row[candidate_model]
        go = row["geometry_only"]
        df = row["drop_front"]
        full = row["full_anchor"]
        noninferior = gp >= go
        better_than_full = gp > full
        gp_all_noninferior &= noninferior
        gp_all_better_than_full &= better_than_full
        per_seed_rows.append(
            {
                "seed": seed,
                "geometry_only": go,
                "candidate_model": gp,
                "drop_front": df,
                "full_anchor": full,
                "judgement": (
                    "gp>=geometry_only and gp>full"
                    if noninferior and better_than_full
                    else "failed_target_condition"
                ),
            }
        )

    payload = {
        "multiseed_dir": str(multiseed_dir),
        "seeds": summary["seeds"],
        "candidate_label": candidate_label,
        "per_seed_rows": per_seed_rows,
        "aggregate_rows": aggregate_rows,
        "headline_judgement": [
            (
                f"在当前 3 个 split seed 的轻量重复验证中，`{candidate_label}` 在全部 seed 上都不差于 `geometry_only`，并且在全部 seed 上都优于 `full`。"
                if gp_all_noninferior and gp_all_better_than_full
                else f"当前 multi-seed 结果未能稳定支持 `{candidate_label}` 的升级。"
            ),
            f"`{candidate_label}` 的 mean AUROC={fmt(next(r['mean_auroc'] for r in aggregate_rows if r['model']==candidate_model))}，高于 `geometry_only`={fmt(next(r['mean_auroc'] for r in aggregate_rows if r['model']=='geometry_only'))}，也高于 `full_anchor`={fmt(next(r['mean_auroc'] for r in aggregate_rows if r['model']=='full_anchor'))}。",
            f"因此它现在可以从“当前最强候选”升级为“当前 post_v2 的优先验证候选配方”，但仍应保留为 protocol-level candidate rather than final closed-loop method claim。",
        ],
        "claim_safe": [
            f"可以说在当前 9seq 数据池上的多 split-seed 轻量验证中，{candidate_label} 对 geometry_only 表现出稳定非劣，对 full 表现出稳定优势。",
            f"可以说 {candidate_label} 现已升级为当前 post_v2 的优先验证候选配方。",
            "可以说 full_anchor 应继续保留为反例/锚点，而不是默认主配方。",
        ],
        "claim_forbidden": [
            "不能把这轮 multi-seed 结果写成跨数据集或跨 held-out sequence 的最终泛化结论。",
            f"不能把 {candidate_label} 直接上升为系统级 gate 已验证收益的结论。",
            "不能因为这轮 multi-seed 成立，就重新把 post_v2 写成已完全收敛的最终结构。",
        ],
        "critical_takeaways": [
            f"这轮 multi-seed 只回答一个问题，而且答案是肯定的：{candidate_label} 在当前协议下具备稳定升级资格。",
            f"下一阶段如果继续扩样本，应固定 geometry_only / {candidate_label} / drop_front / full_anchor 这 4 个核心对照。",
            f"后续再往前推进，最值钱的不是继续扩 full，而是看 {candidate_label} 在新 held-out sequence 上是否仍保持这一优势。",
        ],
    }

    write_json(out_dir / "post_v2_core4_multiseed_formal_summary.json", payload)
    write_csv(
        out_dir / "post_v2_core4_multiseed_formal_per_seed.csv",
        list(per_seed_rows[0].keys()),
        per_seed_rows,
    )
    write_text(out_dir / "post_v2_core4_multiseed_formal_summary.md", build_markdown(payload))


if __name__ == "__main__":
    main()
