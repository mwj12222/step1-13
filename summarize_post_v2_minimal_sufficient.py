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
    main = payload["main_results"]
    mini = payload["minimal_sufficient_rows"]
    lines = []
    lines.append("# post_v2 9seq 最小充分指标集正式小结")
    lines.append("")
    lines.append("## 结论先行")
    lines.append("")
    for item in payload["headline_judgement"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## 9seq 主结果")
    lines.append("")
    lines.append("| model | test AUROC | test AUPRC | test Brier | test ECE |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in main:
        lines.append(
            f"| {row['name']} | {fmt(row['auroc'])} | {fmt(row['auprc'])} | {fmt(row['brier'])} | {fmt(row['ece'])} |"
        )
    lines.append("")
    lines.append("## 最小充分指标集排序")
    lines.append("")
    lines.append("| rank | name | features | test AUROC | test AUPRC | note |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for idx, row in enumerate(mini, start=1):
        lines.append(
            f"| {idx} | {row['name']} | `{row['features']}` | {fmt(row['auroc'])} | {fmt(row['auprc'])} | {row['note']} |"
        )
    lines.append("")
    lines.append("## 当前可以正式收口的判断")
    lines.append("")
    for item in payload["critical_takeaways"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Allowed / Forbidden Claims")
    lines.append("")
    for item in payload["claim_safe"]:
        lines.append(f"- Allowed: {item}")
    for item in payload["claim_forbidden"]:
        lines.append(f"- Forbidden: {item}")
    lines.append("")
    lines.append("## 下一轮固定核心对照")
    lines.append("")
    for item in payload["next_round_core_set"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## 下一步建议")
    lines.append("")
    for item in payload["next_steps"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Summarize the 9seq post_v2 minimal-sufficient-set result into a formal one-page note.")
    ap.add_argument("--result_root", required=True)
    ap.add_argument("--minimal_sets_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    result_root = Path(args.result_root).expanduser().resolve()
    minimal_sets_dir = Path(args.minimal_sets_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    logistic_metrics = load_json(next(result_root.glob("model_post_v2_min_default_logistic_*/metrics.json")))
    dirrisk_metrics = load_json(next(result_root.glob("model_post_v2_min_default_dir_risk_*/metrics.json")))
    ablation_rows = load_csv_rows(next(result_root.glob("post_v2_ablations_*/post_v2_ablation_summary.csv")))
    minimal_rows = load_csv_rows(minimal_sets_dir / "post_v2_minimal_sufficient_summary.csv")

    rule_test = logistic_metrics["rule_baseline"]["test"]
    full_test = logistic_metrics["splits"]["test"]
    dirrisk_test = dirrisk_metrics["splits"]["test"]
    ablation_map = {row["name"]: row for row in ablation_rows}
    geom = ablation_map["post_v2_geometry_only"]
    drop_front = ablation_map["post_v2_drop_front"]
    best_minimal = minimal_rows[0]

    payload = {
        "result_root": str(result_root),
        "minimal_sets_dir": str(minimal_sets_dir),
        "main_results": [
            {
                "name": "rule_q_post_geom_only",
                "auroc": rule_test["auroc"],
                "auprc": rule_test["auprc"],
                "brier": rule_test["brier"],
                "ece": rule_test["ece"],
            },
            {
                "name": "full_anchor",
                "auroc": full_test["auroc"],
                "auprc": full_test["auprc"],
                "brier": full_test["brier"],
                "ece": full_test["ece"],
            },
            {
                "name": "dir_risk",
                "auroc": dirrisk_test["auroc"],
                "auprc": dirrisk_test["auprc"],
                "brier": dirrisk_test["brier"],
                "ece": dirrisk_test["ece"],
            },
            {
                "name": "geometry_only",
                "auroc": geom["auroc"],
                "auprc": geom["auprc"],
                "brier": geom["brier"],
                "ece": geom["ece"],
            },
            {
                "name": "drop_front",
                "auroc": drop_front["auroc"],
                "auprc": drop_front["auprc"],
                "brier": drop_front["brier"],
                "ece": drop_front["ece"],
            },
        ],
        "minimal_sufficient_rows": minimal_rows,
        "headline_judgement": [
            f"9seq seed20260432 上，`full` 不再适合做默认 post 配方：其 test AUROC={fmt(full_test['auroc'])}，落后于 `geometry_only`={fmt(geom['auroc'])}、`drop_front`={fmt(drop_front['auroc'])}，也落后于规则式 `Q_post_geom_only`={fmt(rule_test['auroc'])}。",
            f"当前最强最小充分候选是 `geometry + parallax`，test AUROC={fmt(best_minimal['auroc'])}；它仅用 4 个特征，就超过了 10 维 full anchor。",
            "当前 post_v2 不能再被写死成 geometry-dominated 或 candidate-dominated；更准确的状态是：结构仍在重审，但 full block mixing 已被证实不稳。",
        ],
        "critical_takeaways": [
            "不能再把 full 当默认配方。它现在更像一个保留的反例锚点，用来证明当前 block mixing 会伤害泛化。",
            "geometry 已经重新回到强位置，但最有价值的增量不是整个 candidate block，而是单一的 `parallax_px_candidate`。",
            "front 不能因为单特征偶尔有信号就整块回归主配方；至少在当前 9seq 上，`geometry + front_p_static` 反而退化。",
            "当前最稳的 v_next 候选是 `geometry + parallax`，但它仍然需要多-seed 或多 holdout 验证后，才能升为固定主配方。",
        ],
        "claim_safe": [
            "可以说当前 9seq 证据已足以否定“full 是默认稳定 post 配方”的写法。",
            "可以说 geometry + parallax 是当前最强、最克制的最小充分候选。",
            "可以说 post_v2 结构尚未收敛，因此项目叙事必须保持中性并持续做结构重审。",
        ],
        "claim_forbidden": [
            "不能把当前 post_v2 写成已证明的 geometry-dominated 任务。",
            "不能把 geometry + parallax 直接写成最终定型配方或系统级 gate 结论。",
            "不能把当前 9seq 单 seed 结果写成稳健跨场景泛化证据。",
        ],
        "next_round_core_set": [
            "`geometry_only`",
            "`geometry + parallax`",
            "`drop_front`",
            "`full anchor`",
        ],
        "next_steps": [
            "用 2-3 个 split seeds 先做轻量重复验证，检查 geometry + parallax 的增益是否稳定。",
            "后续每扩一批样本，先固定重跑 4 个核心对照，不再默认把 full 当主模型。",
            "如果 multi-seed 结果仍支持 geometry + parallax，再考虑把它升成 Q_post v_next 的核心候选。",
        ],
    }

    write_json(out_dir / "post_v2_minimal_sufficient_formal_summary.json", payload)
    write_csv(
        out_dir / "post_v2_minimal_sufficient_formal_main_results.csv",
        list(payload["main_results"][0].keys()),
        payload["main_results"],
    )
    write_text(out_dir / "post_v2_minimal_sufficient_formal_summary.md", build_markdown(payload))


if __name__ == "__main__":
    main()
