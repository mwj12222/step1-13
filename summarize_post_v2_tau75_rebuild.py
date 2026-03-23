#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_OUT_DIR = PROJECT_ROOT / "docs" / "research" / "init_risk_post_v2_tau75_rebuild_20260320"
DEFAULT_TAU_AUDIT_DIR = PROJECT_ROOT / "docs" / "research" / "init_risk_post_v2_tau_audit_20260320"
DEFAULT_OLD_SHARED_DIR = PROJECT_ROOT / "docs" / "research" / "init_risk_post_v2_shared_hard_cases_20260319"
DEFAULT_NEW_SHARED_DIR = PROJECT_ROOT / "docs" / "research" / "init_risk_post_v2_shared_hard_cases_tau7p5_20260320"
DEFAULT_OLD_CONDITIONAL_JSON = Path("/mnt/g/Result/VIODE/post_v2_rebuild_9seq/post_v2_conditional_parallax_holdout_20260319/conditional_parallax_holdout_summary.json")


OLD_HOLDOUTS = {
    "city_night/0_none": Path("/mnt/g/Result/VIODE/post_v2_rebuild_9seq/post_v2_holdout_city_night_0_none_20260319/post_v2_core4_holdout_summary.csv"),
    "city_day/2_mid": Path("/mnt/g/Result/VIODE/post_v2_rebuild_9seq/post_v2_holdout_city_day_2_mid_20260319/post_v2_core4_holdout_summary.csv"),
    "city_night/1_low": Path("/mnt/g/Result/VIODE/post_v2_rebuild_9seq/post_v2_holdout_city_night_1_low_gated_20260319/post_v2_core4_holdout_summary.csv"),
    "city_day/0_none": Path("/mnt/g/Result/VIODE/post_v2_rebuild_9seq/post_v2_holdout_city_day_0_none_gated_20260319/post_v2_core4_holdout_summary.csv"),
    "parking_lot/3_high": Path("/mnt/g/Result/VIODE/post_v2_rebuild_9seq/post_v2_holdout_parking_lot_3_high_gated_20260319/post_v2_core4_holdout_summary.csv"),
}

NEW_HOLDOUTS = {
    "city_night/0_none": Path("/mnt/g/Result/VIODE/post_v2_tau_audit_20260320/post_v2_holdout_city_night_0_none_tau7p5/post_v2_core4_holdout_summary.csv"),
    "city_day/2_mid": Path("/mnt/g/Result/VIODE/post_v2_tau_audit_20260320/post_v2_holdout_city_day_2_mid_tau7p5/post_v2_core4_holdout_summary.csv"),
    "city_night/1_low": Path("/mnt/g/Result/VIODE/post_v2_tau_audit_20260320/post_v2_holdout_city_night_1_low_tau7p5/post_v2_core4_holdout_summary.csv"),
    "city_day/0_none": Path("/mnt/g/Result/VIODE/post_v2_tau_audit_20260320/post_v2_holdout_city_day_0_none_tau7p5/post_v2_core4_holdout_summary.csv"),
    "parking_lot/3_high": Path("/mnt/g/Result/VIODE/post_v2_tau_audit_20260320/post_v2_holdout_parking_lot_3_high_tau7p5/post_v2_core4_holdout_summary.csv"),
}


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def fmt(v, nd: int = 4) -> str:
    if pd.isna(v):
        return "-"
    return f"{float(v):.{nd}f}"


def model_map(df: pd.DataFrame) -> dict[str, dict]:
    return {str(r["model"]): r for r in df.to_dict("records")}


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_markdown(payload: dict) -> str:
    lines = []
    lines.append("# post_v2 tau=7.5 rebuild summary")
    lines.append("")
    lines.append("## 结论先行")
    lines.append("")
    for item in payload["headline"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## 标签人口变化")
    lines.append("")
    lines.append("| tau | labeled_rows | bad_ratio_labeled | future_high_gt_rot |")
    lines.append("| --- | --- | --- | --- |")
    for row in payload["population_rows"]:
        lines.append(
            f"| {row['tau']} | {row['labeled_rows']} | {fmt(row['bad_ratio_labeled'])} | {row['future_high_gt_rot']} |"
        )
    lines.append("")
    lines.append("## held-out 排序变化（geometry_only vs gated）")
    lines.append("")
    lines.append("| sequence | tau5 geometry | tau5 gated | tau7.5 geometry | tau7.5 gated | winner shift |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for row in payload["holdout_rows"]:
        lines.append(
            f"| {row['sequence']} | {fmt(row['tau5_geometry'])} | {fmt(row['tau5_gated'])} | {fmt(row['tau75_geometry'])} | {fmt(row['tau75_gated'])} | {row['winner_shift']} |"
        )
    lines.append("")
    lines.append("## 共享难例变化")
    lines.append("")
    lines.append("| version | both_wrong | gated_fixes_geo | geo_beats_gated |")
    lines.append("| --- | --- | --- | --- |")
    for row in payload["shared_rows"]:
        lines.append(
            f"| {row['version']} | {row['both_wrong']} | {row['gated_fixes_geo']} | {row['geo_beats_gated']} |"
        )
    lines.append("")
    lines.append("## 收口")
    lines.append("")
    for item in payload["judgement"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize tau=7.5 post_v2 rebuild results.")
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tau_audit_dir", type=Path, default=DEFAULT_TAU_AUDIT_DIR)
    parser.add_argument("--old_shared_dir", type=Path, default=DEFAULT_OLD_SHARED_DIR)
    parser.add_argument("--new_shared_dir", type=Path, default=DEFAULT_NEW_SHARED_DIR)
    parser.add_argument("--old_conditional_json", type=Path, default=DEFAULT_OLD_CONDITIONAL_JSON)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pop = load_csv(args.tau_audit_dir / "tau_population_summary.csv")
    counter = load_csv(args.tau_audit_dir / "tau_shared_wrong_counterfactual.csv")
    old_shared_seq = load_csv(args.old_shared_dir / "post_v2_shared_hard_cases_by_sequence.csv")
    new_shared_seq = load_csv(args.new_shared_dir / "post_v2_shared_hard_cases_by_sequence.csv")
    old_conditional = load_json(args.old_conditional_json)
    old_conditional_by_seq = {str(x["test_sequence"]): x for x in old_conditional["sequences"]}

    holdout_rows = []
    gated_wins_old = 0
    gated_wins_new = 0
    for seq, old_path in OLD_HOLDOUTS.items():
        new_path = NEW_HOLDOUTS[seq]
        old_map = model_map(load_csv(old_path))
        new_map = model_map(load_csv(new_path))
        old_geo = float(old_map["geometry_only"]["auroc"])
        if "geometry_plus_gated_parallax" in old_map:
            old_gated = float(old_map["geometry_plus_gated_parallax"]["auroc"])
        else:
            old_gated = float(old_conditional_by_seq[seq]["selected_row"]["test_auroc"])
        new_geo = float(new_map["geometry_only"]["auroc"])
        new_gated = float(new_map["geometry_plus_gated_parallax"]["auroc"])
        if old_gated > old_geo:
            gated_wins_old += 1
        if new_gated > new_geo:
            gated_wins_new += 1
        if (old_gated > old_geo) and (new_gated > new_geo):
            shift = "gated stays ahead"
        elif (old_gated <= old_geo) and (new_gated > new_geo):
            shift = "geometry -> gated"
        elif (old_gated > old_geo) and (new_gated <= new_geo):
            shift = "gated -> geometry"
        else:
            shift = "geometry stays ahead"
        holdout_rows.append(
            {
                "sequence": seq,
                "tau5_geometry": old_geo,
                "tau5_gated": old_gated,
                "tau75_geometry": new_geo,
                "tau75_gated": new_gated,
                "winner_shift": shift,
            }
        )

    shared_rows = [
        {
            "version": "tau5.0",
            "both_wrong": int(old_shared_seq["shared_wrong_count"].sum()),
            "gated_fixes_geo": int(old_shared_seq["gated_fixes_geo_count"].sum()),
            "geo_beats_gated": int(old_shared_seq["geo_beats_gated_count"].sum()),
        },
        {
            "version": "tau7.5",
            "both_wrong": int(new_shared_seq["shared_wrong_count"].sum()),
            "gated_fixes_geo": int(new_shared_seq["gated_fixes_geo_count"].sum()),
            "geo_beats_gated": int(new_shared_seq["geo_beats_gated_count"].sum()),
        },
    ]

    pop_rows = pop[pop["tau"].isin(["tau5.0", "tau7.5"])].to_dict("records")
    tau5_counter = counter[counter["tau"] == "tau5.0"].iloc[0].to_dict()
    tau75_counter = counter[counter["tau"] == "tau7.5"].iloc[0].to_dict()

    headline = [
        f"`tau=7.5` 确实收紧了标签边界：`future_high_gt_rot` 从 {int(pop[pop['tau']=='tau5.0']['future_high_gt_rot'].iloc[0])} 降到 {int(pop[pop['tau']=='tau7.5']['future_high_gt_rot'].iloc[0])}，而 `labeled_rows` 仍保持 {int(pop[pop['tau']=='tau7.5']['labeled_rows'].iloc[0])} 不变。",
        f"共享难例也同步下降：5 条 held-out 的 both-wrong 从 {shared_rows[0]['both_wrong']} 降到 {shared_rows[1]['both_wrong']}；反事实表里当前 590 条测试样本的 both-wrong 也从 {int(tau5_counter['both_wrong_counterfactual'])} 降到 {int(tau75_counter['both_wrong_counterfactual'])}。",
        f"但模型排序优势没有同步稳定：`geometry+gated_parallax` 在 tau5.0 时于 {gated_wins_old}/5 条 held-out 上领先，到了 tau7.5 仍然只是 {gated_wins_new}/5 条 held-out 上领先。",
        "因此这轮结果更适合被定性成“标签边界收紧有效”，而不是“新标签已经自动扶正了 gated 候选的 held-out 优势”。",
    ]

    judgement = [
        "tau=7.5 值得保留为认真候选标签版本，因为它用很小代价清掉了一批 future_high_gt_rot@K=1 的边界型共享难例。",
        "但 tau=7.5 还不够支持直接切主线：它虽然让 both-wrong 下降了，却没有让 geometry+gated_parallax 相对 geometry_only 的 held-out 优势整体变稳。",
        "所以更稳的后续是：把 tau=7.5 保留为标签对照主候选，同时继续审 future_high_gt_rot@K=1 的定义，而不是现在就整体替换 tau=5.0 主线。",
    ]

    payload = {
        "headline": headline,
        "judgement": judgement,
        "population_rows": pop_rows,
        "holdout_rows": holdout_rows,
        "shared_rows": shared_rows,
    }

    write_json(out_dir / "post_v2_tau75_rebuild_summary.json", payload)
    write_csv(out_dir / "post_v2_tau75_rebuild_holdout_comparison.csv", holdout_rows)
    write_csv(out_dir / "post_v2_tau75_rebuild_shared_comparison.csv", shared_rows)
    (out_dir / "post_v2_tau75_rebuild_summary.md").write_text(build_markdown(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
