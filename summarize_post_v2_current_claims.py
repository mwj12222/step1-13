#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = None
for p in [THIS_FILE.parent, *THIS_FILE.parent.parents]:
    if (p / "configs").is_dir() and ((p / "pipelines").is_dir() or (p / " pipelines").is_dir()):
        PROJECT_ROOT = p
        break
if PROJECT_ROOT is None:
    raise RuntimeError(f"Cannot locate project root from {THIS_FILE}")


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


def fmt(v, nd=4):
    return f"{float(v):.{nd}f}"


def model_map(rows: list[dict]) -> dict[str, dict]:
    return {str(r["model"]): r for r in rows}


def build_markdown(payload: dict) -> str:
    lines = []
    lines.append("# post_v2 current claims")
    lines.append("")
    lines.append("## 结论先行")
    lines.append("")
    for item in payload["headline_judgement"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## 证据收口")
    lines.append("")
    lines.append("| evidence block | key result | implication |")
    lines.append("| --- | --- | --- |")
    for row in payload["evidence_table"]:
        lines.append(f"| {row['evidence_block']} | {row['key_result']} | {row['implication']} |")
    lines.append("")
    lines.append("## 当前能说")
    lines.append("")
    for item in payload["allowed_claims"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## 当前不能说")
    lines.append("")
    for item in payload["forbidden_claims"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## 当前候选配方层级")
    lines.append("")
    lines.append("| tier | formulation | current status |")
    lines.append("| --- | --- | --- |")
    for row in payload["candidate_tiers"]:
        lines.append(f"| {row['tier']} | {row['formulation']} | {row['status']} |")
    lines.append("")
    lines.append("## 下一轮固定核心对照")
    lines.append("")
    for item in payload["core_controls"]:
        lines.append(f"- `{item}`")
    lines.append("")
    lines.append("## 下一步最该验证的问题")
    lines.append("")
    for item in payload["next_validation_questions"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Merge post_v2 reassessment and conditional-parallax evidence into a single current-claims document.")
    ap.add_argument(
        "--minimal_json",
        default=str(PROJECT_ROOT / "docs" / "research" / "init_risk_post_v2_minimal_sufficient_20260319" / "post_v2_minimal_sufficient_formal_summary.json"),
    )
    ap.add_argument(
        "--gated_multiseed_json",
        default=str(PROJECT_ROOT / "docs" / "research" / "init_risk_post_v2_core4_gated_multiseed_20260319" / "post_v2_core4_multiseed_formal_summary.json"),
    )
    ap.add_argument(
        "--holdout_city_night_json",
        default=str(PROJECT_ROOT / "docs" / "research" / "init_risk_post_v2_holdout_city_night_0_none_20260319" / "post_v2_sequence_holdout_summary.json"),
    )
    ap.add_argument(
        "--holdout_city_day_json",
        default=str(PROJECT_ROOT / "docs" / "research" / "init_risk_post_v2_holdout_city_day_2_mid_20260319" / "post_v2_sequence_holdout_summary.json"),
    )
    ap.add_argument(
        "--holdout_city_night_1_low_gated_json",
        default=str(PROJECT_ROOT / "docs" / "research" / "init_risk_post_v2_holdout_city_night_1_low_gated_20260319" / "post_v2_sequence_holdout_summary.json"),
    )
    ap.add_argument(
        "--holdout_city_day_0_none_gated_json",
        default="/mnt/g/Result/VIODE/post_v2_rebuild_9seq/post_v2_holdout_city_day_0_none_gated_20260319/formal_summary/post_v2_sequence_holdout_summary.json",
    )
    ap.add_argument(
        "--holdout_parking_lot_3_high_gated_json",
        default="/mnt/g/Result/VIODE/post_v2_rebuild_9seq/post_v2_holdout_parking_lot_3_high_gated_20260319/formal_summary/post_v2_sequence_holdout_summary.json",
    )
    ap.add_argument(
        "--conditional_json",
        default=str(PROJECT_ROOT / "docs" / "research" / "init_risk_post_v2_conditional_parallax_20260319" / "conditional_parallax_holdout_summary.json"),
    )
    ap.add_argument(
        "--shared_hard_cases_json",
        default=str(PROJECT_ROOT / "docs" / "research" / "init_risk_post_v2_shared_hard_cases_20260319" / "post_v2_shared_hard_cases_summary.json"),
    )
    ap.add_argument(
        "--shared_hard_case_structure_json",
        default=str(PROJECT_ROOT / "docs" / "research" / "init_risk_post_v2_shared_hard_case_structure_20260319" / "shared_hard_case_structure_summary.json"),
    )
    ap.add_argument(
        "--tau75_rebuild_json",
        default=str(PROJECT_ROOT / "docs" / "research" / "init_risk_post_v2_tau75_rebuild_20260320" / "post_v2_tau75_rebuild_summary.json"),
    )
    ap.add_argument(
        "--guard_rebuild_json",
        default=str(PROJECT_ROOT / "docs" / "research" / "init_risk_post_v2_guard_rebuild_20260320" / "post_v2_guard_rebuild_summary.json"),
    )
    ap.add_argument(
        "--future_high_gt_rot_both_wrong_segments_json",
        default=str(PROJECT_ROOT / "docs" / "research" / "init_risk_future_high_gt_rot_k1_both_wrong_segments_20260320" / "future_high_gt_rot_both_wrong_segments_summary.json"),
    )
    ap.add_argument(
        "--out_dir",
        default=str(PROJECT_ROOT / "docs" / "research" / "init_risk_post_v2_current_claims_20260319"),
    )
    args = ap.parse_args()

    minimal = load_json(Path(args.minimal_json).expanduser().resolve())
    gated_multiseed = load_json(Path(args.gated_multiseed_json).expanduser().resolve())
    holdout_night = load_json(Path(args.holdout_city_night_json).expanduser().resolve())
    holdout_day = load_json(Path(args.holdout_city_day_json).expanduser().resolve())
    holdout_night_1_low_gated = load_json(Path(args.holdout_city_night_1_low_gated_json).expanduser().resolve())
    holdout_city_day_0_none_gated = load_json(Path(args.holdout_city_day_0_none_gated_json).expanduser().resolve())
    holdout_parking_lot_3_high_gated = load_json(Path(args.holdout_parking_lot_3_high_gated_json).expanduser().resolve())
    conditional = load_json(Path(args.conditional_json).expanduser().resolve())
    shared_hard_cases = load_json(Path(args.shared_hard_cases_json).expanduser().resolve())
    shared_hard_case_structure = load_json(Path(args.shared_hard_case_structure_json).expanduser().resolve())
    tau75_rebuild = load_json(Path(args.tau75_rebuild_json).expanduser().resolve())
    guard_rebuild = load_json(Path(args.guard_rebuild_json).expanduser().resolve())
    future_high_gt_rot_both_wrong_segments = load_json(
        Path(args.future_high_gt_rot_both_wrong_segments_json).expanduser().resolve()
    )

    minimal_main = {str(r["name"]): r for r in minimal["main_results"]}
    minimal_rows = {str(r["name"]): r for r in minimal["minimal_sufficient_rows"]}
    holdout_night_map = model_map(holdout_night["rows"])
    holdout_day_map = model_map(holdout_day["rows"])
    holdout_night_1_low_gated_map = model_map(holdout_night_1_low_gated["rows"])
    holdout_city_day_0_none_gated_map = model_map(holdout_city_day_0_none_gated["rows"])
    holdout_parking_lot_3_high_gated_map = model_map(holdout_parking_lot_3_high_gated["rows"])
    cond_seq = {str(r["test_sequence"]): r for r in conditional["sequences"]}
    gated_multiseed_agg = {str(r["model"]): r for r in gated_multiseed["aggregate_rows"]}

    evidence_table = [
        {
            "evidence_block": "9seq single-split reassessment",
            "key_result": f"full={fmt(minimal_main['full_anchor']['auroc'])}, geometry_only={fmt(minimal_main['geometry_only']['auroc'])}, drop_front={fmt(minimal_main['drop_front']['auroc'])}, rule={fmt(minimal_main['rule_q_post_geom_only']['auroc'])}",
            "implication": "full 不是默认稳定配方；geometry 与 drop-front 都比 full 更稳。",
        },
        {
            "evidence_block": "9seq minimal sufficient set",
            "key_result": f"geometry+parallax={fmt(minimal_rows['post_v2_geometry_plus_parallax']['auroc'])}, geometry_only={fmt(minimal_rows['post_v2_geometry_only']['auroc'])}, full={fmt(minimal_rows['post_v2_full_anchor']['auroc'])}",
            "implication": "在单 split 上，最强增量来自 parallax，而不是整个 candidate/front block。",
        },
        {
            "evidence_block": "3-seed split repeat (gated core4)",
            "key_result": f"mean AUROC: gated={fmt(gated_multiseed_agg['geometry_plus_gated_parallax']['mean_auroc'])}, geometry={fmt(gated_multiseed_agg['geometry_only']['mean_auroc'])}, drop_front={fmt(gated_multiseed_agg['drop_front']['mean_auroc'])}, full={fmt(gated_multiseed_agg['full_anchor']['mean_auroc'])}",
            "implication": "在同一 9seq 数据池内，geometry+gated_parallax 在 3/3 seeds 上都是第一，并稳定优于 geometry_only 与 full。",
        },
        {
            "evidence_block": "explicit held-out: city_night/0_none",
            "key_result": f"geometry={fmt(holdout_night_map['geometry_only']['auroc'])}, gp={fmt(holdout_night_map['geometry_plus_parallax']['auroc'])}, gated={fmt(cond_seq['city_night/0_none']['selected_row']['test_auroc'])}",
            "implication": "plain parallax 不再领先；条件激活能回收一部分增益，但仍略低于 geometry_only。",
        },
        {
            "evidence_block": "explicit held-out: city_day/2_mid",
            "key_result": f"geometry={fmt(holdout_day_map['geometry_only']['auroc'])}, gp={fmt(holdout_day_map['geometry_plus_parallax']['auroc'])}, gated={fmt(cond_seq['city_day/2_mid']['selected_row']['test_auroc'])}",
            "implication": "条件化 parallax 已经反超 plain parallax，也反超 geometry_only。",
        },
        {
            "evidence_block": "explicit held-out: city_night/1_low (gated core4)",
            "key_result": f"gated={fmt(holdout_night_1_low_gated_map['geometry_plus_gated_parallax']['auroc'])}, geometry={fmt(holdout_night_1_low_gated_map['geometry_only']['auroc'])}, full={fmt(holdout_night_1_low_gated_map['full_anchor']['auroc'])}",
            "implication": "在新的 held-out 端点上，geometry+gated_parallax 明确拿到第一，进一步说明它不只是 split-seed 内强势。",
        },
        {
            "evidence_block": "explicit held-out: city_day/0_none (gated core4)",
            "key_result": f"full={fmt(holdout_city_day_0_none_gated_map['full_anchor']['auroc'])}, gated={fmt(holdout_city_day_0_none_gated_map['geometry_plus_gated_parallax']['auroc'])}, geometry={fmt(holdout_city_day_0_none_gated_map['geometry_only']['auroc'])}",
            "implication": "在这个静态端点上，gated 仅小幅高于 geometry，但仍未超过 full，说明它不是处处占优的全局替代。",
        },
        {
            "evidence_block": "explicit held-out: parking_lot/3_high (gated core4)",
            "key_result": f"gated={fmt(holdout_parking_lot_3_high_gated_map['geometry_plus_gated_parallax']['auroc'])}, drop_front={fmt(holdout_parking_lot_3_high_gated_map['drop_front']['auroc'])}, geometry={fmt(holdout_parking_lot_3_high_gated_map['geometry_only']['auroc'])}",
            "implication": "在高动态端点上，gated 再次拿到第一，说明它对一部分 harder sequence 的补益是可复现的。",
        },
        {
            "evidence_block": "shared hard cases across 5 held-outs",
            "key_result": "both-wrong=192/590, gated_fixes_geo=35, geo_beats_gated=29",
            "implication": "当前剩余瓶颈已明显转向共享难例；geometry 与 gated 共同失败的样本仍然很多，后续需要看标签边界或缺失特征。",
        },
        {
            "evidence_block": "shared hard-case structure (horizon/trigger/sequence)",
            "key_result": "all shared_wrong come from K=1; 166/192 are positive; future_high_gt_rot=134, future_solver_fail=26",
            "implication": "共享难例主导项更像短时标签边界正样本，而 gated 修掉的是高视差高旋转子集，说明 recoverability 锚点有用但不是主体瓶颈。",
        },
        {
            "evidence_block": "future_high_gt_rot@K=1 both-wrong finer segments",
            "key_result": "future_rot 5_to_7p5=49/134, parallax lt_12p7=59/134, densest cell=5_to_7p5 x lt_12p7 = 23/134",
            "implication": "主体共享难例更像轻中度 future rotation + 低当前视差的 city_day 边界样本；同一触发定义在 city_day 与 city_night/parking_lot 上语义并不一致。",
        },
        {
            "evidence_block": "tau=7.5 label-control rebuild",
            "key_result": "future_high_gt_rot 338->227, both-wrong 192->156, but gated lead shrinks from 4/5 to 2/5 held-outs",
            "implication": "tau=7.5 适合作为标签对照主候选，因为它能清理边界样本；但它没有自动扶正 gated 的 held-out 排序优势，因此不能直接替换主线。",
        },
        {
            "evidence_block": "recoverability-aware guard rebuild",
            "key_result": "both-wrong 156->134, gated_fixes_geo 14->33, but gated becomes 0/5 held-out winner while drop_front becomes 4/5 winner",
            "implication": "recoverability-aware guard 在标签层有选择性价值，但它改写了任务排序结构，不支持直接切换主线标签协议。",
        },
    ]

    headline_judgement = [
        "post_v2 仍然有可学习信号，但当前结构解释还不能写死成 geometry-dominated 或 candidate-dominated。",
        "full 不是当前默认主配方，这点已经稳定；front block 的当前混法会伤害泛化，这点也已经稳定。",
        "geometry 是当前较强的排序基底；plain geometry+parallax 比 full 更值得推进，但它跨新增 held-out 的优势不稳，因此不能再作为主候选。",
        "geometry+gated_parallax 已经成为当前最优先推进的 Q_post v_next 候选：它在 gated multi-seed 中 3/3 第一，并在 city_day/2_mid、city_night/1_low、parking_lot/3_high 这些 held-out 上领先；但在 city_night/0_none 上仍低于 geometry，在 city_day/0_none 上也未超过 full，因此整体 held-out 证据仍然是 mixed。",
        "共享难例已经成为当前更主要的瓶颈：跨 5 条 held-out，geometry 与 gated 仍共同误判了 192/590 条测试样本；而且这些难例几乎都落在 K=1 的 next-window 触发正样本上，说明后续需要转向标签边界与 recoverability 锚点。",
        f"更细分段后，这批 `future_high_gt_rot@K=1` 的 `both-wrong` 主体被进一步压缩成 `轻中度 future rotation + 低当前视差`：主导未来旋转带是 `{future_high_gt_rot_both_wrong_segments['dominant_bands'][0]['band']}`，主导当前视差带是 `{future_high_gt_rot_both_wrong_segments['dominant_bands'][1]['band']}`，而且主体集中在 city_day 端点。",
        "tau=7.5 目前应被保留为标签对照主候选，而不是主线替换版：它能收紧 `future_high_gt_rot@K=1` 的边界样本，但没有把 gated 相对 geometry 的 held-out 优势整体变稳。",
        "recoverability-aware guard 有标签层价值，但当前不支持主线切换：它进一步减少了共享难例，却把 held-out 最优结构推向了 drop_front/full，而不是让 geometry+gated_parallax 更稳。",
    ]

    allowed_claims = [
        "可以说 post_v2 在当前 9seq 数据池与新增 held-out 检查下，仍保持可学习排序信号。",
        "可以说 geometry 是当前较强的后验排序信号，而 front block 当前混法有害。",
        "可以说 full 不再适合作为默认稳定 post 配方，应继续保留为反例锚点。",
        "可以说 plain geometry+parallax 仍然优于 full，但它跨新增 held-out 的优势不稳，因此不再适合作为当前主候选。",
        "可以说 geometry+gated_parallax 是当前最优先推进的候选配方，因为它在 gated multi-seed 中 3/3 第一，并在多个新 held-out 端点上取得领先。",
        "可以说 geometry+gated_parallax 当前相对 geometry_only 的证据状态是 mixed but improving：它在 city_day/2_mid、city_night/1_low、parking_lot/3_high 上领先，在 city_night/0_none 上仍略逊，在 city_day/0_none 上仅小幅领先但未超过 full。",
        "可以说共享难例已成为当前主要瓶颈，因为跨 5 条 held-out 仍有 192/590 条 both-wrong 样本。",
        "可以说当前共享难例更像边界型难例而不是明显几何崩坏，因为它们集中在 K=1 的 next-window 触发正样本，整体呈现更低 parallax、更低 reproj_p90、但 tri_points 仍不低的特征。",
        "可以说 `future_high_gt_rot@K=1` 的 both-wrong 主体已被进一步定位为轻中度 future rotation + 低当前视差，尤其集中在 city_day/0_none 与 city_day/2_mid 这两组边界段上。",
        "可以说 recoverability 锚点仍是次级缺口，因为 gated 主要修掉的是高视差高旋转正样本，而不是共享难例主体。",
        "可以说 tau=7.5 是当前最值得保留的标签对照主候选，因为它能明显清理 future_high_gt_rot 的边界样本，同时不改变 labeled_rows。",
        "可以说 recoverability-aware guard 值得保留为定义实验结果，因为它在标签层比单纯 tau=7.5 更有选择性。",
    ]

    forbidden_claims = [
        "不能把 post_v2 明确写成已证明的 geometry-dominated 任务。",
        "不能再把 full 写成默认更强或更完整的主配方。",
        "不能把 DirRisk 写成当前更强主线，因为它并未带来额外收益。",
        "不能把 plain geometry+parallax 直接写成已升级完成的最终首选配方。",
        "不能把 geometry+gated_parallax 直接写成最终定稿，因为它虽然已是当前主候选，但更广 held-out 验证仍未完成。",
        "不能把当前结果写成跨场景最终泛化结论，也不能写成系统级 gate 收益已闭环。",
        "不能说 post 结构已经收敛。",
        "不能把当前共享难例简单写成纯几何失败，因为主体样本的几何分数和重投影并未明显崩坏。",
        "不能因为 tau=7.5 清理了边界样本，就直接把它升级成替代 tau=5.0 的默认主标签版本。",
        "不能因为 recoverability-aware guard 进一步减少了共享难例，就直接把它切进主线标签协议；它当前更像定义实验结果，而不是稳定升级版。",
    ]

    candidate_tiers = [
        {
            "tier": "Anchor",
            "formulation": "geometry_only",
            "status": "当前最稳的几何基底；在 city_night/0_none 上仍是最强，在其他端点则开始被 gated 挑战。",
        },
        {
            "tier": "Primary candidate",
            "formulation": "geometry + gated_parallax",
            "status": "当前最优先推进的 Q_post v_next 候选；在 gated multi-seed 中 3/3 第一，并在多个 held-out 上领先，但整体 held-out 仍是 mixed。",
        },
        {
            "tier": "Label-control candidate",
            "formulation": "Y_bad_v2_min_default with tau_gt_rot_med_deg=7.5",
            "status": "当前最值得保留的标签对照主候选；它明显削减 future_high_gt_rot 边界样本与共享难例，但还没有带来 geometry+gated_parallax 排序优势的同步稳定。",
        },
        {
            "tier": "Definition experiment",
            "formulation": "future_high_gt_rot with current_parallax guard (>=12.693 px)",
            "status": "标签层更有选择性，但训练级会把 held-out 最优结构推向 drop_front/full；当前只适合作为定义实验结果，不支持主线切换。",
        },
        {
            "tier": "Diagnostic comparator",
            "formulation": "geometry + parallax",
            "status": "保留为诊断对照，不再作为主候选；新增 held-out 已表明其 sequence dependence 明显。",
        },
        {
            "tier": "Ablation anchor",
            "formulation": "drop_front",
            "status": "继续保留为 block-level 对照，用来证明 front mixing 的伤害。",
        },
        {
            "tier": "Counterexample anchor",
            "formulation": "full_anchor",
            "status": "作为反例与历史配方保留，不再作为默认主模型。",
        },
    ]

    core_controls = [
        "geometry_only",
        "geometry + gated_parallax",
        "drop_front",
        "full_anchor",
    ]

    next_validation_questions = [
        "geometry+gated_parallax 是否在更多新增 held-out sequence 上持续不差于 geometry_only？",
        "geometry+gated_parallax 的增益是否集中在特定 high-parallax / hard-positive 段，而不是稳定的全局提升？",
        "共享难例是否集中在特定 horizon / trigger / sequence 结构上，从而提示标签边界或 recoverability 锚点缺失？",
        "city_day/0_none 与 city_day/2_mid 这两组主导边界段，是否存在一致的 motion / trigger 邻域模式，从而支持更细的 sequence-aware 标签边界解释？",
        "future_high_gt_rot@K=1 是否应先调整旋转阈值，再考虑改 horizon 或引入复合触发定义？",
        "recoverability-aware guard 是否值得继续留在定义实验线，而不是升级为主线标签协议？",
    ]

    payload = {
        "headline_judgement": headline_judgement,
        "evidence_table": evidence_table,
        "allowed_claims": allowed_claims,
        "forbidden_claims": forbidden_claims,
        "candidate_tiers": candidate_tiers,
        "core_controls": core_controls,
        "next_validation_questions": next_validation_questions,
        "source_docs": {
            "minimal_sufficient": str(Path(args.minimal_json).expanduser().resolve()),
            "gated_multiseed": str(Path(args.gated_multiseed_json).expanduser().resolve()),
            "holdout_city_night": str(Path(args.holdout_city_night_json).expanduser().resolve()),
            "holdout_city_day": str(Path(args.holdout_city_day_json).expanduser().resolve()),
            "holdout_city_night_1_low_gated": str(Path(args.holdout_city_night_1_low_gated_json).expanduser().resolve()),
            "holdout_city_day_0_none_gated": str(Path(args.holdout_city_day_0_none_gated_json).expanduser().resolve()),
            "holdout_parking_lot_3_high_gated": str(Path(args.holdout_parking_lot_3_high_gated_json).expanduser().resolve()),
            "conditional": str(Path(args.conditional_json).expanduser().resolve()),
            "shared_hard_cases": str(Path(args.shared_hard_cases_json).expanduser().resolve()),
            "shared_hard_case_structure": str(Path(args.shared_hard_case_structure_json).expanduser().resolve()),
            "tau75_rebuild": str(Path(args.tau75_rebuild_json).expanduser().resolve()),
            "guard_rebuild": str(Path(args.guard_rebuild_json).expanduser().resolve()),
        },
    }

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "post_v2_current_claims.json", payload)
    write_text(out_dir / "post_v2_current_claims.md", build_markdown(payload))
    write_csv(out_dir / "post_v2_current_claims_evidence.csv", list(evidence_table[0].keys()), evidence_table)
    print(f"[PostV2CurrentClaims] saved -> {out_dir}")


if __name__ == "__main__":
    main()
