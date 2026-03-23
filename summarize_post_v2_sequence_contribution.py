#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict


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
    if v is None or v == "":
        return "-"
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return str(v)


def main():
    ap = argparse.ArgumentParser(description="Summarize sequence-level contribution of newly added post_v2 sequences.")
    ap.add_argument("--structure_shift_summary_json", required=True)
    ap.add_argument("--structure_drift_json", required=True)
    ap.add_argument("--y_bad_v2_audit_json", required=True)
    ap.add_argument("--y_bad_v2_labels_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    shift = load_json(Path(args.structure_shift_summary_json))
    drift = load_json(Path(args.structure_drift_json))
    audit = load_json(Path(args.y_bad_v2_audit_json))
    label_rows = load_csv_rows(Path(args.y_bad_v2_labels_csv))
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    old_cov = shift["legacy_dataset"]["coverage_summary"]
    new_cov = shift["new_dataset"]["coverage_summary"]
    by_seq = audit["by_sequence"]
    seq_rows = drift["sequence_contribution"]

    seq_map = {row["sequence"]: row for row in seq_rows}
    focus_sequences = ["city_night/2_mid", "parking_lot/2_mid"]

    focus_rows = []
    total_new_labeled = 0
    total_new_bad = 0
    total_new_rows = 0
    trigger_by_sequence = defaultdict(lambda: defaultdict(int))
    for row in label_rows:
        seq = row.get("sequence", "")
        trig = row.get("Y_bad_v2_min_trigger", "")
        if not seq or not trig:
            continue
        trigger_by_sequence[seq][trig] += 1

    for seq in focus_sequences:
        a = by_seq.get(seq, {})
        d = seq_map.get(seq, {})
        labeled = int(a.get("labeled_rows", 0))
        bad = int(a.get("bad_rows", 0))
        rows = int(a.get("num_rows", 0))
        total_new_rows += rows
        total_new_labeled += labeled
        total_new_bad += bad
        trigger_map = dict(trigger_by_sequence.get(seq, {}))
        focus_rows.append(
            {
                "sequence": seq,
                "split": d.get("split", ""),
                "rows": rows,
                "accept_rows": int(a.get("accept_rows", 0)),
                "labeled_rows": labeled,
                "bad_rows": bad,
                "bad_ratio": (bad / labeled) if labeled else None,
                "mean_p_hat": d.get("mean_p_hat"),
                "brier": d.get("brier"),
                "auroc": d.get("auroc"),
                "top_feature_signature": d.get("top_feature_signature", ""),
                "trigger_counts": json.dumps(trigger_map, ensure_ascii=False),
            }
        )

    old_labeled = int(old_cov["labeled_rows"])
    new_labeled = int(new_cov["labeled_rows"])
    labeled_increase = new_labeled - old_labeled

    total_trigger_counts = {}
    for seq in focus_sequences:
        for k, v in trigger_by_sequence.get(seq, {}).items():
            total_trigger_counts[k] = total_trigger_counts.get(k, 0) + int(v)

    contribution = {
        "dataset_level": {
            "old_labeled_rows": old_labeled,
            "new_labeled_rows": new_labeled,
            "labeled_row_increase": labeled_increase,
            "old_bad_ratio": old_cov["bad_ratio_labeled"],
            "new_bad_ratio": new_cov["bad_ratio_labeled"],
            "focus_sequence_rows": total_new_rows,
            "focus_sequence_labeled_rows": total_new_labeled,
            "focus_sequence_bad_rows": total_new_bad,
            "focus_sequence_labeled_share_of_new_dataset": (total_new_labeled / new_labeled) if new_labeled else None,
            "focus_sequence_bad_share_of_new_dataset": (total_new_bad / int(round(new_labeled * new_cov["bad_ratio_labeled"]))) if new_labeled else None,
        },
        "focus_sequences": focus_rows,
        "focus_sequence_trigger_counts": total_trigger_counts,
        "headline_takeaways": [
            "city_night/2_mid 和 parking_lot/2_mid 不是简单增加了原始资产，而是实际增加了 post_v2 的可标注窗口人口。",
            "这两条新序列对 5 条集新增的 363 个 labeled rows 只贡献了 10 个左右的净增量中的一部分，但它们改变了 sequence mix，并把新场景风格引入了 train/val。",
            "parking_lot/2_mid 在 train 中行数很少，但 feature signature 偏移明显，更像是“少量但风格强”的新序列；这类序列更容易改写 block ablation 结论。",
            "city_night/2_mid 当前只落在 val，且只有 3 条 labeled rows；它对 split 改善帮助有限，但它暴露了当前 val 极薄的问题。 ",
        ],
        "claim_safe": [
            "可以说新序列已经把 post_v2 的标签人口和场景多样性向前推进了一步。",
            "可以说新序列带来的主要收益不只是样本数增加，还包括 sequence mix 的改变和结构结论的纠偏。",
            "可以说当前最需要继续补的是 split 空缺，而不是继续堆相同类型的 2_mid。"
        ],
        "claim_forbidden": [
            "不能把 city_night/2_mid 和 parking_lot/2_mid 的加入写成已经解决 split 偏置。",
            "不能把当前两条新序列的结构影响写成稳定的跨场景规律。",
            "不能把少量新序列造成的 block 变化直接写成 front 已经主导或 geometry 已经失效。"
        ],
        "next_expansion_recommendation": [
            "下一轮扩样本优先补非 2_mid，并优先补能进入 val/test 的序列。",
            "比起继续加一个新的 2_mid，更值得优先尝试 city_night/1_low 和 parking_lot/1_low，先看它们是否能提供更健康的 accept_rows 和 future_horizon_ok_rows。",
            "在继续扩之前，不需要先升级模型；先用新样本回答 split 与 sequence mix 是否在主导当前结构漂移。"
        ],
    }

    md = f"""# post_v2 新序列贡献小结

## 结论先行

- `city_night/2_mid` 和 `parking_lot/2_mid` 已经不只是新资产，而是已经真实进入 `post_v2_min_default` 的标签人口。
- 这两条新序列的最大价值，不只是把 labeled rows 从 `{old_labeled}` 扩到 `{new_labeled}`，更在于它们把新的 sequence style 引入了当前 5 条集，进而促发了我们对旧 block 叙事的纠偏。
- 但它们还没有解决 split 偏置：`city_night/2_mid` 当前只在 `val`，且 labeled rows 很少；`parking_lot/2_mid` 虽进入 `train`，但样本量仍然很小。

## 数据级变化

- 旧 4 条集 labeled rows: `{old_labeled}`
- 新 5 条集 labeled rows: `{new_labeled}`
- labeled rows 净增加: `{labeled_increase}`
- 新 5 条集整体 bad ratio: `{fmt(new_cov["bad_ratio_labeled"])}`
- 两条新序列合计 labeled rows: `{total_new_labeled}`
- 两条新序列合计 bad rows: `{total_new_bad}`
- 两条新序列占新 5 条集 labeled rows 比例: `{fmt((total_new_labeled / new_labeled) if new_labeled else None)}`
- 两条新序列 trigger 合计: `{json.dumps(total_trigger_counts, ensure_ascii=False)}`

## 两条新序列各自带来了什么

| 序列 | split | rows | accept_rows | labeled_rows | bad_rows | bad_ratio | AUROC | Brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
""" + "\n".join(
        f"| {r['sequence']} | {r['split']} | {r['rows']} | {r['accept_rows']} | {r['labeled_rows']} | {r['bad_rows']} | {fmt(r['bad_ratio'])} | {fmt(r['auroc'])} | {fmt(r['brier'])} |"
        for r in focus_rows
    ) + f"""

## 解释性收口

**city_night/2_mid**
- 主要问题不是“完全没贡献”，而是它目前只贡献了极薄的 `val` 样本。
- 它的 `front_coverage_ratio` 明显更低，`parallax_px_candidate` 和 `tri_points_candidate` 明显更高，说明它的序列风格和原 `city_day` 组不一样。
- 它目前触发的是混合型短时不稳：既有 `future_high_gt_rot`，也有 `future_reset`，说明它更像“少量但尖锐”的验证序列。
- 这类极薄 `val` 序列更像是在提醒我们：当前 `val` 的结构仍然太脆，不能拿它支撑稳定泛化说法。

**parking_lot/2_mid**
- 它的 labeled rows 不多，但已经足够进入 `train`，并改变当前 train mix。
- 它最明显的签名是：`front_coverage_ratio` 更低、`parallax_px_candidate` 更高、`reproj` 更低。
- 它当前主要补的是 `future_high_gt_rot` 触发，而不是大量 `future_reset`；这说明它更像在补“短时姿态退化”这类不稳样本。
- 这使它更像“少量但风格很强”的新序列；这种序列很容易改写 block ablation 的相对排序。

## 批判性判断

- 当前两条新序列的真正贡献，是让我们看到：旧的 `geometry-dominated` 叙事并不稳，结构结论会随着 sequence mix 变化而漂移。
- 但这一步还不是“split 已经解决”。现在只能说：我们开始看到结构纠偏，而不是已经得到稳健新结构。
- 因为 `city_night/2_mid` 只在 `val`，`parking_lot/2_mid` 又很小，所以当前更像是一次**结构敏感性暴露**，而不是一次稳定结构重建。

## 现在能说与不能说

- Allowed: 可以说两条新序列已经真实增加了 `post_v2` 的标签人口与场景多样性。
- Allowed: 可以说它们的价值不只是增加样本数，更是改变了 sequence mix，并触发了对旧结构结论的纠偏。
- Allowed: 可以说下一轮扩样本应优先补 split 空缺，而不是继续平均堆更多 `2_mid`。
- Forbidden: 不能把这两条新序列的加入写成已经解决 split 偏置。
- Forbidden: 不能把当前 block 变化直接写成稳定的新规律。
- Forbidden: 不能把 `front_only` 上升解读成前端已经成为 post_v2 主导信号。

## 下一步建议

- 下一轮优先扩非 `2_mid` 序列，尤其优先考虑 `city_night/1_low` 与 `parking_lot/1_low`。
- 目标不是单纯再加样本，而是让 `val/test` 不再由单一场景或单一 dynamic level 代表。
- 在补这两条之前，不需要先升级模型；先继续把 sequence mix 和 split 纠偏做实。
"""

    write_json(out_dir / "post_v2_sequence_contribution_summary.json", contribution)
    write_csv(
        out_dir / "post_v2_sequence_contribution.csv",
        fieldnames=[
            "sequence",
            "split",
            "rows",
            "accept_rows",
            "labeled_rows",
            "bad_rows",
            "bad_ratio",
            "mean_p_hat",
            "brier",
            "auroc",
            "top_feature_signature",
            "trigger_counts",
        ],
        rows=focus_rows,
    )
    write_text(out_dir / "post_v2_sequence_contribution_summary.md", md)
    print(f"[post_v2_sequence_contribution] saved -> {out_dir}")


if __name__ == "__main__":
    main()
