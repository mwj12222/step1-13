#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_LABELS_CSV = Path(
    "/mnt/g/Result/VIODE/post_v2_expand_valtest_20260319/step11_and_label_audit/"
    "y_bad_v2_min_default_9seq/y_bad_v2_min_labels.csv"
)
DEFAULT_RISK_DATASET_CSV = Path(
    "/mnt/g/Result/VIODE/post_v2_rebuild_9seq/risk_dataset_9seq_seed20260432/risk_dataset.csv"
)
DEFAULT_SHARED_HARD_CASES_CSV = Path(
    "docs/research/init_risk_post_v2_shared_hard_cases_20260319/post_v2_shared_hard_cases_rows.csv"
)
DEFAULT_OUT_DIR = Path(
    "docs/research/init_risk_future_high_gt_rot_k1_semantics_20260320"
)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: dict) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def fmt(v, nd: int = 4) -> str:
    if pd.isna(v):
        return "-"
    return f"{float(v):.{nd}f}"


def build_markdown(payload: dict) -> str:
    lines = []
    lines.append("# future_high_gt_rot@K=1 semantics audit")
    lines.append("")
    lines.append("## 结论先行")
    lines.append("")
    for item in payload["headline"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## 触发未来窗口状态")
    lines.append("")
    lines.append("| future gate_post | rows | ratio |")
    lines.append("| --- | --- | --- |")
    for row in payload["future_gatepost_summary"]:
        lines.append(f"| {row['future_gate_post']} | {row['rows']} | {fmt(row['ratio'])} |")
    lines.append("")
    lines.append("## 阈值保留率")
    lines.append("")
    lines.append("| tau | cohort | kept_rows | total_rows | keep_rate |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in payload["tau_retention"]:
        lines.append(
            f"| {row['tau']} | {row['cohort']} | {row['kept_rows']} | {row['total_rows']} | {fmt(row['keep_rate'])} |"
        )
    lines.append("")
    lines.append("## 当前窗口 vs 触发未来窗口画像")
    lines.append("")
    lines.append("| cohort | rows | future gt_rot | future reproj | current parallax | current reproj_p90 | current Q_post_geom_only |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in payload["profile_summary"]:
        lines.append(
            f"| {row['cohort']} | {row['rows']} | {fmt(row['future_gt_rot_med_deg'])} | {fmt(row['future_reproj_med_px'])} | {fmt(row['current_parallax_px_candidate'])} | {fmt(row['current_reproj_p90_px'])} | {fmt(row['current_Q_post_geom_only'])} |"
        )
    lines.append("")
    lines.append("## 判断")
    lines.append("")
    for item in payload["judgement"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit future_high_gt_rot@K=1 semantics.")
    parser.add_argument("--labels_csv", type=Path, default=DEFAULT_LABELS_CSV)
    parser.add_argument("--risk_dataset_csv", type=Path, default=DEFAULT_RISK_DATASET_CSV)
    parser.add_argument("--shared_hard_cases_csv", type=Path, default=DEFAULT_SHARED_HARD_CASES_CSV)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(args.labels_csv)
    risk = pd.read_csv(
        args.risk_dataset_csv,
        usecols=[
            "sample_uid",
            "sequence",
            "window_id",
            "Q_post_geom_only",
            "parallax_px_candidate",
            "reproj_p90_px",
            "tri_points_candidate",
        ],
    )
    shared = pd.read_csv(
        args.shared_hard_cases_csv,
        usecols=["sample_uid", "sequence", "shared_wrong", "gated_fixes_geo", "geo_beats_gated"],
    )

    cur = labels[labels["Y_bad_v2_min_default_trigger"] == "future_high_gt_rot"].copy()
    cur["sample_uid"] = cur["sample_uid"].astype(str)
    cur["sequence"] = cur["sequence"].astype(str)
    cur["variant_tag"] = cur["variant_tag"].astype(str)
    cur["window_id"] = pd.to_numeric(cur["window_id"], errors="coerce")
    cur["trigger_window_id"] = pd.to_numeric(cur["Y_bad_v2_min_default_trigger_window_id"], errors="coerce")

    future = labels[
        [
            "sequence",
            "variant_tag",
            "window_id",
            "gate_post",
            "solver_ok",
            "geom_ok",
            "accepted_but_bad_geom",
            "reproj_med_px",
            "gt_rot_med_deg",
            "gt_trans_dir_med_deg",
            "success_strict",
        ]
    ].copy()
    future = future.rename(columns={c: f"future_{c}" for c in future.columns if c not in ["sequence", "variant_tag"]})

    merged = cur.merge(
        future,
        left_on=["sequence", "variant_tag", "trigger_window_id"],
        right_on=["sequence", "variant_tag", "future_window_id"],
        how="left",
    )
    merged = merged.merge(risk, on=["sample_uid", "sequence", "window_id"], how="left")
    merged = merged.merge(shared, on=["sample_uid", "sequence"], how="left")
    for col in ["shared_wrong", "gated_fixes_geo", "geo_beats_gated"]:
        merged[col] = merged[col].fillna(0).astype(int)

    gatepost = (
        merged.groupby("future_gate_post")
        .size()
        .reset_index(name="rows")
        .sort_values("rows", ascending=False)
    )
    gatepost["ratio"] = gatepost["rows"] / len(merged)

    tau_rows = []
    tau_values = [5.0, 7.5, 10.0]
    cohorts = {
        "all_future_high_gt_rot": merged.index == merged.index,
        "shared_wrong": merged["shared_wrong"] == 1,
        "gated_fixes_geo": merged["gated_fixes_geo"] == 1,
        "geo_beats_gated": merged["geo_beats_gated"] == 1,
    }
    for tau in tau_values:
        keep = pd.to_numeric(merged["future_gt_rot_med_deg"], errors="coerce") > tau
        for cohort_name, mask in cohorts.items():
            total = int(mask.sum())
            kept = int((keep & mask).sum())
            tau_rows.append(
                {
                    "tau": tau,
                    "cohort": cohort_name,
                    "kept_rows": kept,
                    "total_rows": total,
                    "keep_rate": (kept / total) if total else None,
                }
            )
    tau_df = pd.DataFrame(tau_rows)

    profile_rows = []
    for cohort_name, mask in cohorts.items():
        df = merged[mask].copy()
        profile_rows.append(
            {
                "cohort": cohort_name,
                "rows": int(len(df)),
                "future_gt_rot_med_deg": pd.to_numeric(df["future_gt_rot_med_deg"], errors="coerce").mean(),
                "future_reproj_med_px": pd.to_numeric(df["future_reproj_med_px"], errors="coerce").mean(),
                "current_parallax_px_candidate": pd.to_numeric(df["parallax_px_candidate"], errors="coerce").mean(),
                "current_reproj_p90_px": pd.to_numeric(df["reproj_p90_px"], errors="coerce").mean(),
                "current_Q_post_geom_only": pd.to_numeric(df["Q_post_geom_only"], errors="coerce").mean(),
            }
        )
    profile_df = pd.DataFrame(profile_rows)

    all_accept_ratio = gatepost.loc[gatepost["future_gate_post"] == "accept", "ratio"]
    shared_keep_75 = tau_df[(tau_df["tau"] == 7.5) & (tau_df["cohort"] == "shared_wrong")]["keep_rate"].iloc[0]
    gfix_keep_75 = tau_df[(tau_df["tau"] == 7.5) & (tau_df["cohort"] == "gated_fixes_geo")]["keep_rate"].iloc[0]
    shared_keep_10 = tau_df[(tau_df["tau"] == 10.0) & (tau_df["cohort"] == "shared_wrong")]["keep_rate"].iloc[0]
    gfix_keep_10 = tau_df[(tau_df["tau"] == 10.0) & (tau_df["cohort"] == "gated_fixes_geo")]["keep_rate"].iloc[0]

    headline = [
        f"`future_high_gt_rot@K=1` 的触发未来窗口几乎是纯 `accept` 分支（accept ratio={fmt(all_accept_ratio.iloc[0] if len(all_accept_ratio) else None)})，说明这条 trigger 本质上是在抓“下一窗口仍被系统接受、但 GT 旋转已超阈”的情形，而不是和 reset/solver/geom fail 混在一起。",
        f"`tau` 收紧能清掉边界样本，但选择性并不强：`tau=7.5` 下共享难例保留率 {fmt(shared_keep_75)}，`gated` 修复样本保留率 {fmt(gfix_keep_75)}；到 `tau=10.0` 时两者分别变成 {fmt(shared_keep_10)} 和 {fmt(gfix_keep_10)}。",
        "这说明阈值收紧是有效的粗粒度标签清理，但它并不会天然只删掉边界错例而保住 recoverability 真正有帮助的那批高运动样本。",
    ]

    judgement = [
        "继续审 `future_high_gt_rot@K=1` 时，优先方向仍然是审阈值，而不是先改 horizon：因为当前这条 trigger 的未来窗口本身已经是 accept 分支，改 gate/geom 条件不会带来真正的新信息。",
        "但阈值并不是足够精细的最终定义轴：`tau=7.5/10` 会同时裁掉共享难例和 gated 真正修掉的高运动样本，所以它更像标签边界清理，不像高选择性的语义修正。",
        "如果后续还要继续收紧定义，比起引入未来窗口的 gate/geom 条件，更值得考虑的是把 `future_high_gt_rot` 和当前窗口的 recoverability 线索联动，例如 parallax / observability 侧的条件，而不是只继续抬纯旋转阈值。",
    ]

    payload = {
        "headline": headline,
        "judgement": judgement,
        "summary": {
            "future_high_gt_rot_rows": int(len(merged)),
            "shared_wrong_rows": int((merged["shared_wrong"] == 1).sum()),
            "gated_fix_rows": int((merged["gated_fixes_geo"] == 1).sum()),
            "geo_beats_rows": int((merged["geo_beats_gated"] == 1).sum()),
        },
        "future_gatepost_summary": gatepost.to_dict(orient="records"),
        "tau_retention": tau_df.to_dict(orient="records"),
        "profile_summary": profile_df.to_dict(orient="records"),
    }

    gatepost.to_csv(out_dir / "future_high_gt_rot_k1_future_gatepost_summary.csv", index=False)
    tau_df.to_csv(out_dir / "future_high_gt_rot_k1_tau_retention.csv", index=False)
    profile_df.to_csv(out_dir / "future_high_gt_rot_k1_profile_summary.csv", index=False)
    write_json(out_dir / "future_high_gt_rot_k1_semantics_audit.json", payload)
    (out_dir / "future_high_gt_rot_k1_semantics_audit.md").write_text(build_markdown(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
