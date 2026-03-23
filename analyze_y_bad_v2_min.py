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


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = set()
    for row in rows:
        keys |= set(row.keys())
    fieldnames = sorted(keys)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def write_selected_csv(path: Path, rows: list[dict], columns: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in columns})


def build_sequence_summary(v2_rows: list[dict]) -> list[dict]:
    out = []
    by_seq = {}
    for row in v2_rows:
        seq = str(row.get("sequence", ""))
        by_seq.setdefault(seq, []).append(row)
    for seq, rows in sorted(by_seq.items()):
        labeled = [r for r in rows if r.get("Y_bad_v2_min") in ("0", "1", 0, 1)]
        bad = sum(int(r["Y_bad_v2_min"]) for r in labeled)
        trigger_counts = {}
        for r in labeled:
            t = str(r.get("Y_bad_v2_min_trigger", ""))
            trigger_counts[t] = trigger_counts.get(t, 0) + 1
        out.append(
            {
                "sequence": seq,
                "num_rows": len(rows),
                "accept_rows": sum(1 for r in rows if str(r.get("gate_post", "")) == "accept"),
                "labeled_rows": len(labeled),
                "bad_rows": int(bad),
                "bad_ratio_labeled": float(bad / max(1, len(labeled))) if labeled else None,
                "trigger_composition": json.dumps(trigger_counts, ensure_ascii=False),
            }
        )
    return out


def build_crosstab(joined_rows: list[dict]) -> dict:
    table = {}
    for row in joined_rows:
        v2 = row.get("Y_bad_v2_min")
        if v2 not in ("0", "1", 0, 1):
            continue
        v1 = int(float(row.get("Y_bad_v1", 0)))
        v2 = int(float(v2))
        key = f"v1_{v1}__v2_{v2}"
        table[key] = table.get(key, 0) + 1
    return table


def build_joined_rows(risk_rows: list[dict], v2_rows: list[dict]) -> list[dict]:
    risk_step11 = [r for r in risk_rows if r.get("sample_type") == "step11"]
    risk_by_attempt = {str(r["attempt_uid"]): r for r in risk_step11}
    joined = []
    for v2 in v2_rows:
        attempt_uid = str(v2["attempt_uid"])
        risk = risk_by_attempt.get(attempt_uid)
        if risk is None:
            continue
        row = dict(risk)
        row["Y_bad_v2_min"] = v2.get("Y_bad_v2_min", "")
        row["Y_bad_v2_min_source"] = v2.get("Y_bad_v2_min_source", "")
        row["prov_Y_bad_v2_min"] = v2.get("prov_Y_bad_v2_min", "")
        row["Y_bad_v2_min_trigger"] = v2.get("Y_bad_v2_min_trigger", "")
        row["Y_bad_v2_min_trigger_window_id"] = v2.get("Y_bad_v2_min_trigger_window_id", "")
        row["Y_bad_v2_min_label_version"] = v2.get("Y_bad_v2_min_label_version", "Y_bad_v2_min")
        row["Y_bad_v2_min_label_scope"] = v2.get("Y_bad_v2_min_label_scope", "accepted_horizon_instability_proxy")
        row["Y_bad_v2_min_horizon_windows"] = v2.get("Y_bad_v2_min_horizon_windows", "")
        joined.append(row)
    return joined


def export_minimal_post_task(out_dir: Path, joined_rows: list[dict], post_manifest: dict):
    labeled = [r for r in joined_rows if r.get("Y_bad_v2_min") in ("0", "1", 0, 1)]
    task_metadata_columns = list(post_manifest["task_metadata_columns"])
    allowed_feature_columns = list(post_manifest["allowed_feature_columns"])
    label_columns = ["Y_bad_v2_min"]
    columns = task_metadata_columns + allowed_feature_columns + label_columns

    splits = sorted({str(r.get("dataset_row_split", "unsplit")) for r in labeled})
    split_files = {}
    for split in splits:
        rows = [r for r in labeled if str(r.get("dataset_row_split", "unsplit")) == split]
        out_csv = out_dir / f"risk_dataset_post_v2_min_{split}.csv"
        write_selected_csv(out_csv, rows, columns)
        split_files[split] = str(out_csv)

    task_manifest = {
        "task_name": "post_v2_min_default",
        "allowed_feature_columns": allowed_feature_columns,
        "label_columns": label_columns,
        "task_metadata_columns": task_metadata_columns,
        "split_policy": post_manifest.get("split_policy", "strict_by_sequence"),
        "label_version": "Y_bad_v2_min",
        "label_scope": "accepted_horizon_instability_proxy",
        "task_scope": "post_accept_horizon_instability_feasibility",
        "allowed_claims": [
            "Current protocol supports a learnable post-accept horizon instability signal on labeled accept windows.",
            "This label is closer to short-horizon acceptance stability than Y_bad_v1, but is still not a full system-level failure label.",
        ],
        "forbidden_claims": [
            "Do not claim system-level failure prediction or full downstream gate benefit from Y_bad_v2_min.",
            "Do not claim cross-scene or cross-dynamic-level generalization from the current 4-sequence split.",
        ],
        "proxy_bias_notes": [
            "Y_bad_v2_min only covers accepted windows with sufficient future horizon.",
            "Y_bad_v2_min is a short-horizon instability proxy and not yet a reset/reinit/tracking-failure ground truth.",
        ],
        "experimental_limit_flags": [
            "small_sample",
            "single_protocol_family",
            "dynamic_level_split_coupling",
            "accepted_window_subset_only",
            "proxy_label_only",
        ],
        "source_schema_versions": post_manifest.get("source_schema_versions", {}),
        "legality_board_path": post_manifest.get("legality_board_path", ""),
        "legality_board_hash": post_manifest.get("legality_board_hash", ""),
        "protocol_columns_excluded": post_manifest.get("protocol_columns_excluded", []),
        "posthoc_columns_excluded": post_manifest.get("posthoc_columns_excluded", []),
        "data_quality_columns_excluded": post_manifest.get("data_quality_columns_excluded", []),
        "forbidden_columns": post_manifest.get("forbidden_columns", []),
        "output_split_csvs": split_files,
    }
    write_json(out_dir / "risk_dataset_post_v2_min_manifest.json", task_manifest)
    return labeled, task_manifest


def main():
    ap = argparse.ArgumentParser(description="Analyze Y_bad_v2_min default label and prepare a minimal training package.")
    ap.add_argument("--risk_dataset_csv", required=True)
    ap.add_argument("--post_manifest", required=True)
    ap.add_argument("--v2_labels_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    risk_rows = load_csv_rows(Path(args.risk_dataset_csv).expanduser().resolve())
    post_manifest = load_json(Path(args.post_manifest).expanduser().resolve())
    v2_rows = load_csv_rows(Path(args.v2_labels_csv).expanduser().resolve())

    sequence_summary = build_sequence_summary(v2_rows)
    joined_rows = build_joined_rows(risk_rows, v2_rows)
    crosstab = build_crosstab(joined_rows)

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "y_bad_v2_min_sequence_summary.csv", sequence_summary)
    write_json(out_dir / "y_bad_v1_vs_v2_min_crosstab.json", crosstab)
    write_csv(out_dir / "risk_dataset_step11_with_v2_min.csv", joined_rows)
    write_csv(out_dir / "risk_dataset.csv", joined_rows)

    labeled_rows, task_manifest = export_minimal_post_task(out_dir, joined_rows, post_manifest)
    audit = {
        "num_step11_rows": len(joined_rows),
        "num_v2_labeled_rows": len(labeled_rows),
        "y_bad_v2_min_bad_ratio": (
            float(sum(int(r["Y_bad_v2_min"]) for r in labeled_rows) / max(1, len(labeled_rows)))
            if labeled_rows else None
        ),
        "sequence_summary_csv": str(out_dir / "y_bad_v2_min_sequence_summary.csv"),
        "crosstab_json": str(out_dir / "y_bad_v1_vs_v2_min_crosstab.json"),
        "joined_csv": str(out_dir / "risk_dataset_step11_with_v2_min.csv"),
        "post_v2_manifest_json": str(out_dir / "risk_dataset_post_v2_min_manifest.json"),
    }
    write_json(out_dir / "y_bad_v2_min_analysis_audit.json", audit)

    print(f"[Y_bad_v2_min_analysis] step11_rows={len(joined_rows)}")
    print(f"[Y_bad_v2_min_analysis] labeled_rows={len(labeled_rows)} bad_ratio={audit['y_bad_v2_min_bad_ratio']}")
    print(f"[Y_bad_v2_min_analysis] saved -> {out_dir}")


if __name__ == "__main__":
    main()
