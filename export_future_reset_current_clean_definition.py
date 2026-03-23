#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = None
for p in [THIS_FILE.parent, *THIS_FILE.parent.parents]:
    if (p / 'configs').is_dir() and ((p / 'pipelines').is_dir() or (p / ' pipelines').is_dir()):
        PROJECT_ROOT = p
        break
if PROJECT_ROOT is None:
    raise RuntimeError(f'Cannot locate project root from {THIS_FILE}')

DEFAULT_BASE_ROOT = Path('/mnt/g/Result/VIODE/post_v2_rebuild_9seq/risk_dataset_9seq_seed20260432')
DEFAULT_OUT_DIR = PROJECT_ROOT / 'docs' / 'research' / 'init_risk_future_reset_current_clean_definition_20260323'
EXPORT_FIELDS = [
    'split',
    'sample_uid',
    'sequence',
    'variant_tag',
    'window_id',
    'sample_type',
    'Y_bad_v2_min_default',
    'Y_bad_v2_min_default_trigger',
    'Q_post_geom_only',
    'pnp_success_rate',
    'cheirality_ratio',
    'reproj_med_px',
    'reproj_p90_px',
    'parallax_px_candidate',
    'tri_points_candidate',
    'front_coverage_ratio',
    'front_p_static',
    'front_p_band',
    'front_kept_dyn_ratio',
]


def load_csv_rows(path: Path) -> list[dict]:
    with open(path, 'r', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)


def safe_float(v, default=None):
    try:
        x = float(v)
    except Exception:
        return default
    if math.isnan(x) or math.isinf(x):
        return default
    return x


def safe_int(v, default=None):
    try:
        return int(float(v))
    except Exception:
        return default


def mean(values: list[float]):
    return sum(values) / len(values) if values else None


def fmt(v, nd=4):
    if v is None:
        return '-'
    try:
        return f'{float(v):.{nd}f}'
    except Exception:
        return str(v)


def build_split_uid_map(base_root: Path) -> dict[str, set[str]]:
    split_paths = {
        'train': base_root / 'risk_dataset_post_v2_min_default_train.csv',
        'val': base_root / 'risk_dataset_post_v2_min_default_val.csv',
    }
    out = {}
    for split, path in split_paths.items():
        rows = load_csv_rows(path)
        out[split] = {r['sample_uid'] for r in rows}
    return out


def is_future_reset_current_clean(row: dict, args) -> bool:
    return (
        row.get('sample_type', '') == 'step11'
        and safe_int(row.get('Y_bad_v2_min_default'), 0) == 1
        and row.get('Y_bad_v2_min_default_trigger', '') == 'future_reset'
        and safe_float(row.get('Q_post_geom_only'), -1.0) >= float(args.clean_q_post_geom_min)
        and safe_float(row.get('pnp_success_rate'), -1.0) >= float(args.clean_pnp_success_min)
        and safe_float(row.get('cheirality_ratio'), -1.0) >= float(args.clean_cheirality_min)
        and safe_float(row.get('reproj_med_px'), 1e9) <= float(args.clean_reproj_med_max)
        and safe_float(row.get('reproj_p90_px'), 1e9) <= float(args.clean_reproj_p90_max)
    )


def build_export_rows(split: str, split_uids: set[str], full_rows_by_uid: dict[str, dict], args) -> list[dict]:
    rows = []
    for uid in sorted(split_uids):
        base = full_rows_by_uid.get(uid)
        if not base:
            continue
        if not is_future_reset_current_clean(base, args):
            continue
        row = {'split': split}
        for key in EXPORT_FIELDS[1:]:
            row[key] = base.get(key, '')
        rows.append(row)
    return rows


def summarize_split(split: str, rows: list[dict]) -> dict:
    return {
        'split': split,
        'num_rows': len(rows),
        'mean_Q_post_geom_only': mean([safe_float(r.get('Q_post_geom_only')) for r in rows if safe_float(r.get('Q_post_geom_only')) is not None]),
        'mean_pnp_success_rate': mean([safe_float(r.get('pnp_success_rate')) for r in rows if safe_float(r.get('pnp_success_rate')) is not None]),
        'mean_cheirality_ratio': mean([safe_float(r.get('cheirality_ratio')) for r in rows if safe_float(r.get('cheirality_ratio')) is not None]),
        'mean_reproj_med_px': mean([safe_float(r.get('reproj_med_px')) for r in rows if safe_float(r.get('reproj_med_px')) is not None]),
        'mean_reproj_p90_px': mean([safe_float(r.get('reproj_p90_px')) for r in rows if safe_float(r.get('reproj_p90_px')) is not None]),
        'mean_parallax_px_candidate': mean([safe_float(r.get('parallax_px_candidate')) for r in rows if safe_float(r.get('parallax_px_candidate')) is not None]),
        'mean_tri_points_candidate': mean([safe_float(r.get('tri_points_candidate')) for r in rows if safe_float(r.get('tri_points_candidate')) is not None]),
        'mean_front_coverage_ratio': mean([safe_float(r.get('front_coverage_ratio')) for r in rows if safe_float(r.get('front_coverage_ratio')) is not None]),
    }


def build_markdown(base_root: Path, out_dir: Path, split_stats: list[dict], train_rows: list[dict], val_rows: list[dict], args) -> str:
    all_rows = train_rows + val_rows
    lines = []
    lines.append('# future_reset but current clean: formal internal definition')
    lines.append('')
    lines.append('## Role')
    lines.append('')
    lines.append('这份文档把 `future_reset but current clean` 正式固化为当前初始化风险主线中的**内部研究对象**。')
    lines.append('')
    lines.append('它的作用是：')
    lines.append('')
    lines.append('- 作为后续 feature-first internal repair 的统一目标子集')
    lines.append('- 作为 auxiliary-label 路线是否需要启动的判断对象')
    lines.append('- 作为离线 gate policy 是否覆盖主盲点的检查对象')
    lines.append('')
    lines.append('它**不是** external held-out 协议的一部分，也**不是**主标签协议的替代。')
    lines.append('')
    lines.append('## Canonical Definition')
    lines.append('')
    lines.append('只有同时满足以下条件的样本，才属于当前正式定义下的 `future_reset but current clean`：')
    lines.append('')
    lines.append('1. `sample_type = step11`')
    lines.append('2. `Y_bad_v2_min_default = 1`')
    lines.append('3. `Y_bad_v2_min_default_trigger = future_reset`')
    lines.append(f'4. `Q_post_geom_only >= {args.clean_q_post_geom_min}`')
    lines.append(f'5. `pnp_success_rate >= {args.clean_pnp_success_min}`')
    lines.append(f'6. `cheirality_ratio >= {args.clean_cheirality_min}`')
    lines.append(f'7. `reproj_med_px <= {args.clean_reproj_med_max}`')
    lines.append(f'8. `reproj_p90_px <= {args.clean_reproj_p90_max}`')
    lines.append('')
    lines.append('等价布尔式：')
    lines.append('')
    lines.append('```text')
    lines.append('future_reset_current_clean =')
    lines.append('  1[sample_type=step11]')
    lines.append('  * 1[Y_bad_v2_min_default=1]')
    lines.append('  * 1[Y_bad_v2_min_default_trigger=future_reset]')
    lines.append(f'  * 1[Q_post_geom_only>={args.clean_q_post_geom_min}]')
    lines.append(f'  * 1[pnp_success_rate>={args.clean_pnp_success_min}]')
    lines.append(f'  * 1[cheirality_ratio>={args.clean_cheirality_min}]')
    lines.append(f'  * 1[reproj_med_px<={args.clean_reproj_med_max}]')
    lines.append(f'  * 1[reproj_p90_px<={args.clean_reproj_p90_max}]')
    lines.append('```')
    lines.append('')
    lines.append('## Reproducibility Contract')
    lines.append('')
    lines.append('固定数据源：')
    lines.append('')
    lines.append(f'- `base_root = {base_root}`')
    lines.append(f'- full table: `{base_root / "risk_dataset.csv"}`')
    lines.append(f'- train split: `{base_root / "risk_dataset_post_v2_min_default_train.csv"}`')
    lines.append(f'- val split: `{base_root / "risk_dataset_post_v2_min_default_val.csv"}`')
    lines.append('')
    lines.append('固定导出命令：')
    lines.append('')
    lines.append('```bash')
    lines.append('python3 scripts/sfm_init/export_future_reset_current_clean_definition.py')
    lines.append('```')
    lines.append('')
    lines.append('固定输出目录：')
    lines.append('')
    lines.append(f'- `{out_dir}`')
    lines.append('')
    lines.append('## Current Reference Counts')
    lines.append('')
    lines.append('| split | num_rows | mean_Q_post_geom_only | mean_pnp_success_rate | mean_cheirality_ratio | mean_reproj_med_px | mean_reproj_p90_px | mean_parallax | mean_tri_points |')
    lines.append('| --- | --- | --- | --- | --- | --- | --- | --- | --- |')
    for row in split_stats:
        lines.append(
            f"| {row['split']} | {row['num_rows']} | {fmt(row['mean_Q_post_geom_only'])} | {fmt(row['mean_pnp_success_rate'])} | {fmt(row['mean_cheirality_ratio'])} | {fmt(row['mean_reproj_med_px'])} | {fmt(row['mean_reproj_p90_px'])} | {fmt(row['mean_parallax_px_candidate'])} | {fmt(row['mean_tri_points_candidate'])} |"
        )
    lines.append('')
    lines.append(f'- total rows in canonical subset: `{len(all_rows)}`')
    lines.append('')
    lines.append('## Canonical Sample UIDs')
    lines.append('')
    if not all_rows:
        lines.append('- No rows matched the current definition on the reference internal train/val split.')
    else:
        for row in all_rows:
            lines.append(f"- `{row['split']}`: `{row['sample_uid']}`")
    lines.append('')
    lines.append('## Usage Boundary')
    lines.append('')
    lines.append('- This subset is for internal analysis, feature design, and gate coverage checks only.')
    lines.append('- It must not be used to redefine the external held-out protocol.')
    lines.append('- It must not be used to reopen the frozen baseline by changing the main label protocol directly.')
    lines.append('- If future thresholds are changed, that must create a new dated definition directory rather than silently mutating this one.')
    lines.append('')
    return '\n'.join(lines) + '\n'


def main():
    ap = argparse.ArgumentParser(description='Export the formal internal definition and canonical subset for future_reset but current clean.')
    ap.add_argument('--base_root', default=str(DEFAULT_BASE_ROOT))
    ap.add_argument('--out_dir', default=str(DEFAULT_OUT_DIR))
    ap.add_argument('--clean_q_post_geom_min', type=float, default=0.99)
    ap.add_argument('--clean_pnp_success_min', type=float, default=0.99)
    ap.add_argument('--clean_cheirality_min', type=float, default=0.99)
    ap.add_argument('--clean_reproj_med_max', type=float, default=0.10)
    ap.add_argument('--clean_reproj_p90_max', type=float, default=0.30)
    args = ap.parse_args()

    base_root = Path(args.base_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    full_rows_by_uid = {r['sample_uid']: r for r in load_csv_rows(base_root / 'risk_dataset.csv')}
    split_uid_map = build_split_uid_map(base_root)

    train_rows = build_export_rows('train', split_uid_map['train'], full_rows_by_uid, args)
    val_rows = build_export_rows('val', split_uid_map['val'], full_rows_by_uid, args)
    all_rows = train_rows + val_rows
    split_stats = [
        summarize_split('train', train_rows),
        summarize_split('val', val_rows),
        summarize_split('trainval', all_rows),
    ]

    write_csv(out_dir / 'future_reset_current_clean_train.csv', EXPORT_FIELDS, train_rows)
    write_csv(out_dir / 'future_reset_current_clean_val.csv', EXPORT_FIELDS, val_rows)
    write_csv(out_dir / 'future_reset_current_clean_trainval.csv', EXPORT_FIELDS, all_rows)
    write_csv(out_dir / 'future_reset_current_clean_split_stats.csv', list(split_stats[0].keys()), split_stats)
    write_text(out_dir / 'future_reset_current_clean_sample_uids.txt', '\n'.join(r['sample_uid'] for r in all_rows) + ('\n' if all_rows else ''))

    manifest = {
        'research_object': 'future_reset_but_current_clean',
        'role': 'internal_only',
        'base_root': str(base_root),
        'definition': {
            'sample_type': 'step11',
            'label_col': 'Y_bad_v2_min_default',
            'label_value': 1,
            'trigger_col': 'Y_bad_v2_min_default_trigger',
            'trigger_value': 'future_reset',
            'thresholds': {
                'Q_post_geom_only_min': float(args.clean_q_post_geom_min),
                'pnp_success_rate_min': float(args.clean_pnp_success_min),
                'cheirality_ratio_min': float(args.clean_cheirality_min),
                'reproj_med_px_max': float(args.clean_reproj_med_max),
                'reproj_p90_px_max': float(args.clean_reproj_p90_max),
            },
            'logic_expression': 'step11 AND Y_bad_v2_min_default=1 AND trigger=future_reset AND Q_post_geom_only>=thr AND pnp_success_rate>=thr AND cheirality_ratio>=thr AND reproj_med_px<=thr AND reproj_p90_px<=thr',
        },
        'current_counts': {row['split']: row['num_rows'] for row in split_stats},
        'outputs': {
            'train_csv': str(out_dir / 'future_reset_current_clean_train.csv'),
            'val_csv': str(out_dir / 'future_reset_current_clean_val.csv'),
            'trainval_csv': str(out_dir / 'future_reset_current_clean_trainval.csv'),
            'split_stats_csv': str(out_dir / 'future_reset_current_clean_split_stats.csv'),
            'sample_uids_txt': str(out_dir / 'future_reset_current_clean_sample_uids.txt'),
            'definition_md': str(out_dir / 'future_reset_current_clean_definition.md'),
        },
    }
    write_json(out_dir / 'future_reset_current_clean_definition_manifest.json', manifest)
    write_text(out_dir / 'future_reset_current_clean_definition.md', build_markdown(base_root, out_dir, split_stats, train_rows, val_rows, args))
    print(f'[future_reset_current_clean_definition] saved -> {out_dir}')


if __name__ == '__main__':
    main()
