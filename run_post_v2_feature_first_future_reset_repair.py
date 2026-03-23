#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
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
DEFAULT_OUT_DIR = PROJECT_ROOT / 'docs' / 'research' / 'init_risk_feature_first_internal_repair_20260323'

FULL_ANCHOR_COLS = [
    'front_p_static',
    'front_p_band',
    'front_coverage_ratio',
    'front_kept_dyn_ratio',
    'parallax_px_candidate',
    'tri_points_candidate',
    'pnp_success_rate',
    'reproj_med_px',
    'reproj_p90_px',
    'cheirality_ratio',
]

NEW_FEATURE_COLS = [
    'clean_geom_flag',
    'clean_geom_inv_parallax',
    'clean_geom_high_support',
    'clean_geom_cover_parallax',
]


def load_csv_rows(path: Path) -> list[dict]:
    with open(path, 'r', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def load_csv_header(path: Path) -> list[str]:
    with open(path, 'r', encoding='utf-8') as f:
        return list(csv.reader(f))[0]


def load_json(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


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


def compute_clean_features(row: dict, args) -> dict[str, float]:
    q_post = safe_float(row.get('Q_post_geom_only'), -1.0)
    pnp = safe_float(row.get('pnp_success_rate'), -1.0)
    che = safe_float(row.get('cheirality_ratio'), -1.0)
    reproj_med = safe_float(row.get('reproj_med_px'), 1e9)
    reproj_p90 = safe_float(row.get('reproj_p90_px'), 1e9)
    parallax = safe_float(row.get('parallax_px_candidate'), 0.0) or 0.0
    tri_points = safe_float(row.get('tri_points_candidate'), 0.0) or 0.0
    coverage = safe_float(row.get('front_coverage_ratio'), 0.0) or 0.0
    clean_flag = 1.0 if (
        q_post >= float(args.clean_q_post_geom_min)
        and pnp >= float(args.clean_pnp_success_min)
        and che >= float(args.clean_cheirality_min)
        and reproj_med <= float(args.clean_reproj_med_max)
        and reproj_p90 <= float(args.clean_reproj_p90_max)
    ) else 0.0
    return {
        'clean_geom_flag': clean_flag,
        'clean_geom_inv_parallax': clean_flag / (1.0 + max(0.0, parallax)),
        'clean_geom_high_support': clean_flag * math.log1p(max(0.0, tri_points)),
        'clean_geom_cover_parallax': clean_flag * max(0.0, coverage) / (1.0 + max(0.0, parallax)),
    }


def is_clean_future_reset(row: dict, args) -> bool:
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


def materialize_augmented_package(base_root: Path, out_dir: Path, args) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    full_rows = load_csv_rows(base_root / 'risk_dataset.csv')
    full_header = load_csv_header(base_root / 'risk_dataset.csv')
    full_rows_by_uid = {r['sample_uid']: r for r in full_rows}

    if any(col not in full_header for col in NEW_FEATURE_COLS):
        full_header = full_header + [c for c in NEW_FEATURE_COLS if c not in full_header]
    for row in full_rows:
        row.update({k: f'{v:.12g}' for k, v in compute_clean_features(row, args).items()})
    write_csv(out_dir / 'risk_dataset.csv', full_header, full_rows)

    for split in ['train', 'val', 'test']:
        src = base_root / f'risk_dataset_post_v2_min_default_{split}.csv'
        rows = load_csv_rows(src)
        header = load_csv_header(src)
        if any(col not in header for col in NEW_FEATURE_COLS):
            header = header + [c for c in NEW_FEATURE_COLS if c not in header]
        for row in rows:
            base = full_rows_by_uid[row['sample_uid']]
            row.update({k: f'{v:.12g}' for k, v in compute_clean_features(base, args).items()})
        write_csv(out_dir / src.name, header, rows)

    manifest = load_json(base_root / 'risk_dataset_post_v2_min_default_manifest.json')
    allowed = list(manifest.get('allowed_feature_columns', []))
    for col in NEW_FEATURE_COLS:
        if col not in allowed:
            allowed.append(col)
    manifest['allowed_feature_columns'] = allowed
    if isinstance(manifest.get('ablation_feature_columns'), list):
        ablation = list(manifest['ablation_feature_columns'])
        for col in NEW_FEATURE_COLS:
            if col not in ablation:
                ablation.append(col)
        manifest['ablation_feature_columns'] = ablation
    manifest['feature_first_future_reset_repair'] = {
        'feature_block': list(NEW_FEATURE_COLS),
        'source_definition': 'future_reset but current clean canonical definition',
        'thresholds': {
            'Q_post_geom_only_min': float(args.clean_q_post_geom_min),
            'pnp_success_rate_min': float(args.clean_pnp_success_min),
            'cheirality_ratio_min': float(args.clean_cheirality_min),
            'reproj_med_px_max': float(args.clean_reproj_med_max),
            'reproj_p90_px_max': float(args.clean_reproj_p90_max),
        },
        'description': 'Internal-only clean-geometry interaction features for future_reset current-clean repair.',
    }
    write_json(out_dir / 'risk_dataset_post_v2_min_default_manifest.json', manifest)
    return out_dir


def run_baseline(python_bin: str, package_dir: Path, out_dir: Path, feature_cols: list[str]) -> dict:
    cmd = [
        python_bin,
        str(PROJECT_ROOT / 'scripts' / 'sfm_init' / 'train_init_risk_baseline.py'),
        '--task', 'post',
        '--model', 'logistic',
        '--train_csv', str(package_dir / 'risk_dataset_post_v2_min_default_train.csv'),
        '--val_csv', str(package_dir / 'risk_dataset_post_v2_min_default_val.csv'),
        '--test_csv', str(package_dir / 'risk_dataset_post_v2_min_default_test.csv'),
        '--manifest', str(package_dir / 'risk_dataset_post_v2_min_default_manifest.json'),
        '--label_col', 'Y_bad_v2_min_default',
        '--out_dir', str(out_dir),
        '--override_feature_columns', ','.join(feature_cols),
    ]
    subprocess.run(cmd, check=True)
    return load_json(out_dir / 'metrics.json')


def percentile_against_negatives(score: float, negatives: list[dict]) -> float | None:
    if not negatives:
        return None
    return sum(1 for r in negatives if float(r['p_hat']) < float(score)) / len(negatives)


def frac_below(items: list[dict], threshold: float | None):
    if not items or threshold is None:
        return None
    return sum(1 for r in items if float(r['p_hat']) < threshold) / len(items)


def summarize_internal_groups(pred_rows: list[dict], base_rows_by_uid: dict[str, dict], args) -> tuple[list[dict], list[dict]]:
    enriched = []
    for pred in pred_rows:
        base = dict(base_rows_by_uid[pred['sample_uid']])
        base['p_hat'] = float(pred['p_hat'])
        base['y_true'] = int(pred['y_true'])
        base['split'] = pred['split']
        enriched.append(base)

    stats = []
    detailed = []
    for split in ['train', 'val']:
        split_rows = [r for r in enriched if r['split'] == split and r.get('sample_type') == 'step11']
        neg = [r for r in split_rows if int(r['y_true']) == 0]
        pos = [r for r in split_rows if int(r['y_true']) == 1]
        neg_scores = sorted(float(r['p_hat']) for r in neg)
        neg_median = neg_scores[len(neg_scores) // 2] if neg_scores else None
        groups = {
            'all_positive': pos,
            'future_reset': [r for r in pos if r.get('Y_bad_v2_min_default_trigger', '') == 'future_reset'],
            'clean_future_reset': [r for r in pos if is_clean_future_reset(r, args)],
        }
        for group_name, items in groups.items():
            stats.append({
                'split': split,
                'group': group_name,
                'num_rows': len(items),
                'mean_p_hat': mean([float(r['p_hat']) for r in items]),
                'mean_negative_p_hat': mean([float(r['p_hat']) for r in neg]),
                'frac_below_neg_median': frac_below(items, neg_median),
                'recall_at_0p5': None if not items else sum(1 for r in items if float(r['p_hat']) >= 0.5) / len(items),
            })
        for row in groups['clean_future_reset']:
            detailed.append({
                'split': split,
                'sample_uid': row['sample_uid'],
                'sequence': row.get('sequence', ''),
                'window_id': row.get('window_id', ''),
                'p_hat': float(row['p_hat']),
                'negative_rank_percentile': percentile_against_negatives(float(row['p_hat']), neg),
                'below_negative_median': '1' if neg_median is not None and float(row['p_hat']) < neg_median else '0',
                'Q_post_geom_only': row.get('Q_post_geom_only', ''),
                'parallax_px_candidate': row.get('parallax_px_candidate', ''),
                'tri_points_candidate': row.get('tri_points_candidate', ''),
                'front_coverage_ratio': row.get('front_coverage_ratio', ''),
            })
    detailed.sort(key=lambda r: (r['split'], float(r['p_hat'])))
    return stats, detailed


def coeff_map(model_dir: Path) -> dict[str, float]:
    rows = load_csv_rows(model_dir / 'coefficients.csv')
    return {r['feature']: float(r['weight']) for r in rows}


def build_markdown(payload: dict) -> str:
    lines = []
    lines.append('# feature-first internal repair for future_reset current-clean')
    lines.append('')
    lines.append('## Main Takeaway')
    lines.append('')
    for line in payload['takeaways']:
        lines.append(f'- {line}')
    lines.append('')
    lines.append('## Experiment Table')
    lines.append('')
    lines.append('| experiment | num_features | test AUROC | test AUPRC | test Brier | train clean FR mean p | train clean FR below neg median | val clean FR mean p | val clean FR below neg median |')
    lines.append('| --- | --- | --- | --- | --- | --- | --- | --- | --- |')
    for row in payload['summary_rows']:
        lines.append(
            f"| {row['experiment']} | {row['num_features']} | {fmt(row['test_auroc'])} | {fmt(row['test_auprc'])} | {fmt(row['test_brier'])} | {fmt(row['train_clean_future_reset_mean_p_hat'])} | {fmt(row['train_clean_future_reset_frac_below_neg_median'])} | {fmt(row['val_clean_future_reset_mean_p_hat'])} | {fmt(row['val_clean_future_reset_frac_below_neg_median'])} |"
        )
    lines.append('')
    lines.append('## Canonical Clean future_reset Rows')
    lines.append('')
    lines.append('| experiment | split | sample_uid | p_hat | neg-rank percentile | below neg median |')
    lines.append('| --- | --- | --- | --- | --- | --- |')
    for row in payload['canonical_rows']:
        lines.append(
            f"| {row['experiment']} | {row['split']} | {row['sample_uid']} | {fmt(row['p_hat'])} | {fmt(row['negative_rank_percentile'])} | {row['below_negative_median']} |"
        )
    lines.append('')
    lines.append('## New Feature Weights')
    lines.append('')
    lines.append('| experiment | clean_geom_flag | clean_geom_inv_parallax | clean_geom_high_support | clean_geom_cover_parallax |')
    lines.append('| --- | --- | --- | --- | --- |')
    for row in payload['weight_rows']:
        lines.append(
            f"| {row['experiment']} | {fmt(row['clean_geom_flag'])} | {fmt(row['clean_geom_inv_parallax'])} | {fmt(row['clean_geom_high_support'])} | {fmt(row['clean_geom_cover_parallax'])} |"
        )
    lines.append('')
    return '\n'.join(lines) + '\n'


def main():
    ap = argparse.ArgumentParser(description='Run feature-first internal repair around future_reset current-clean on VIODE.')
    ap.add_argument('--base_root', default=str(DEFAULT_BASE_ROOT))
    ap.add_argument('--out_dir', default=str(DEFAULT_OUT_DIR))
    ap.add_argument('--python_bin', default=str(Path.home() / 'projects' / 'venv' / 'bin' / 'python'))
    ap.add_argument('--clean_q_post_geom_min', type=float, default=0.99)
    ap.add_argument('--clean_pnp_success_min', type=float, default=0.99)
    ap.add_argument('--clean_cheirality_min', type=float, default=0.99)
    ap.add_argument('--clean_reproj_med_max', type=float, default=0.10)
    ap.add_argument('--clean_reproj_p90_max', type=float, default=0.30)
    args = ap.parse_args()

    base_root = Path(args.base_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    package_dir = materialize_augmented_package(base_root, out_dir / 'package_feature_first_repair', args)
    base_rows_by_uid = {r['sample_uid']: r for r in load_csv_rows(package_dir / 'risk_dataset.csv')}

    experiments = [
        ('full_anchor_baseline', list(FULL_ANCHOR_COLS), 'Frozen full-anchor reference.'),
        ('full_anchor_plus_clean_geom_flag', list(FULL_ANCHOR_COLS) + ['clean_geom_flag'], 'Add clean geometry flag only.'),
        ('full_anchor_plus_clean_geom_inv_parallax', list(FULL_ANCHOR_COLS) + ['clean_geom_flag', 'clean_geom_inv_parallax'], 'Add clean flag plus inverse-parallax interaction.'),
        ('full_anchor_plus_clean_geom_high_support', list(FULL_ANCHOR_COLS) + ['clean_geom_flag', 'clean_geom_high_support'], 'Add clean flag plus support interaction.'),
        ('full_anchor_plus_clean_geom_cover_parallax', list(FULL_ANCHOR_COLS) + ['clean_geom_flag', 'clean_geom_cover_parallax'], 'Add clean flag plus coverage-parallax interaction.'),
        ('full_anchor_plus_all_clean_geom', list(FULL_ANCHOR_COLS) + list(NEW_FEATURE_COLS), 'Add the full clean-geometry interaction block.'),
    ]

    summary_rows = []
    canonical_rows = []
    weight_rows = []
    baseline_test_auroc = None
    baseline_train_clean = None
    baseline_val_clean = None

    for exp_name, feature_cols, note in experiments:
        model_dir = out_dir / 'runs' / exp_name
        metrics = run_baseline(args.python_bin, package_dir, model_dir, feature_cols)
        pred_rows = load_csv_rows(model_dir / 'predictions.csv')
        stats_rows, detail_rows = summarize_internal_groups(pred_rows, base_rows_by_uid, args)
        stats_map = {(r['split'], r['group']): r for r in stats_rows}
        test_metrics = metrics['splits']['test']
        row = {
            'experiment': exp_name,
            'num_features': len(feature_cols),
            'features': ','.join(feature_cols),
            'test_auroc': test_metrics['auroc'],
            'test_auprc': test_metrics['auprc'],
            'test_brier': test_metrics['brier'],
            'test_ece': test_metrics['ece'],
            'train_future_reset_mean_p_hat': stats_map.get(('train', 'future_reset'), {}).get('mean_p_hat'),
            'train_future_reset_frac_below_neg_median': stats_map.get(('train', 'future_reset'), {}).get('frac_below_neg_median'),
            'val_future_reset_mean_p_hat': stats_map.get(('val', 'future_reset'), {}).get('mean_p_hat'),
            'val_future_reset_frac_below_neg_median': stats_map.get(('val', 'future_reset'), {}).get('frac_below_neg_median'),
            'train_clean_future_reset_mean_p_hat': stats_map.get(('train', 'clean_future_reset'), {}).get('mean_p_hat'),
            'train_clean_future_reset_frac_below_neg_median': stats_map.get(('train', 'clean_future_reset'), {}).get('frac_below_neg_median'),
            'val_clean_future_reset_mean_p_hat': stats_map.get(('val', 'clean_future_reset'), {}).get('mean_p_hat'),
            'val_clean_future_reset_frac_below_neg_median': stats_map.get(('val', 'clean_future_reset'), {}).get('frac_below_neg_median'),
            'note': note,
        }
        if baseline_test_auroc is None:
            baseline_test_auroc = row['test_auroc']
            baseline_train_clean = row['train_clean_future_reset_mean_p_hat']
            baseline_val_clean = row['val_clean_future_reset_mean_p_hat']
        row['delta_test_auroc_vs_baseline'] = None if row['test_auroc'] is None or baseline_test_auroc is None else float(row['test_auroc']) - float(baseline_test_auroc)
        row['delta_train_clean_future_reset_mean_p_hat_vs_baseline'] = None if row['train_clean_future_reset_mean_p_hat'] is None or baseline_train_clean is None else float(row['train_clean_future_reset_mean_p_hat']) - float(baseline_train_clean)
        row['delta_val_clean_future_reset_mean_p_hat_vs_baseline'] = None if row['val_clean_future_reset_mean_p_hat'] is None or baseline_val_clean is None else float(row['val_clean_future_reset_mean_p_hat']) - float(baseline_val_clean)
        summary_rows.append(row)

        for d in detail_rows:
            rr = dict(d)
            rr['experiment'] = exp_name
            canonical_rows.append(rr)

        weights = coeff_map(model_dir)
        weight_rows.append({
            'experiment': exp_name,
            'clean_geom_flag': weights.get('clean_geom_flag'),
            'clean_geom_inv_parallax': weights.get('clean_geom_inv_parallax'),
            'clean_geom_high_support': weights.get('clean_geom_high_support'),
            'clean_geom_cover_parallax': weights.get('clean_geom_cover_parallax'),
        })

    summary_rows.sort(key=lambda r: (-(r['val_clean_future_reset_mean_p_hat'] if r['val_clean_future_reset_mean_p_hat'] is not None else -1e9), -(r['train_clean_future_reset_mean_p_hat'] if r['train_clean_future_reset_mean_p_hat'] is not None else -1e9), -(r['test_auroc'] if r['test_auroc'] is not None else -1e9)))
    canonical_rows.sort(key=lambda r: (r['sample_uid'], -(r['p_hat'] if r['p_hat'] is not None else -1e9)))

    best = summary_rows[0]
    takeaways = [
        f"Best clean-future-reset uplift on val comes from `{best['experiment']}` with val clean mean p={fmt(best['val_clean_future_reset_mean_p_hat'])} and test AUROC={fmt(best['test_auroc'])}.",
        f"Frozen reference `full_anchor_baseline` starts at train/val clean mean p={fmt(baseline_train_clean)}/{fmt(baseline_val_clean)}.",
        'Use this block only as an internal repair experiment; it does not reopen the frozen external held-out protocol.',
    ]

    write_csv(out_dir / 'feature_first_repair_summary.csv', list(summary_rows[0].keys()), summary_rows)
    write_csv(out_dir / 'feature_first_repair_canonical_rows.csv', list(canonical_rows[0].keys()), canonical_rows)
    write_csv(out_dir / 'feature_first_repair_new_feature_weights.csv', list(weight_rows[0].keys()), weight_rows)
    payload = {
        'base_root': str(base_root),
        'package_dir': str(package_dir),
        'python_bin': str(args.python_bin),
        'new_feature_cols': list(NEW_FEATURE_COLS),
        'summary_rows': summary_rows,
        'canonical_rows': canonical_rows,
        'weight_rows': weight_rows,
        'takeaways': takeaways,
    }
    write_json(out_dir / 'feature_first_repair_summary.json', payload)
    write_text(out_dir / 'feature_first_repair_summary.md', build_markdown(payload))
    print(f'[feature_first_repair] saved -> {out_dir}')


if __name__ == '__main__':
    main()
