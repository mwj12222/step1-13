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

DEFAULT_REPAIR_ROOT = PROJECT_ROOT / 'docs' / 'research' / 'init_risk_feature_first_internal_repair_20260323'
DEFAULT_OUT_DIR = PROJECT_ROOT / 'docs' / 'research' / 'init_risk_clean_geom_flag_offline_gate_definition_20260323'
DEFAULT_PACKAGE = DEFAULT_REPAIR_ROOT / 'package_feature_first_repair' / 'risk_dataset.csv'
DEFAULT_PREDICTIONS = DEFAULT_REPAIR_ROOT / 'runs' / 'full_anchor_plus_clean_geom_flag' / 'predictions.csv'
DEFAULT_DELAY_THRESHOLD = 0.50
DEFAULT_RESET_THRESHOLD = 0.90


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


def is_clean_future_reset(row: dict) -> bool:
    return (
        row.get('sample_type', '') == 'step11'
        and safe_int(row.get('Y_bad_v2_min_default'), 0) == 1
        and row.get('Y_bad_v2_min_default_trigger', '') == 'future_reset'
        and safe_float(row.get('clean_geom_flag'), 0.0) >= 0.5
    )


def gate_action(p_hat: float, delay_threshold: float, reset_threshold: float) -> str:
    if p_hat >= float(reset_threshold):
        return 'reset'
    if p_hat >= float(delay_threshold):
        return 'delay'
    return 'accept'


def gate_reason(base_row: dict, p_hat: float, delay_threshold: float, reset_threshold: float) -> str:
    clean_flag = safe_float(base_row.get('clean_geom_flag'), 0.0) >= 0.5
    action = gate_action(p_hat, delay_threshold, reset_threshold)
    if clean_flag and action == 'reset':
        return 'clean_future_instability_high_risk'
    if clean_flag and action == 'delay':
        return 'clean_future_instability_watchlist'
    if action == 'reset':
        return 'general_high_post_risk'
    if action == 'delay':
        return 'general_marginal_post_risk'
    return 'low_post_risk'


def backend_hint(action: str) -> str:
    mapping = {
        'accept': 'accept',
        'delay': 'delay',
        'reset': 'fallback_or_reject',
    }
    return mapping[action]


def percentile_against_negatives(score: float, negatives: list[dict]) -> float | None:
    if not negatives:
        return None
    return sum(1 for r in negatives if float(r['p_hat']) < float(score)) / len(negatives)


def build_rows(base_rows_by_uid: dict[str, dict], prediction_rows: list[dict], delay_threshold: float, reset_threshold: float) -> list[dict]:
    out = []
    for pred in prediction_rows:
        base = base_rows_by_uid[pred['sample_uid']]
        p_hat = float(pred['p_hat'])
        action = gate_action(p_hat, delay_threshold, reset_threshold)
        reason = gate_reason(base, p_hat, delay_threshold, reset_threshold)
        row = {
            'split': pred['split'],
            'sample_uid': pred['sample_uid'],
            'sequence': pred.get('sequence', base.get('sequence', '')),
            'window_id': pred.get('window_id', base.get('window_id', '')),
            'y_true': safe_int(pred.get('y_true'), 0),
            'p_hat': p_hat,
            'offline_gate_action': action,
            'offline_gate_reason': reason,
            'backend_action_hint': backend_hint(action),
            'clean_geom_flag': safe_float(base.get('clean_geom_flag'), 0.0),
            'Q_post': safe_float(base.get('Q_post')),
            'Q_post_geom_only': safe_float(base.get('Q_post_geom_only')),
            'gate_post_native': base.get('gate_post', ''),
            'gate_post_reason_native': base.get('gate_post_reason', ''),
            'Y_bad_v2_min_default_trigger': base.get('Y_bad_v2_min_default_trigger', ''),
        }
        out.append(row)
    return out


def summarize_actions(rows: list[dict]) -> list[dict]:
    out = []
    for split in ['train', 'val', 'test']:
        srows = [r for r in rows if r['split'] == split]
        if not srows:
            continue
        for action in ['accept', 'delay', 'reset']:
            arows = [r for r in srows if r['offline_gate_action'] == action]
            out.append({
                'split': split,
                'action': action,
                'num_rows': len(arows),
                'ratio': len(arows) / len(srows),
                'positive_ratio': mean([float(r['y_true']) for r in arows]),
                'mean_p_hat': mean([float(r['p_hat']) for r in arows]),
            })
    return out


def summarize_targets(rows: list[dict]) -> list[dict]:
    out = []
    for split in ['train', 'val', 'test']:
        srows = [r for r in rows if r['split'] == split]
        if not srows:
            continue
        groups = {
            'future_reset': [r for r in srows if r['Y_bad_v2_min_default_trigger'] == 'future_reset'],
            'clean_future_reset': [r for r in srows if r['Y_bad_v2_min_default_trigger'] == 'future_reset' and safe_float(r['clean_geom_flag'], 0.0) >= 0.5],
        }
        for group_name, grows in groups.items():
            out.append({
                'split': split,
                'group': group_name,
                'num_rows': len(grows),
                'accept_ratio': None if not grows else sum(1 for r in grows if r['offline_gate_action'] == 'accept') / len(grows),
                'delay_ratio': None if not grows else sum(1 for r in grows if r['offline_gate_action'] == 'delay') / len(grows),
                'reset_ratio': None if not grows else sum(1 for r in grows if r['offline_gate_action'] == 'reset') / len(grows),
                'delay_or_reset_ratio': None if not grows else sum(1 for r in grows if r['offline_gate_action'] in ('delay', 'reset')) / len(grows),
                'mean_p_hat': mean([float(r['p_hat']) for r in grows]),
            })
    return out


def build_canonical_rows(rows: list[dict]) -> list[dict]:
    out = []
    for split in ['train', 'val']:
        srows = [r for r in rows if r['split'] == split]
        neg = [r for r in srows if int(r['y_true']) == 0]
        for row in srows:
            if row['Y_bad_v2_min_default_trigger'] != 'future_reset' or safe_float(row['clean_geom_flag'], 0.0) < 0.5:
                continue
            out.append({
                'split': split,
                'sample_uid': row['sample_uid'],
                'p_hat': row['p_hat'],
                'offline_gate_action': row['offline_gate_action'],
                'offline_gate_reason': row['offline_gate_reason'],
                'native_gate_post': row['gate_post_native'],
                'native_gate_post_reason': row['gate_post_reason_native'],
                'negative_rank_percentile': percentile_against_negatives(float(row['p_hat']), neg),
            })
    return out


def build_markdown(payload: dict) -> str:
    lines = []
    lines.append('# clean_geom_flag offline gate definition')
    lines.append('')
    lines.append('## Role')
    lines.append('')
    lines.append('这份文档把 `clean_geom_flag` 作为当前 internal feature-first 修正后的**离线 gate 候选信号**接入门控定义。')
    lines.append('')
    lines.append('这不是 backend 最终策略，也不是 external held-out 协议。')
    lines.append('它的作用是：')
    lines.append('')
    lines.append('- 给 `full_anchor_plus_clean_geom_flag` 一个明确的动作层解释')
    lines.append('- 把 learned risk score 变成 `accept / delay / reset` 离线动作')
    lines.append('- 检查 repaired canonical hard case 是否会被 gate 保护到')
    lines.append('')
    lines.append('## Candidate Signal')
    lines.append('')
    lines.append('- risk scorer: `full_anchor_plus_clean_geom_flag`')
    lines.append('- added feature: `clean_geom_flag`')
    lines.append('- semantics: current geometry looks clean, but the repaired internal model is allowed to score it as post-accept short-horizon risk')
    lines.append('')
    lines.append('## Action Definition')
    lines.append('')
    lines.append(f'- `accept`: `p_hat < {payload["delay_threshold"]:.2f}`')
    lines.append(f'- `delay`: `{payload["delay_threshold"]:.2f} <= p_hat < {payload["reset_threshold"]:.2f}`')
    lines.append(f'- `reset`: `p_hat >= {payload["reset_threshold"]:.2f}`')
    lines.append('')
    lines.append('backend 映射提示：')
    lines.append('')
    lines.append('- `accept -> accept`')
    lines.append('- `delay -> delay / ask for more evidence`')
    lines.append('- `reset -> fallback_or_reject`')
    lines.append('')
    lines.append('## Reason Codes')
    lines.append('')
    lines.append('- `clean_future_instability_high_risk`: `clean_geom_flag=1` 且进入 `reset` band')
    lines.append('- `clean_future_instability_watchlist`: `clean_geom_flag=1` 且进入 `delay` band')
    lines.append('- `general_high_post_risk`: 非 clean-geom 特例，但进入 `reset` band')
    lines.append('- `general_marginal_post_risk`: 非 clean-geom 特例，但进入 `delay` band')
    lines.append('- `low_post_risk`: 落在 `accept` band')
    lines.append('')
    lines.append('## Threshold Rationale')
    lines.append('')
    lines.append(f'- `delay_threshold={payload["delay_threshold"]:.2f}` 保持与当前 internal 分析里常用的 `0.5` 风险决策点一致。')
    lines.append(f'- `reset_threshold={payload["reset_threshold"]:.2f}` 用于定义高置信风险带，使 repaired canonical clean-future-reset 样本进入最强保护动作，而不把大多数一般样本直接推入 reset。')
    lines.append('')
    lines.append('## Action Summary')
    lines.append('')
    lines.append('| split | action | n | ratio | positive ratio | mean p_hat |')
    lines.append('| --- | --- | --- | --- | --- | --- |')
    for row in payload['action_summary_rows']:
        lines.append(f"| {row['split']} | {row['action']} | {row['num_rows']} | {fmt(row['ratio'])} | {fmt(row['positive_ratio'])} | {fmt(row['mean_p_hat'])} |")
    lines.append('')
    lines.append('## Target-Subset Coverage')
    lines.append('')
    lines.append('| split | group | n | accept | delay | reset | delay_or_reset | mean p_hat |')
    lines.append('| --- | --- | --- | --- | --- | --- | --- | --- |')
    for row in payload['target_summary_rows']:
        lines.append(f"| {row['split']} | {row['group']} | {row['num_rows']} | {fmt(row['accept_ratio'])} | {fmt(row['delay_ratio'])} | {fmt(row['reset_ratio'])} | {fmt(row['delay_or_reset_ratio'])} | {fmt(row['mean_p_hat'])} |")
    lines.append('')
    lines.append('## Canonical Clean future_reset Rows')
    lines.append('')
    lines.append('| split | sample_uid | p_hat | action | reason | native gate_post | native reason | neg-rank percentile |')
    lines.append('| --- | --- | --- | --- | --- | --- | --- | --- |')
    for row in payload['canonical_rows']:
        lines.append(f"| {row['split']} | {row['sample_uid']} | {fmt(row['p_hat'])} | {row['offline_gate_action']} | {row['offline_gate_reason']} | {row['native_gate_post']} | {row['native_gate_post_reason']} | {fmt(row['negative_rank_percentile'])} |")
    lines.append('')
    lines.append('## Usage Boundary')
    lines.append('')
    lines.append('- This gate definition is internal-only and exists to bridge repaired reliability scoring to an offline action policy.')
    lines.append('- It does not claim backend benefit yet; that requires a later offline gate simulation and then backend integration.')
    lines.append('- It does not reopen the frozen external held-out protocol.')
    lines.append('')
    return '\n'.join(lines) + '\n'


def main():
    ap = argparse.ArgumentParser(description='Define an internal offline gate candidate around full_anchor_plus_clean_geom_flag.')
    ap.add_argument('--package_csv', default=str(DEFAULT_PACKAGE))
    ap.add_argument('--predictions_csv', default=str(DEFAULT_PREDICTIONS))
    ap.add_argument('--out_dir', default=str(DEFAULT_OUT_DIR))
    ap.add_argument('--delay_threshold', type=float, default=DEFAULT_DELAY_THRESHOLD)
    ap.add_argument('--reset_threshold', type=float, default=DEFAULT_RESET_THRESHOLD)
    args = ap.parse_args()

    if float(args.reset_threshold) <= float(args.delay_threshold):
        raise ValueError('reset_threshold must be greater than delay_threshold')

    package_rows = load_csv_rows(Path(args.package_csv).expanduser().resolve())
    prediction_rows = load_csv_rows(Path(args.predictions_csv).expanduser().resolve())
    base_rows_by_uid = {r['sample_uid']: r for r in package_rows}
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = build_rows(base_rows_by_uid, prediction_rows, float(args.delay_threshold), float(args.reset_threshold))
    action_summary_rows = summarize_actions(rows)
    target_summary_rows = summarize_targets(rows)
    canonical_rows = build_canonical_rows(rows)

    write_csv(out_dir / 'clean_geom_flag_offline_gate_rows.csv', list(rows[0].keys()), rows)
    write_csv(out_dir / 'clean_geom_flag_offline_gate_action_summary.csv', list(action_summary_rows[0].keys()), action_summary_rows)
    write_csv(out_dir / 'clean_geom_flag_offline_gate_target_summary.csv', list(target_summary_rows[0].keys()), target_summary_rows)
    write_csv(out_dir / 'clean_geom_flag_offline_gate_canonical_rows.csv', list(canonical_rows[0].keys()), canonical_rows)

    payload = {
        'package_csv': str(Path(args.package_csv).expanduser().resolve()),
        'predictions_csv': str(Path(args.predictions_csv).expanduser().resolve()),
        'delay_threshold': float(args.delay_threshold),
        'reset_threshold': float(args.reset_threshold),
        'action_summary_rows': action_summary_rows,
        'target_summary_rows': target_summary_rows,
        'canonical_rows': canonical_rows,
    }
    write_json(out_dir / 'clean_geom_flag_offline_gate_definition.json', payload)
    write_text(out_dir / 'clean_geom_flag_offline_gate_definition.md', build_markdown(payload))
    print(f'[clean_geom_flag_offline_gate] saved -> {out_dir}')


if __name__ == '__main__':
    main()
