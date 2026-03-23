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
DEFAULT_OUT_DIR = PROJECT_ROOT / 'docs' / 'research' / 'init_risk_clean_geom_flag_offline_gate_simulation_20260323'
DEFAULT_PACKAGE = DEFAULT_REPAIR_ROOT / 'package_feature_first_repair' / 'risk_dataset.csv'
DEFAULT_BASELINE_PRED = DEFAULT_REPAIR_ROOT / 'runs' / 'full_anchor_baseline' / 'predictions.csv'
DEFAULT_FLAG_PRED = DEFAULT_REPAIR_ROOT / 'runs' / 'full_anchor_plus_clean_geom_flag' / 'predictions.csv'
DEFAULT_DELAY_THRESHOLD = 0.50
DEFAULT_RESET_THRESHOLD = 0.90

POLICY_NATIVE = 'native_gate_post'
POLICY_BASELINE = 'baseline_score_gate'
POLICY_FLAG = 'clean_geom_flag_score_gate'


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


def score_gate_action(p_hat: float, delay_threshold: float, reset_threshold: float) -> str:
    if p_hat >= float(reset_threshold):
        return 'reset'
    if p_hat >= float(delay_threshold):
        return 'delay'
    return 'accept'


def native_gate_action(base_row: dict) -> str:
    action = str(base_row.get('gate_post', '') or '').strip()
    if action in ('accept', 'delay', 'reset'):
        return action
    return 'accept'


def collect_policy_rows(
    package_rows_by_uid: dict[str, dict],
    prediction_rows: list[dict],
    delay_threshold: float,
    reset_threshold: float,
) -> list[dict]:
    baseline_by_uid = {r['sample_uid']: r for r in prediction_rows if r.get('policy_name') == POLICY_BASELINE}
    flag_by_uid = {r['sample_uid']: r for r in prediction_rows if r.get('policy_name') == POLICY_FLAG}
    sample_uids = sorted(set(baseline_by_uid.keys()) & set(flag_by_uid.keys()))
    rows = []
    for uid in sample_uids:
        base = package_rows_by_uid[uid]
        pred_base = baseline_by_uid[uid]
        pred_flag = flag_by_uid[uid]
        split = pred_flag['split']
        y_true = safe_int(pred_flag['y_true'], 0)
        baseline_p = float(pred_base['p_hat'])
        flag_p = float(pred_flag['p_hat'])
        rows.append({
            'sample_uid': uid,
            'split': split,
            'sequence': base.get('sequence', pred_flag.get('sequence', '')),
            'window_id': base.get('window_id', pred_flag.get('window_id', '')),
            'y_true': y_true,
            'trigger': base.get('Y_bad_v2_min_default_trigger', ''),
            'clean_geom_flag': 1 if safe_float(base.get('clean_geom_flag'), 0.0) >= 0.5 else 0,
            'native_gate_post': native_gate_action(base),
            'native_gate_post_reason': base.get('gate_post_reason', ''),
            'baseline_p_hat': baseline_p,
            'flag_p_hat': flag_p,
            'baseline_action': score_gate_action(baseline_p, delay_threshold, reset_threshold),
            'flag_action': score_gate_action(flag_p, delay_threshold, reset_threshold),
        })
    return rows


def summarize_policy(rows: list[dict], split: str, policy: str) -> dict:
    srows = [r for r in rows if r['split'] == split]
    if not srows:
        return {}
    action_key = {
        POLICY_NATIVE: 'native_gate_post',
        POLICY_BASELINE: 'baseline_action',
        POLICY_FLAG: 'flag_action',
    }[policy]
    score_key = {
        POLICY_NATIVE: None,
        POLICY_BASELINE: 'baseline_p_hat',
        POLICY_FLAG: 'flag_p_hat',
    }[policy]

    positives = [r for r in srows if int(r['y_true']) == 1]
    negatives = [r for r in srows if int(r['y_true']) == 0]

    def action_of(row):
        return row[action_key]

    protected_pos = [r for r in positives if action_of(r) in ('delay', 'reset')]
    accepted_pos = [r for r in positives if action_of(r) == 'accept']
    delayed_pos = [r for r in positives if action_of(r) == 'delay']
    reset_pos = [r for r in positives if action_of(r) == 'reset']
    delayed_neg = [r for r in negatives if action_of(r) == 'delay']
    reset_neg = [r for r in negatives if action_of(r) == 'reset']
    intervened_neg = [r for r in negatives if action_of(r) in ('delay', 'reset')]
    all_intervened = [r for r in srows if action_of(r) in ('delay', 'reset')]
    future_reset = [r for r in positives if r['trigger'] == 'future_reset']
    clean_future_reset = [r for r in future_reset if int(r['clean_geom_flag']) == 1]
    protected_future_reset = [r for r in future_reset if action_of(r) in ('delay', 'reset')]
    protected_clean_future_reset = [r for r in clean_future_reset if action_of(r) in ('delay', 'reset')]

    return {
        'split': split,
        'policy': policy,
        'num_rows': len(srows),
        'num_positive': len(positives),
        'num_negative': len(negatives),
        'positive_ratio': len(positives) / len(srows) if srows else None,
        'intervention_rate': len(all_intervened) / len(srows) if srows else None,
        'unsafe_accept_count': len(accepted_pos),
        'unsafe_accept_rate_on_positive': len(accepted_pos) / len(positives) if positives else None,
        'protected_positive_count': len(protected_pos),
        'protected_positive_rate': len(protected_pos) / len(positives) if positives else None,
        'positive_delay_count': len(delayed_pos),
        'positive_reset_count': len(reset_pos),
        'negative_delay_count': len(delayed_neg),
        'negative_reset_count': len(reset_neg),
        'negative_intervention_count': len(intervened_neg),
        'negative_intervention_rate': len(intervened_neg) / len(negatives) if negatives else None,
        'negative_reset_rate': len(reset_neg) / len(negatives) if negatives else None,
        'future_reset_count': len(future_reset),
        'future_reset_protected_count': len(protected_future_reset),
        'future_reset_protected_rate': len(protected_future_reset) / len(future_reset) if future_reset else None,
        'clean_future_reset_count': len(clean_future_reset),
        'clean_future_reset_protected_count': len(protected_clean_future_reset),
        'clean_future_reset_protected_rate': len(protected_clean_future_reset) / len(clean_future_reset) if clean_future_reset else None,
        'mean_score': None if score_key is None else mean([float(r[score_key]) for r in srows]),
    }


def compare_against_reference(summary_rows: list[dict], policy: str, reference_policy: str) -> list[dict]:
    out = []
    by_key = {(r['split'], r['policy']): r for r in summary_rows}
    for split in ['train', 'val', 'test']:
        cur = by_key.get((split, policy))
        ref = by_key.get((split, reference_policy))
        if not cur or not ref:
            continue
        extra_neg_int = int(cur['negative_intervention_count']) - int(ref['negative_intervention_count'])
        avoided_unsafe = int(ref['unsafe_accept_count']) - int(cur['unsafe_accept_count'])
        out.append({
            'split': split,
            'policy': policy,
            'reference_policy': reference_policy,
            'avoided_unsafe_accept_count': avoided_unsafe,
            'avoided_unsafe_accept_rate_delta': None if cur['unsafe_accept_rate_on_positive'] is None or ref['unsafe_accept_rate_on_positive'] is None else float(ref['unsafe_accept_rate_on_positive']) - float(cur['unsafe_accept_rate_on_positive']),
            'extra_negative_intervention_count': extra_neg_int,
            'extra_negative_reset_count': int(cur['negative_reset_count']) - int(ref['negative_reset_count']),
            'extra_total_intervention_count': int(round(float(cur['intervention_rate']) * cur['num_rows'])) - int(round(float(ref['intervention_rate']) * ref['num_rows'])),
            'clean_future_reset_protection_delta': None if cur['clean_future_reset_protected_rate'] is None or ref['clean_future_reset_protected_rate'] is None else float(cur['clean_future_reset_protected_rate']) - float(ref['clean_future_reset_protected_rate']),
            'benefit_per_extra_negative_intervention': None if extra_neg_int <= 0 else avoided_unsafe / extra_neg_int,
        })
    return out


def build_canonical_rows(rows: list[dict]) -> list[dict]:
    out = []
    for split in ['train', 'val']:
        for row in rows:
            if row['split'] != split:
                continue
            if row['trigger'] != 'future_reset' or int(row['clean_geom_flag']) != 1:
                continue
            out.append({
                'split': split,
                'sample_uid': row['sample_uid'],
                'native_gate_post': row['native_gate_post'],
                'baseline_p_hat': row['baseline_p_hat'],
                'baseline_action': row['baseline_action'],
                'flag_p_hat': row['flag_p_hat'],
                'flag_action': row['flag_action'],
            })
    return out


def build_markdown(payload: dict) -> str:
    lines = []
    lines.append('# clean_geom_flag offline gate simulation')
    lines.append('')
    lines.append('## Main Takeaway')
    lines.append('')
    for line in payload['takeaways']:
        lines.append(f'- {line}')
    lines.append('')
    lines.append('## Policies')
    lines.append('')
    lines.append(f'- `{POLICY_NATIVE}`: use package-native `gate_post` on the matched prediction sample set')
    lines.append(f'- `{POLICY_BASELINE}`: apply `{payload["delay_threshold"]:.2f}/{payload["reset_threshold"]:.2f}` thresholds to `full_anchor_baseline` scores')
    lines.append(f'- `{POLICY_FLAG}`: apply `{payload["delay_threshold"]:.2f}/{payload["reset_threshold"]:.2f}` thresholds to `full_anchor_plus_clean_geom_flag` scores')
    lines.append('')
    lines.append('## Policy Summary')
    lines.append('')
    lines.append('| split | policy | unsafe accept | protected positive | neg intervention | neg reset | future_reset protected | clean_future_reset protected | intervention rate |')
    lines.append('| --- | --- | --- | --- | --- | --- | --- | --- | --- |')
    for row in payload['summary_rows']:
        lines.append(
            f"| {row['split']} | {row['policy']} | {row['unsafe_accept_count']} | {row['protected_positive_count']} | {row['negative_intervention_count']} | {row['negative_reset_count']} | {row['future_reset_protected_count']} | {row['clean_future_reset_protected_count']} | {fmt(row['intervention_rate'])} |"
        )
    lines.append('')
    lines.append('## Benefit-Cost vs Native')
    lines.append('')
    lines.append('| split | policy | avoided unsafe accept | extra neg intervention | extra neg reset | benefit per extra neg intervention | clean FR protection delta |')
    lines.append('| --- | --- | --- | --- | --- | --- | --- |')
    for row in payload['delta_vs_native_rows']:
        lines.append(
            f"| {row['split']} | {row['policy']} | {row['avoided_unsafe_accept_count']} | {row['extra_negative_intervention_count']} | {row['extra_negative_reset_count']} | {fmt(row['benefit_per_extra_negative_intervention'])} | {fmt(row['clean_future_reset_protection_delta'])} |"
        )
    lines.append('')
    lines.append('## Benefit-Cost vs Baseline Score Gate')
    lines.append('')
    lines.append('| split | policy | avoided unsafe accept | extra neg intervention | extra neg reset | benefit per extra neg intervention | clean FR protection delta |')
    lines.append('| --- | --- | --- | --- | --- | --- | --- |')
    for row in payload['delta_vs_baseline_rows']:
        lines.append(
            f"| {row['split']} | {row['policy']} | {row['avoided_unsafe_accept_count']} | {row['extra_negative_intervention_count']} | {row['extra_negative_reset_count']} | {fmt(row['benefit_per_extra_negative_intervention'])} | {fmt(row['clean_future_reset_protection_delta'])} |"
        )
    lines.append('')
    lines.append('## Canonical Clean future_reset Rows')
    lines.append('')
    lines.append('| split | sample_uid | native gate | baseline p/action | clean_geom_flag p/action |')
    lines.append('| --- | --- | --- | --- | --- |')
    for row in payload['canonical_rows']:
        lines.append(
            f"| {row['split']} | {row['sample_uid']} | {row['native_gate_post']} | {fmt(row['baseline_p_hat'])} / {row['baseline_action']} | {fmt(row['flag_p_hat'])} / {row['flag_action']} |"
        )
    lines.append('')
    lines.append('## Interpretation Boundary')
    lines.append('')
    lines.append('- This is an offline internal simulation, not a backend result.')
    lines.append('- Benefits are quantified as avoided unsafe accepts and hard-case protection.')
    lines.append('- Costs are quantified as extra negative interventions and resets.')
    lines.append('- No external held-out tuning is involved.')
    lines.append('')
    return '\n'.join(lines) + '\n'


def main():
    ap = argparse.ArgumentParser(description='Run offline gate simulation for clean_geom_flag candidate.')
    ap.add_argument('--package_csv', default=str(DEFAULT_PACKAGE))
    ap.add_argument('--baseline_predictions_csv', default=str(DEFAULT_BASELINE_PRED))
    ap.add_argument('--flag_predictions_csv', default=str(DEFAULT_FLAG_PRED))
    ap.add_argument('--out_dir', default=str(DEFAULT_OUT_DIR))
    ap.add_argument('--delay_threshold', type=float, default=DEFAULT_DELAY_THRESHOLD)
    ap.add_argument('--reset_threshold', type=float, default=DEFAULT_RESET_THRESHOLD)
    args = ap.parse_args()

    if float(args.reset_threshold) <= float(args.delay_threshold):
        raise ValueError('reset_threshold must be greater than delay_threshold')

    package_rows_by_uid = {r['sample_uid']: r for r in load_csv_rows(Path(args.package_csv).expanduser().resolve())}
    baseline_pred = load_csv_rows(Path(args.baseline_predictions_csv).expanduser().resolve())
    for r in baseline_pred:
        r['policy_name'] = POLICY_BASELINE
    flag_pred = load_csv_rows(Path(args.flag_predictions_csv).expanduser().resolve())
    for r in flag_pred:
        r['policy_name'] = POLICY_FLAG

    rows = collect_policy_rows(package_rows_by_uid, baseline_pred + flag_pred, float(args.delay_threshold), float(args.reset_threshold))
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for split in ['train', 'val', 'test']:
        for policy in [POLICY_NATIVE, POLICY_BASELINE, POLICY_FLAG]:
            summary = summarize_policy(rows, split, policy)
            if summary:
                summary_rows.append(summary)

    delta_vs_native_rows = compare_against_reference(summary_rows, POLICY_BASELINE, POLICY_NATIVE) + compare_against_reference(summary_rows, POLICY_FLAG, POLICY_NATIVE)
    delta_vs_baseline_rows = compare_against_reference(summary_rows, POLICY_FLAG, POLICY_BASELINE)
    canonical_rows = build_canonical_rows(rows)

    takeaways = []
    by_key = {(r['split'], r['policy']): r for r in summary_rows}
    for split in ['train', 'val', 'test']:
        native = by_key.get((split, POLICY_NATIVE))
        flag = by_key.get((split, POLICY_FLAG))
        if native and flag:
            takeaways.append(
                f"{split}: `{POLICY_FLAG}` avoids {int(native['unsafe_accept_count']) - int(flag['unsafe_accept_count'])} unsafe accepts at the cost of {int(flag['negative_intervention_count']) - int(native['negative_intervention_count'])} extra negative interventions."
            )
    val_delta = next((r for r in delta_vs_baseline_rows if r['split'] == 'val'), None)
    if val_delta is not None:
        takeaways.append(
            f"val: compared with `{POLICY_BASELINE}`, `{POLICY_FLAG}` avoids {val_delta['avoided_unsafe_accept_count']} additional unsafe accept with {val_delta['extra_negative_intervention_count']} extra negative intervention."
        )

    write_csv(out_dir / 'clean_geom_flag_offline_gate_policy_summary.csv', list(summary_rows[0].keys()), summary_rows)
    write_csv(out_dir / 'clean_geom_flag_offline_gate_delta_vs_native.csv', list(delta_vs_native_rows[0].keys()), delta_vs_native_rows)
    write_csv(out_dir / 'clean_geom_flag_offline_gate_delta_vs_baseline.csv', list(delta_vs_baseline_rows[0].keys()), delta_vs_baseline_rows)
    write_csv(out_dir / 'clean_geom_flag_offline_gate_canonical_policy_rows.csv', list(canonical_rows[0].keys()), canonical_rows)

    payload = {
        'delay_threshold': float(args.delay_threshold),
        'reset_threshold': float(args.reset_threshold),
        'summary_rows': summary_rows,
        'delta_vs_native_rows': delta_vs_native_rows,
        'delta_vs_baseline_rows': delta_vs_baseline_rows,
        'canonical_rows': canonical_rows,
        'takeaways': takeaways,
    }
    write_json(out_dir / 'clean_geom_flag_offline_gate_simulation.json', payload)
    write_text(out_dir / 'clean_geom_flag_offline_gate_simulation.md', build_markdown(payload))
    print(f'[clean_geom_flag_offline_gate_sim] saved -> {out_dir}')


if __name__ == '__main__':
    main()
