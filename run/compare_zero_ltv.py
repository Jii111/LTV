"""Aggregate the f = 0 (zero-shot) vs f = LTV paired comparison across benchmarks.

Reads results/ltv/<model>/<dataset>/result_dict.json produced by run_our_ltv.py
(cells carry d_NTP / L_mse_logit and their *_zero_ref counterparts) and emits:
  1. an aligned table: per benchmark, d_NTP: zero -> LTV and
     L_mse_logit: zero -> LTV, with run counts and reduction percentages;
  2. zero_vs_ltv_summary.json holding everything in the table (per-dataset
     means/stds, reduction %, and cross-benchmark averages).

Usage:
  python -m run.compare_zero_ltv --results_dir results/ltv \
      --model_name meta-llama/Llama-3.1-8B
"""

import argparse
import glob
import json
import os
import statistics as st


def collect_cells(result_dict_path):
    with open(result_dict_path) as f:
        j = json.load(f)
    cells = []
    for run in j.get('test_result', {}).get('ltv', []):
        for q_entry in run.values():
            for lam_entry in q_entry.values():
                cells.append(lam_entry)
    return cells


def mean_std(xs):
    if not xs:
        return None, None
    return st.mean(xs), (st.stdev(xs) if len(xs) > 1 else 0.0)


def reduction_pct(zero, ltv):
    if zero is None or ltv is None or zero == 0:
        return None
    return 100.0 * (ltv - zero) / zero


def main(args):
    model_dir = os.path.join(args.results_dir, args.model_name)
    per_dataset = {}

    for path in sorted(glob.glob(os.path.join(model_dir, '*', 'result_dict.json'))):
        ds = os.path.basename(os.path.dirname(path))
        cells = collect_cells(path)

        d_ltv, d_ltv_std = mean_std([c['d_NTP'] for c in cells if 'd_NTP' in c])
        d_zero, d_zero_std = mean_std([c['d_NTP_zero_ref'] for c in cells if 'd_NTP_zero_ref' in c])
        m_ltv, m_ltv_std = mean_std([c['L_mse_logit'] for c in cells if 'L_mse_logit' in c])
        m_zero, m_zero_std = mean_std([c['L_mse_logit_zero_ref'] for c in cells if 'L_mse_logit_zero_ref' in c])

        if d_ltv is None:
            print(f"[skip] {ds}: no d_NTP cells in {path}")
            continue

        per_dataset[ds] = {
            'runs': len(cells),
            'd_NTP': {
                'zero': d_zero, 'zero_std': d_zero_std,
                'ltv': d_ltv, 'ltv_std': d_ltv_std,
                'reduction_pct': reduction_pct(d_zero, d_ltv),
            },
            'L_mse_logit': {
                'zero': m_zero, 'zero_std': m_zero_std,
                'ltv': m_ltv, 'ltv_std': m_ltv_std,
                'reduction_pct': reduction_pct(m_zero, m_ltv),
            },
        }

    if not per_dataset:
        print(f"No result_dict.json with pair data under {model_dir}")
        return

    def avg_of(metric_key, side):
        vals = [v[metric_key][side] for v in per_dataset.values() if v[metric_key][side] is not None]
        return st.mean(vals) if vals else None

    average = {}
    for key in ('d_NTP', 'L_mse_logit'):
        z, l = avg_of(key, 'zero'), avg_of(key, 'ltv')
        average[key] = {'zero': z, 'ltv': l, 'reduction_pct': reduction_pct(z, l)}

    summary = {'model': args.model_name, 'per_dataset': per_dataset, 'average': average}
    out_path = args.out or os.path.join(model_dir, 'zero_vs_ltv_summary.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)

    def fmt_pair(entry, prec):
        z, l, r = entry['zero'], entry['ltv'], entry['reduction_pct']
        zs = f"{z:.{prec}f}" if z is not None else '-'
        ls = f"{l:.{prec}f}" if l is not None else '-'
        tail = f" ({r:+.0f}%)" if r is not None else ''
        return f"{zs:>10} -> {ls:<10}{tail}"

    print(f"\n{'dataset':>14} | {'runs':>4} | {'d_NTP: zero -> LTV':>32} | {'L_mse_logit: zero -> LTV':>34}")
    for ds, v in per_dataset.items():
        print(f"{ds:>14} | {v['runs']:>4} | {fmt_pair(v['d_NTP'], 4):>32} | {fmt_pair(v['L_mse_logit'], 2):>34}")
    print(f"{'average':>14} | {'':>4} | {fmt_pair(average['d_NTP'], 4):>32} | {fmt_pair(average['L_mse_logit'], 2):>34}")
    print(f"\nSaved -> {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='results/ltv')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.1-8B')
    parser.add_argument('--out', type=str, default=None)
    main(parser.parse_args())
