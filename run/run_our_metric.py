import argparse
import json
import os
import time
from typing import Dict, List, Tuple, Union

import torch

import core.metric as metric
import core.utils.utils as utils


def load_logits(file_path: str) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], Dict]:
    data = torch.load(file_path, map_location='cpu')
    if isinstance(data, dict) and 'logits' in data:
        return data['logits'], data
    return data, {}


def collect_logits_map(logit_dir: str, prefix: str) -> Dict[str, Dict]:
    logit_map: Dict[str, Dict] = {}
    if not os.path.isdir(logit_dir):
        return logit_map

    for fname in os.listdir(logit_dir):
        if not (fname.startswith(prefix) and fname.endswith('.pt')):
            continue
        path = os.path.join(logit_dir, fname)
        logits, meta = load_logits(path)
        run_name = meta.get('run_name') or fname[len(prefix):-3]
        logit_map[run_name] = {
            'logits': logits,
            'meta': meta,
            'file': path,
        }
    return logit_map


def main(args):
    cfg = utils.load_config(args.config_path)

    model_name = cfg['model_name']
    datasets = cfg['datasets']
    method_name = cfg.get('method_name', 'Ours')
    result_dir = cfg['result_dir']
    save_dir = cfg.get('save_dir', os.path.join(result_dir,'metric_result'))
    plot_path = cfg.get('plot_path', os.path.join(save_dir, 'd_NTP.png'))

    os.makedirs(save_dir, exist_ok=True)

    datasets_map = {ds: {method_name: []} for ds in datasets}
    result_dict = {'per_file': {}, 'summary': {}}

    is_qwen = 'qwen' in model_name.lower()

    for dataset in datasets:
        icl_logit_dir_tpl = 'results/ltv/{model}/{dataset}/logits_icl'
        icl_dir = os.path.join(result_dir,model_name, dataset,'logits_icl')
        tv_dir = os.path.join(result_dir,model_name, dataset,'logits_tv')

        icl_map = collect_logits_map(icl_dir, prefix='icl_')
        tv_map = collect_logits_map(tv_dir, prefix='tv_')

        if dataset not in result_dict['per_file']:
            result_dict['per_file'][dataset] = []

        for run_name, tv_entry in tv_map.items():
            icl_entry = icl_map.get(run_name)
            if icl_entry is None:
                continue

            d_ntp = metric.compute_d_NTP(
                icl_entry['logits'], tv_entry['logits'], is_qwen=is_qwen
            )

            datasets_map.setdefault(dataset, {}).setdefault(method_name, []).append(d_ntp)
            result_dict['per_file'][dataset].append({
                'run_name': run_name,
                'd_NTP': d_ntp,
                'icl_file': icl_entry['file'],
                'tv_file': tv_entry['file'],
                'meta': {k: v for k, v in tv_entry['meta'].items() if k not in ('logits', 'labels')}
            })

        if dataset in datasets_map:
            vals = datasets_map[dataset].get(method_name, [])
            result_dict['summary'][dataset] = {
                'count': len(vals),
                'mean_d_NTP': float(sum(vals) / len(vals)) if vals else None,
            }

    result_path = os.path.join(save_dir, 'd_NTP_result.json')
    with open(result_path, 'w') as f:
        json.dump(result_dict, f, indent=4)

    metric.plot_d_NTP(datasets_map, plot_path)

    print(f"Saved d_NTP results -> {result_path}")
    print(f"Saved plot -> {plot_path}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='config/config_our_metric.py')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    start = time.time()
    main(args)
    print(f"Done in {time.time() - start:.2f}s")
