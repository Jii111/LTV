"""
Translation generation experiment runner (WMT14 En->Fr).

Evaluates:
  - zero_shot        : no context examples
  - few_shot          : n_shot in-context examples prepended to each query
  - bucket_adaptive   : W_0/W_early/W_late ridge, injected at every decode
                        step (main method — see core/tv_bucket.py)
  - w0_only_adaptive  : same W_0, injected once at prefill only (control —
                        equivalent to the old single-point M2-Adaptive)

Metric: corpus-level BLEU + chrF (sacrebleu).

Usage (run from repo root, with PYTHONPATH=repo root):
  python run/run_translation_gen.py --config_path config/config_translation_gen.py --gpu 0
"""

import argparse
import gc
import itertools
import json
import os
import random
import re
import multiprocessing
from multiprocessing import Process, Queue
from typing import List

import sacrebleu
import torch
from tqdm import tqdm
from transformers import StoppingCriteria, StoppingCriteriaList

import core.utils.utils as utils
import core.tv_bucket as tvb
from core.wrapper_base import LlamaWrapper as M2AdaptiveWrapper


class StopStringCriteria(StoppingCriteria):
    """Stop generation once all sequences in the batch match stop_pattern (a regex)."""
    def __init__(self, tokenizer, stop_pattern: str, prompt_len: int):
        self.tokenizer = tokenizer
        self.stop_pattern = stop_pattern
        self.prompt_len = prompt_len

    def __call__(self, input_ids: torch.LongTensor, scores, **_kwargs) -> bool:
        for seq in input_ids:
            text = self.tokenizer.decode(seq[self.prompt_len:], skip_special_tokens=True)
            if re.search(self.stop_pattern, text) is None:
                return False
        return True


def load_translation_dataset(dataset_name, split, max_data_num=None, seed=42):
    if dataset_name == 'wmt14_enfr':
        from our_datasets.wmt14_enfr import WMT14EnFrDataset
        return WMT14EnFrDataset(split=split, max_data_num=max_data_num, seed=seed)
    else:
        raise ValueError(f"Unknown translation dataset: {dataset_name}")


@torch.no_grad()
def generate_predictions(
    model, tokenizer, prompts: List[str], stop_pattern: str,
    max_new_tokens: int = 48, batch_size: int = 1, dataset=None,
    return_raw: bool = False, do_sample: bool = False,
    temperature: float = 1.0, top_p: float = 1.0,
) -> List[str]:
    preds, raw_texts = [], []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i: i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True).to(model.device)
        prompt_len = enc["input_ids"].shape[1]
        stopping_criteria = StoppingCriteriaList([
            StopStringCriteria(tokenizer, stop_pattern, prompt_len)
        ])
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens, do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id, stopping_criteria=stopping_criteria,
        )
        if do_sample:
            gen_kwargs['temperature'] = temperature
            gen_kwargs['top_p'] = top_p
        out_ids = model.generate(**enc, **gen_kwargs)
        for ids in out_ids:
            text = tokenizer.decode(ids[prompt_len:], skip_special_tokens=True)
            parsed = dataset.parse_prediction(text) if dataset else text.strip()
            preds.append(parsed)
            raw_texts.append(text)
    if return_raw:
        return preds, raw_texts
    return preds


def translation_metrics(preds: List[str], refs: List[str]) -> dict:
    if not preds:
        return {'bleu': 0.0, 'chrf': 0.0, 'n': 0}
    bleu = sacrebleu.corpus_bleu(preds, [refs])
    chrf = sacrebleu.corpus_chrf(preds, [refs])
    return {'bleu': bleu.score, 'chrf': chrf.score, 'n': len(preds)}


def _log_predictions(method, queries, preds, targets, n=3):
    print(f"\n[PREDICTIONS:{method}] (first {min(n, len(queries))} samples)")
    for q, g, p in zip(queries[:n], targets[:n], preds[:n]):
        print(f"  src={repr(q[:60])}\n    gold={repr(g)}\n    pred={repr(p)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    cfg = args.config
    utils.set_seed(cfg['seed'])
    device = utils.set_device(args.gpu)

    save_dir = os.path.join(cfg['exp_name'], args.model_name.replace('/', '_'), args.dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)

    print(f"\n{'='*60}\nTranslation Generation: {args.model_name} on {args.dataset_name}\n{'='*60}\n")

    model, tokenizer, model_config = utils.load_model_tokenizer(
        args.model_name, device, output_hidden_states=True,
        load_in_8bit=cfg.get('load_in_8bit', False),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_wrapper = M2AdaptiveWrapper(model, tokenizer, model_config, device)

    train_ds = load_translation_dataset(args.dataset_name, split='train', seed=cfg['seed'])
    val_ds = load_translation_dataset(args.dataset_name, split='validation',
                                      max_data_num=cfg.get('val_data_num'), seed=cfg['seed'])
    # val_ds is loaded for parity with run_math_gen.py but currently unused —
    # lambda selection already happens inside extract_bucket_task_vectors via
    # anchor-level CV over the train-anchor set itself.
    del val_ds
    test_ds = load_translation_dataset(args.dataset_name, split='test',
                                       max_data_num=cfg.get('test_data_num'), seed=cfg['seed'])

    n_shot = cfg['num_shot']
    _ntq = cfg['num_train_queries']
    num_queries_list = _ntq if isinstance(_ntq, list) else [_ntq]
    n_train = max(num_queries_list)  # sample the largest pool once; each num_q slices a prefix of it
    bs = cfg.get('bs', 1)
    max_new = cfg.get('max_new_tokens', getattr(train_ds, 'max_new_tokens', 48))
    stop_pattern = cfg.get('stop_pattern', r'\nEnglish:|\n\n')
    early_range = tuple(cfg.get('early_range', (1, 4)))
    gen_kw = {
        'do_sample': cfg.get('do_sample', False),
        'temperature': cfg.get('temperature', 1.0),
        'top_p': cfg.get('top_p', 1.0),
    }

    result = {'zero_shot': [], 'few_shot': []}
    for num_q in num_queries_list:
        result[f'bucket_adaptive_n{num_q}'] = []
        result[f'w0_only_adaptive_n{num_q}'] = []
    extraction_log = []

    for run_id in tqdm(range(cfg['run_num']), desc="Runs"):
        run_seed = cfg['seed'] + run_id
        utils.set_seed(run_seed)
        print(f"\n{'='*60}\nRun {run_id+1}/{cfg['run_num']}\n{'='*60}")

        all_train = train_ds.all_data[:]
        random.shuffle(all_train)
        demo_items = all_train[:n_shot]
        train_items = all_train[n_shot: n_shot + n_train]

        demo = train_ds.build_demo(demo_items)
        train_queries = [train_ds.get_query(item) for item in train_items]

        test_queries = [test_ds.get_query(item) for item in test_ds.all_data]
        test_targets = [test_ds.get_answer(item) for item in test_ds.all_data]

        # Zero-shot
        preds = generate_predictions(model, tokenizer, test_queries, stop_pattern,
                                     max_new_tokens=max_new, batch_size=bs,
                                     dataset=test_ds, **gen_kw)
        result['zero_shot'].append(translation_metrics(preds, test_targets))
        print(f"[Run {run_id}] Zero-shot   BLEU={result['zero_shot'][-1]['bleu']:.2f} "
              f"chrF={result['zero_shot'][-1]['chrf']:.2f}")
        if run_id == 0:
            _log_predictions('zero_shot', test_queries, preds, test_targets)

        # Few-shot
        fs_prompts = [demo + q for q in test_queries]
        preds = generate_predictions(model, tokenizer, fs_prompts, stop_pattern,
                                     max_new_tokens=max_new, batch_size=bs,
                                     dataset=test_ds, **gen_kw)
        result['few_shot'].append(translation_metrics(preds, test_targets))
        print(f"[Run {run_id}] Few-shot    BLEU={result['few_shot'][-1]['bleu']:.2f} "
              f"chrF={result['few_shot'][-1]['chrf']:.2f}")
        if run_id == 0:
            _log_predictions('few_shot', test_queries, preds, test_targets)

        # num_train_queries sweep: each value gets its own extraction (task
        # vectors depend on how many anchors they were fit on) and its own
        # pair of result rows, matching run_math_baselines.py's convention.
        for num_q in num_queries_list:
            tq = train_queries[:num_q]
            print(f"[Run {run_id}] Extracting bucket task vectors ({num_q} anchors)...")

            Ws, chosen_lambdas, diagnostics = tvb.extract_bucket_task_vectors(
                model_wrapper, tokenizer, demo, tq,
                max_new_tokens=max_new, stop_pattern=stop_pattern, early_range=early_range,
                ridge_lambdas=cfg['ridge_lambdas'], n_cv_folds=cfg['n_cv_folds'],
                repeat_ngram=cfg['repeat_ngram'], repeat_min_repeats=cfg['repeat_min_repeats'],
            )
            print(f"[Run {run_id}] (n={num_q}) Extraction diagnostics: {diagnostics}")
            print(f"[Run {run_id}] (n={num_q}) Chosen lambdas: {chosen_lambdas}")
            extraction_log.append({'run_id': run_id, 'num_train_queries': num_q,
                                   'lambdas': chosen_lambdas, **diagnostics})

            # bucket_adaptive (main method): per-step injection across all buckets
            key = f'bucket_adaptive_n{num_q}'
            with tvb.inject_bucket_stepwise(model_wrapper, W_w0=Ws['w0'], W_early=Ws['early'],
                                            W_late=Ws['late'], early_range=early_range):
                preds = generate_predictions(model, tokenizer, test_queries, stop_pattern,
                                             max_new_tokens=max_new, batch_size=bs,
                                             dataset=test_ds, **gen_kw)
            result[key].append(translation_metrics(preds, test_targets))
            print(f"[Run {run_id}] Bucket-Adaptive(n={num_q}) BLEU={result[key][-1]['bleu']:.2f} "
                  f"chrF={result[key][-1]['chrf']:.2f}")
            if run_id == 0:
                _log_predictions(key, test_queries, preds, test_targets)

            # w0_only_adaptive (control): reuses Ws['w0'], prefill-only injection
            key = f'w0_only_adaptive_n{num_q}'
            with tvb.inject_bucket_stepwise(model_wrapper, W_w0=Ws['w0'], early_range=early_range):
                preds = generate_predictions(model, tokenizer, test_queries, stop_pattern,
                                             max_new_tokens=max_new, batch_size=bs,
                                             dataset=test_ds, **gen_kw)
            result[key].append(translation_metrics(preds, test_targets))
            print(f"[Run {run_id}] W0-Only-Adaptive(n={num_q}) BLEU={result[key][-1]['bleu']:.2f} "
                  f"chrF={result[key][-1]['chrf']:.2f}")
            if run_id == 0:
                _log_predictions(key, test_queries, preds, test_targets)

        with open(os.path.join(save_dir, 'result_dict.json'), 'w') as f:
            json.dump({'result': result, 'extraction_log': extraction_log}, f, indent=2)

    # Summary
    print(f"\n===== Summary =====")
    summary = {}
    for method, runs in result.items():
        for metric in ('bleu', 'chrf'):
            vals = [r[metric] for r in runs]
            mean = sum(vals) / len(vals) if vals else 0.0
            std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5 if vals else 0.0
            summary[f'{method}_{metric}'] = {'mean': mean, 'std': std, 'runs': vals}
            print(f"  {method:<20s} {metric:<5s} = {mean:.2f} ± {std:.2f}")

    with open(os.path.join(save_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {save_dir}")

    del model_wrapper, model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='config/config_translation_gen.py')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--dataset', type=str, default=None)
    return parser.parse_args()


def run_task(gpu_id, cfg, task_queue):
    import argparse as _ap
    while not task_queue.empty():
        model_name, dataset_name = task_queue.get()
        input_args = _ap.Namespace(
            model_name=model_name, dataset_name=dataset_name, gpu=gpu_id, config=cfg,
        )
        try:
            main(input_args)
        finally:
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    args = get_args()
    config = utils.load_config(args.config_path)

    models = [args.model] if args.model else config['models']
    datasets = [args.dataset] if args.dataset else config['datasets']

    combinations = list(itertools.product(models, datasets))
    task_queue = Queue()
    for combo in combinations:
        task_queue.put(combo)

    processes = [Process(target=run_task, args=(gpu_id, config, task_queue))
                 for gpu_id in config['gpus']]
    for p in processes:
        p.start()
    for p in processes:
        p.join()

    print("All translation generation tasks completed.")
