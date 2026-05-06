"""
Cross-model LTV runner.

Runs 4 experiments in one go:
  (1) Reference: ZS / ICL / LTV on small model (Qwen2.5-14B) only
  (2) Reference: ZS / ICL / LTV on large model (Qwen2.5-32B) only
  (3-1) Cross-model: h_ICL(large) + h_ZS(large)  → W → inference on small model
  (3-2) Cross-model: h_ICL(large) + h_ZS(small)  → W → inference on small model

d_NTP is computed for all LTV variants:
  - References (1)(2): ICL logits vs LTV logits, same model
  - Cross-model (3-1)(3-2): small-model ICL logits vs cross-model LTV logits

Memory strategy: load large model first (reference + extraction), unload, then load small model
once for all remaining work.

Existing code (run_our_ltv.py, wrapper_ltv.py, etc.) is untouched.
"""

import argparse
import copy
import gc
import json
import os
import time
import torch
from transformers import AutoTokenizer

import our_datasets as md
import core.evaluator as ev
import core.metric as metric
import core.utils.utils as utils
from LTV_for_public.LTV.core.wrapper_transfer import CrossModelLTVWrapper, compare_lm_head_label_vectors


def compute_lm_head_alignment(lm_head_large: torch.Tensor, lm_head_small: torch.Tensor) -> torch.Tensor:
    """Compute alignment matrix M: hidden_large → hidden_small via lm_head logit matching.

    Solves: W_small @ M ≈ W_large  (least squares over full vocab)
    → M = (W_small^T W_small)^-1 W_small^T W_large

    Args:
        lm_head_large: (vocab_size, d) float32 — large model lm_head weights
        lm_head_small: (vocab_size, d) float32 — small model lm_head weights

    Returns:
        M: (d, d) float32 on CPU
    """
    WsT_Ws = lm_head_small.T @ lm_head_small   # (d, d)
    WsT_Wl = lm_head_small.T @ lm_head_large   # (d, d)
    M = torch.linalg.solve(WsT_Ws, WsT_Wl)     # (d, d)
    print(f"  [diag] M ||M||_F={M.norm():.3f}  max_sv={torch.linalg.svdvals(M)[0]:.3f}")
    return M


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def print_header(title: str) -> None:
    line = "=" * 60
    print(f"\n{line}\n{title}\n{line}\n")


def _make_save_dir(exp_name: str, large_model: str, small_model: str, dataset_name: str) -> str:
    large_tag = large_model.replace("/", "--")
    small_tag = small_model.replace("/", "--")
    save_dir = os.path.join(exp_name, f"{large_tag}__{small_tag}", dataset_name)
    if os.path.exists(save_dir) and 'debug' not in exp_name:
        raise ValueError(f"Directory already exists: {save_dir}. Delete it or change exp_name.")
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def _build_demo(train_dataset, test_dataset, tokenizer, cfg, demo_seed=None):
    """Generate few-shot demo string and return (baseline_demo, demon_indices)."""
    if demo_seed is None:
        demo_seed = cfg['demo_seed']
    test_max_token = test_dataset.get_max_demonstration_token_length(tokenizer)
    demo, _, demon_indices = train_dataset.gen_few_shot_demonstration(
        tokenizer=tokenizer,
        shot_num=cfg['num_shot'],
        max_demonstration_tok_len=test_max_token,
        add_extra_query=cfg['add_extra_query'],
        example_separator=cfg['example_separator'],
        return_data_index=True,
        seed=demo_seed,
        index_info=None,
    )
    if cfg['add_extra_query']:
        first_anchor = train_dataset.get_dmonstration_template()['format'][0]
        baseline_demo = demo[:demo.rfind(first_anchor)] if first_anchor in demo else demo
    else:
        baseline_demo = demo
    return baseline_demo, demon_indices


@torch.no_grad()
def run_reference_experiment(
    wrapper: CrossModelLTVWrapper,
    tokenizer,
    cfg: dict,
    test_dataset,
    baseline_demo: str,
    train_queries: list,
    model_name: str,
) -> tuple:
    """Run ZS / ICL / LTV for one model as a reference experiment.

    Returns:
        result: dict with zero_shot, few_shot, ltv sub-dicts
        test_few_logits: ICL logits tensor (for d_NTP reference)
        test_few_labels: ICL labels list
    """
    result = {'zero_shot': {}, 'few_shot': {}, 'ltv': {}}
    test_evaluator = ev.Evaluator(test_dataset, batch_size=cfg['bs'])
    is_qwen = 'Qwen' in model_name

    # Zero-shot
    print("  Zero-shot...")
    test_zero = test_evaluator.evaluate(
        wrapper, tokenizer, demonstration='', use_cache=cfg['use_cache'], return_logits=False
    )
    result['zero_shot'] = dict(test_zero) if isinstance(test_zero, dict) else {"result": test_zero}
    print(f"    ZS: {test_zero}")

    # Few-shot ICL
    print("  Few-shot ICL...")
    test_few, test_few_logits, test_few_labels = test_evaluator.evaluate(
        wrapper, tokenizer, demonstration=baseline_demo,
        use_cache=cfg['use_cache'], return_logits=cfg['return_logits'],
    )
    result['few_shot'] = dict(test_few) if isinstance(test_few, dict) else {"result": test_few}
    print(f"    ICL: {test_few}")

    # LTV
    for num_queries in cfg['num_train_queries']:
        q_key = f"{num_queries}_queries"
        q_queries = train_queries[:num_queries]

        for r_lambda in cfg['ridge_lambda']:
            lam_key = f"ridge_lambda_{r_lambda}"
            print(f"  LTV ({q_key}, λ={r_lambda})...")

            adaptive_matrix = wrapper.extract_adaptive_task_vector(
                demo=baseline_demo,
                train_queries=q_queries,
                tokenizer=tokenizer,
                batch_size=cfg['extraction_batch_size'],
                ridge_lambda=r_lambda,
                verbose=True,
            )
            W_f = adaptive_matrix.float()
            print(f"    [diag] ||W||_F={W_f.norm():.3f}  max_sv={torch.linalg.svdvals(W_f)[0]:.3f}")

            with wrapper.inject_adaptive_task_vector(adaptive_matrix):
                test_ltv, test_ltv_logits, test_ltv_labels = test_evaluator.evaluate(
                    wrapper, tokenizer, demonstration='',
                    use_cache=cfg['use_cache'], return_logits=cfg['return_logits'],
                )
            ltv_metrics = dict(test_ltv) if isinstance(test_ltv, dict) else {"result": test_ltv}
            print(f"    LTV: {test_ltv}")

            if cfg.get('compute_d_NTP', False) and test_few_labels == test_ltv_labels:
                d_ntp = metric.compute_d_NTP(test_few_logits, test_ltv_logits, is_qwen=is_qwen)
                ltv_metrics['d_NTP'] = d_ntp

            utils.nested_set(result['ltv'], [q_key, lam_key], ltv_metrics)

    return result, test_few_logits, test_few_labels


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

@torch.no_grad()
def main(args):
    cfg = args.config
    seed = cfg['seed']
    load_in_8bit = cfg['load_in_8bit']
    extraction_batch_size = cfg['extraction_batch_size']
    ridge_lambdas = cfg['ridge_lambda']
    num_train_queries_list = cfg['num_train_queries']
    run_num = cfg.get('run_num', 1)
    is_qwen = True  # both models are Qwen

    utils.set_seed(seed)
    device = utils.set_device(cfg['gpus'][0])

    save_dir = _make_save_dir(
        cfg['exp_name'], cfg['large_model'], cfg['small_model'], args.dataset_name
    )
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=4)

    result_dict = {
        'reference_small': {'zero_shot': [], 'few_shot': [], 'ltv': []},
        'reference_large': {'zero_shot': [], 'few_shot': [], 'ltv': []},
        'scenario_3_1': [],
        'scenario_3_2': [],
        'scenario_3_3': [],
        'scenario_3_4': [],
        'scenario_3_5': [],
        'lm_head_similarity': {},
    }

    train_dataset = md.get_dataset(args.dataset_name, split='train', max_data_num=None, seed=seed)
    test_dataset = md.get_dataset(
        args.dataset_name, split='test',
        max_data_num=cfg['test_data_num'],
        sample_mode=cfg['sample_method'],
        seed=seed,
    )

    max_queries = max(num_train_queries_list)

    # Pre-compute small model demo indices for all runs (tokenizer only, no model load)
    _small_tok = AutoTokenizer.from_pretrained(cfg['small_model'])
    _small_tok.pad_token = _small_tok.eos_token
    _small_tok.padding_side = 'left'
    demon_indices_small_per_run = []
    for run_id in range(run_num):
        _, dem_idx = _build_demo(
            train_dataset, test_dataset, _small_tok, cfg, cfg['demo_seed'] + run_id
        )
        demon_indices_small_per_run.append(dem_idx)
    del _small_tok

    # ------------------------------------------------------------------ #
    # Phase 1: Large model (32B)
    #   - For each run: Reference (2) + extract h_ICL, h_ZS
    #   - LM head vectors from 32B (once)
    # ------------------------------------------------------------------ #
    print_header(f"Phase 1: Large model ({cfg['large_model']})")
    large_model, large_tokenizer, large_model_config = utils.load_model_tokenizer(
        cfg['large_model'], device, output_hidden_states=True, load_in_8bit=load_in_8bit
    )
    large_wrapper = CrossModelLTVWrapper(large_model, large_tokenizer, large_model_config, device)

    per_run_large = []  # stores per-run extraction results

    for run_id in range(run_num):
        print_header(f"Phase 1 | Run {run_id + 1}/{run_num}")
        utils.set_seed(seed + run_id)

        baseline_demo_large, demon_indices_large = _build_demo(
            train_dataset, test_dataset, large_tokenizer, cfg, cfg['demo_seed'] + run_id
        )

        exclude_demo = set()
        if demon_indices_large is not None:
            exclude_demo |= set(demon_indices_large)
        if demon_indices_small_per_run[run_id] is not None:
            exclude_demo |= set(demon_indices_small_per_run[run_id])

        train_queries, _ = utils.build_train_queries(
            train_dataset, max_queries, exclude_indices=exclude_demo
        )
        print(f"Collected {len(train_queries)} train queries (excluded {len(exclude_demo)} demo indices).")

        print_header(f"Reference (2) | Run {run_id + 1}: ZS / ICL / LTV on large model")
        ref_large, test_few_logits_large, test_few_labels_large = run_reference_experiment(
            large_wrapper, large_tokenizer, cfg,
            test_dataset, baseline_demo_large, train_queries,
            cfg['large_model'],
        )
        result_dict['reference_large']['zero_shot'].append(ref_large['zero_shot'])
        result_dict['reference_large']['few_shot'].append(ref_large['few_shot'])
        result_dict['reference_large']['ltv'].append(ref_large['ltv'])

        print(f"Extracting h_ICL from large model (run {run_id + 1})...")
        h_icl_large = large_wrapper.extract_icl_hidden(
            baseline_demo_large, train_queries, large_tokenizer,
            batch_size=extraction_batch_size, verbose=True,
        )
        print(f"  h_ICL shape: {h_icl_large.shape}")
        print(f"  [diag] ||h_ICL|| mean={h_icl_large.float().norm(dim=-1).mean():.3f}  std={h_icl_large.float().norm(dim=-1).std():.3f}")

        print(f"Extracting h_ZS from large model (run {run_id + 1})...")
        h_zs_large = large_wrapper.extract_zs_hidden(
            train_queries, large_tokenizer,
            batch_size=extraction_batch_size, verbose=True,
        )
        print(f"  h_ZS shape: {h_zs_large.shape}")
        print(f"  [diag] ||h_ZS|| mean={h_zs_large.float().norm(dim=-1).mean():.3f}  std={h_zs_large.float().norm(dim=-1).std():.3f}")
        delta_large = (h_icl_large - h_zs_large).float()
        print(f"  [diag] ||delta(ICL-ZS)|| mean={delta_large.norm(dim=-1).mean():.3f}  std={delta_large.norm(dim=-1).std():.3f}")

        per_run_large.append({
            'train_queries': train_queries,
            'h_icl': h_icl_large,
            'h_zs_large': h_zs_large,
            'test_few_logits': test_few_logits_large,
            'test_few_labels': test_few_labels_large,
        })

    lm_head_large = None
    lm_head_matrix_large = None
    if cfg.get('compare_lm_head', False):
        print("Extracting LM head label vectors from large model...")
        lm_head_large = large_wrapper.get_lm_head_label_vectors(train_dataset, large_tokenizer)
        print(f"  Labels: {list(lm_head_large.keys())}")
        print("Extracting full LM head matrix from large model (for Scenario 3-3)...")
        lm_head_matrix_large = large_wrapper.get_lm_head_matrix()
        print(f"  lm_head_matrix_large shape: {lm_head_matrix_large.shape}")

    del large_wrapper, large_model, large_tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print("Large model unloaded.")

    # ------------------------------------------------------------------ #
    # Phase 2: Small model (14B) — load once, do everything
    #   - For each run: Reference (1) + h_ZS(small) + Scenarios 3-1/3-2
    #   - LM head comparison (once)
    # ------------------------------------------------------------------ #
    print_header(f"Phase 2: Small model ({cfg['small_model']})")
    small_model, small_tokenizer, small_model_config = utils.load_model_tokenizer(
        cfg['small_model'], device, output_hidden_states=True, load_in_8bit=load_in_8bit
    )
    small_wrapper = CrossModelLTVWrapper(small_model, small_tokenizer, small_model_config, device)

    lm_head_alignment_M = None
    if cfg.get('compare_lm_head', False) and lm_head_large is not None:
        print("Extracting LM head label vectors from small model...")
        lm_head_small = small_wrapper.get_lm_head_label_vectors(train_dataset, small_tokenizer)
        lm_head_sim = compare_lm_head_label_vectors(lm_head_large, lm_head_small)
        result_dict['lm_head_similarity'] = lm_head_sim
        print(f"  LM head cosine similarities: {lm_head_sim}")

        if lm_head_matrix_large is not None:
            print("Computing lm_head alignment matrix M (for Scenario 3-3)...")
            lm_head_matrix_small = small_wrapper.get_lm_head_matrix()
            lm_head_alignment_M = compute_lm_head_alignment(lm_head_matrix_large, lm_head_matrix_small)
            del lm_head_matrix_small, lm_head_matrix_large

    test_evaluator = ev.Evaluator(test_dataset, batch_size=cfg['bs'])

    for run_id in range(run_num):
        print_header(f"Phase 2 | Run {run_id + 1}/{run_num}")
        utils.set_seed(seed + run_id)

        baseline_demo_small, _ = _build_demo(
            train_dataset, test_dataset, small_tokenizer, cfg, cfg['demo_seed'] + run_id
        )
        run_data = per_run_large[run_id]
        train_queries = run_data['train_queries']

        print_header(f"Reference (1) | Run {run_id + 1}: ZS / ICL / LTV on small model")
        ref_small, test_few_logits_small, test_few_labels_small = run_reference_experiment(
            small_wrapper, small_tokenizer, cfg,
            test_dataset, baseline_demo_small, train_queries,
            cfg['small_model'],
        )
        result_dict['reference_small']['zero_shot'].append(ref_small['zero_shot'])
        result_dict['reference_small']['few_shot'].append(ref_small['few_shot'])
        result_dict['reference_small']['ltv'].append(ref_small['ltv'])

        print(f"Extracting h_ZS from small model (run {run_id + 1})...")
        h_zs_small = small_wrapper.extract_zs_hidden(
            train_queries, small_tokenizer,
            batch_size=extraction_batch_size, verbose=True,
        )
        print(f"  h_ZS shape: {h_zs_small.shape}")
        print(f"  [diag] ||h_ZS(small)|| mean={h_zs_small.float().norm(dim=-1).mean():.3f}  std={h_zs_small.float().norm(dim=-1).std():.3f}")

        # Scenario 3-1, 3-2, 3-3, 3-4, 3-5
        s31_run = {}
        s32_run = {}
        s33_run = {}
        s34_run = {}
        s35_run = {}

        for num_queries in num_train_queries_list:
            q_key = f"{num_queries}_queries"
            h_icl_q = run_data['h_icl'][:num_queries]
            h_zs_large_q = run_data['h_zs_large'][:num_queries]
            h_zs_small_q = h_zs_small[:num_queries]

            for r_lambda in ridge_lambdas:
                lam_key = f"ridge_lambda_{r_lambda}"
                print_header(f"Scenario 3-1 | Run {run_id + 1} | {q_key} | λ={r_lambda}")

                W_31 = small_wrapper.fit_adaptive_matrix(h_icl_q, h_zs_large_q, r_lambda)
                W_31f = W_31.float()
                print(f"  W shape: {W_31.shape}  ||W||_F={W_31f.norm():.3f}  max_sv={torch.linalg.svdvals(W_31f)[0]:.3f}")
                with small_wrapper.inject_adaptive_task_vector(W_31):
                    test_31, logits_31, labels_31 = test_evaluator.evaluate(
                        small_wrapper, small_tokenizer, demonstration='',
                        use_cache=cfg['use_cache'], return_logits=cfg['return_logits'],
                    )
                s31_metrics = dict(test_31) if isinstance(test_31, dict) else {"result": test_31}
                print(f"  Scenario 3-1: {test_31}")

                if cfg.get('compute_d_NTP', False) and test_few_labels_small == labels_31:
                    d_ntp_31 = metric.compute_d_NTP(test_few_logits_small, logits_31, is_qwen=is_qwen)
                    s31_metrics['d_NTP'] = d_ntp_31

                utils.nested_set(s31_run, [q_key, lam_key], s31_metrics)

                print_header(f"Scenario 3-2 | Run {run_id + 1} | {q_key} | λ={r_lambda}")

                W_32 = small_wrapper.fit_adaptive_matrix(h_icl_q, h_zs_small_q, r_lambda)
                W_32f = W_32.float()
                print(f"  W shape: {W_32.shape}  ||W||_F={W_32f.norm():.3f}  max_sv={torch.linalg.svdvals(W_32f)[0]:.3f}")
                with small_wrapper.inject_adaptive_task_vector(W_32):
                    test_32, logits_32, labels_32 = test_evaluator.evaluate(
                        small_wrapper, small_tokenizer, demonstration='',
                        use_cache=cfg['use_cache'], return_logits=cfg['return_logits'],
                    )
                s32_metrics = dict(test_32) if isinstance(test_32, dict) else {"result": test_32}
                print(f"  Scenario 3-2: {test_32}")

                if cfg.get('compute_d_NTP', False) and test_few_labels_small == labels_32:
                    d_ntp_32 = metric.compute_d_NTP(test_few_logits_small, logits_32, is_qwen=is_qwen)
                    s32_metrics['d_NTP'] = d_ntp_32

                utils.nested_set(s32_run, [q_key, lam_key], s32_metrics)

                # Scenario 3-3: M @ h_ICL(32B) → 14B 공간 정렬 후 h_ZS(14B)와 fit
                if lm_head_alignment_M is not None:
                    print_header(f"Scenario 3-3 | Run {run_id + 1} | {q_key} | λ={r_lambda}")
                    M = lm_head_alignment_M.to(h_icl_q.dtype)
                    h_icl_q_aligned = (M @ h_icl_q.T).T.cpu()  # (n, d) in 14B space
                    W_33 = small_wrapper.fit_adaptive_matrix(h_icl_q_aligned, h_zs_small_q, r_lambda)
                    W_33f = W_33.float()
                    print(f"  W shape: {W_33.shape}  ||W||_F={W_33f.norm():.3f}  max_sv={torch.linalg.svdvals(W_33f)[0]:.3f}")
                    with small_wrapper.inject_adaptive_task_vector(W_33):
                        test_33, logits_33, labels_33 = test_evaluator.evaluate(
                            small_wrapper, small_tokenizer, demonstration='',
                            use_cache=cfg['use_cache'], return_logits=cfg['return_logits'],
                        )
                    s33_metrics = dict(test_33) if isinstance(test_33, dict) else {"result": test_33}
                    print(f"  Scenario 3-3: {test_33}")

                    if cfg.get('compute_d_NTP', False) and test_few_labels_small == labels_33:
                        d_ntp_33 = metric.compute_d_NTP(test_few_logits_small, logits_33, is_qwen=is_qwen)
                        s33_metrics['d_NTP'] = d_ntp_33

                    utils.nested_set(s33_run, [q_key, lam_key], s33_metrics)

                # Scenario 3-4: W_31을 M으로 small 공간으로 conjugate 변환 후 적용
                # W*_small = M @ W_31 @ M^-1, v = W*_small @ h_ZS_small
                if lm_head_alignment_M is not None:
                    print_header(f"Scenario 3-4 | Run {run_id + 1} | {q_key} | λ={r_lambda}")
                    M = lm_head_alignment_M.to(W_31.dtype)
                    M_inv = torch.linalg.inv(M.float()).to(M.dtype)
                    W_34 = M @ W_31 @ M_inv
                    W_34f = W_34.float()
                    print(f"  W shape: {W_34.shape}  ||W||_F={W_34f.norm():.3f}  max_sv={torch.linalg.svdvals(W_34f)[0]:.3f}")
                    with small_wrapper.inject_adaptive_task_vector(W_34):
                        test_34, logits_34, labels_34 = test_evaluator.evaluate(
                            small_wrapper, small_tokenizer, demonstration='',
                            use_cache=cfg['use_cache'], return_logits=cfg['return_logits'],
                        )
                    s34_metrics = dict(test_34) if isinstance(test_34, dict) else {"result": test_34}
                    print(f"  Scenario 3-4: {test_34}")

                    if cfg.get('compute_d_NTP', False) and test_few_labels_small == labels_34:
                        d_ntp_34 = metric.compute_d_NTP(test_few_logits_small, logits_34, is_qwen=is_qwen)
                        s34_metrics['d_NTP'] = d_ntp_34

                    utils.nested_set(s34_run, [q_key, lam_key], s34_metrics)

        result_dict['scenario_3_1'].append(s31_run)
        result_dict['scenario_3_2'].append(s32_run)
        result_dict['scenario_3_3'].append(s33_run)
        result_dict['scenario_3_4'].append(s34_run)

    # Save
    result_path = os.path.join(save_dir, 'result_dict.json')
    with open(result_path, 'w') as f:
        json.dump(result_dict, f, indent=4)
    print(f"\nResults saved to: {result_path}")

    del small_wrapper, small_model, small_tokenizer
    gc.collect()
    torch.cuda.empty_cache()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='config/config_cross_model.py')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    config = utils.load_config(args.config_path)

    gpu_id = config['gpus'][0]

    for dataset_name in config['datasets']:
        print(f"\nRunning cross-model experiment on dataset: {dataset_name}")
        print(f"  Large: {config['large_model']}")
        print(f"  Small: {config['small_model']}")

        input_args = argparse.Namespace()
        input_args.dataset_name = dataset_name
        input_args.gpu = gpu_id
        input_args.config = copy.deepcopy(config)

        try:
            main(input_args)
        finally:
            gc.collect()
            torch.cuda.empty_cache()
            print(f"CUDA memory cleared for GPU {gpu_id}")
            time.sleep(3)

    print("All cross-model tasks completed.")
