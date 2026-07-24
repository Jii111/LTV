"""
Config for WMT14 En->Fr translation generation experiment.
Use with: run/run_translation_gen.py

Compares: zero_shot, few_shot, bucket_adaptive (W_0/W_early/W_late, per-step
injection), w0_only_adaptive (control: same W_0, prefill-only injection —
equivalent to the old single-point M2-Adaptive from run_math_gen.py).

Extraction is ~2x forward passes (full-length teacher-forced icl/zs
sequences, not just the boundary token) plus a greedy-decode call per
anchor, so num_train_queries/num_shot start smaller here than in
config_math_gen.py — scale up once a run has been timed.
"""

config = {}

config['exp_name'] = 'exps/translation_gen'
config['gpus'] = ['0']
config['models'] = ['meta-llama/Llama-3.1-8B']
config['datasets'] = ['wmt14_enfr']

config['run_num'] = 1
config['seed'] = 42

config['num_shot'] = 30                    # few-shot demo examples
config['num_train_queries'] = [256]     # anchors for task-vector extraction — swept
                                            # externally (like run_math_baselines.py):
                                            # each value gets its own extraction + result rows
                                            # (bucket_adaptive_n{v} / w0_only_adaptive_n{v}).
                                            # Unlike this, ridge_lambdas below is NOT swept
                                            # externally — it's the candidate pool for the
                                            # internal anchor-level CV in extract_bucket_task_vectors.
config['val_data_num'] = 32
config['test_data_num'] = 100

config['bs'] = 1                  # keep 1: injection hook assumes no padding
config['max_new_tokens'] = 48
config['stop_pattern'] = r'\nEnglish:|\n\n'  # literal template boundary — see
                                              # our_datasets/wmt14_enfr.py's
                                              # _NEXT_TURN_PATTERN comment for why

# --- bucket ridge ---
config['early_range'] = (1, 4)               # t=1..4 -> W_early, t>=5 -> W_late
config['ridge_lambdas'] = [5] # [0.1, 1.0, 5.0, 10.0, 50.0]
config['n_cv_folds'] = 5
config['repeat_ngram'] = 3
config['repeat_min_repeats'] = 4

config['load_in_8bit'] = True
config['do_sample'] = False       # greedy for both eval and teacher decode
config['temperature'] = 1.0
config['top_p'] = 1.0
