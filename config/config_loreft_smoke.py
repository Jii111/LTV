"""Smoke test for the LoReFT baseline: one dataset, tiny budgets, one run.
Exercises every code path (both objectives, ICL-target collection, injection,
metrics, curve/info JSON) in ~10 min on one GPU. Expected sanity marks:
EXIT=0; result_dict.json has test_result['loreft'][0]['ce'|'lmse'] with
'curve' entries whose ortho_err ~ 1e-7 and rotate_drift > 0; train_final_lr
follows warmup->decay; acc within a few points of zero-shot is fine at this
budget — the smoke verifies plumbing, not performance.
"""
# load_config() adds this file's directory to sys.path before importing,
# so the base config is importable as a sibling module. Deep-copied so the
# overrides below never leak into the base module's dict.
import copy
from config_our_ltv import config as _base
config = copy.deepcopy(_base)

config['exp_name'] = 'results/loreft_smoke'
config['datasets'] = ['sst2']
config['run_num'] = 1
config['test_data_num'] = 64
config['bs'] = 8

config['loreft'].update({
    'num_train_queries': 24,
    'epochs': 2,
    'samples_per_epoch': 8,   # 16 steps total
})
config['num_train_queries'] = [24]
