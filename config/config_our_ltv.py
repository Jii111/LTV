config = {}

### Main Configuration ###
config['exp_name'] = 'results/ltv'  
config['gpus'] = ['0']
config['models'] = ['meta-llama/Llama-3.1-8B'] # 'meta-llama/Llama-2-7b-hf', 'meta-llama/Llama-2-13b-hf', 'Qwen/Qwen3-8B', 'Qwen/Qwen2.5-7B'
config['datasets'] = ['sst2','sst5','mr','subj','trec','hate_speech18','agnews','dbpedia'] 
config['num_shot'] = 30 # number of shots, which ensures label balance by including the maximum possible examples per label
config['bs'] = 16  # batch size

# Baseline-only sweep: both gradient baselines run with their own published
# training recipes (objective, init, lr, schedule, step count, selection).
# Zero-shot / ICL / LTV rows come from earlier sweeps, so they are OFF here;
# flip run_baseline back on to also record d_NTP / L_mse against the
# same-run ICL logits (without it only acc / macro-F1 are recorded).
config['run_baseline'] = False
config['run_ltv'] = False

# Learned-TV baseline (Yang et al., ICLR 2026): gradient-trained single vector,
# injected at the input of one decoder layer at the label position.
# Their published recipe as-is:
config['run_learned_tv'] = True
config['learned_tv'] = {
    'losses': ['ce'],          # their objective: gold-label first-token CE on zero-shot
                               # prompts ('lmse', our eq.-11 variant, stays implemented)
    'layer': 'mid',            # their best configuration: middle decoder layer
    'lr': 5e-3,                # their released code (train_ltv.py AdamW(..., lr=5e-3));
                               # we follow the code rather than the paper text's 1e-3.
                               # The per-sample linear-decay schedule (their formula,
                               # denominator num_epochs x len(train_set)) is reproduced
                               # inside the wrapper.
    'weight_decay': 0.01,
    'epochs': 10,              # their num_epochs
    'samples_per_epoch': 100,  # their per-epoch sample count -> up to 1,000 batch-1 steps
    'patience': 2,             # their early stopping on held-out accuracy
    'val_ratio': 0.4,          # their 600/400 train/val split ratio, applied to the anchor pool
    'num_train_queries': 256,  # anchor queries; 'ce' consumes their gold labels
    'init_scale': 0.1,         # their U(-0.1, 0.1) init
}

# Learnable-TV baseline (Saglam et al., ACL Findings 2025): a learnable
# (n_layers x n_heads) mixing matrix over per-layer ICL-activation bases,
# added to every decoder layer's output at the label position.
# Their published recipe as-is (batch 1 instead of their 32; their
# 2000-iteration update count is preserved). Caveat kept on record: Adam
# bounds per-entry travel by ~lr per step, so their own budget moves Phi only
# 2000 x 5e-5 = 0.10 against a randn init of std 1 — Phi's scale comes from
# the init, and training perturbs it ~10%. That is their method as published;
# check `curve` (phi_l2 / phi_shift_l2) before interpreting the number.
config['run_learnable_tv'] = True
config['learnable_tv'] = {
    'losses': ['ce'],          # their objective: gold-label CE on label-shuffled k-shot
                               # prompts ('lmse', our eq.-11 variant, stays implemented)
    'k_shot': 10,              # shots in the shuffled CE prompts (their README setting)
    'weight_decay': 0.0,       # their Adam has no weight decay
    'init': 'randn',           # their Gaussian init (ltv.py:17)
    'lr': 5e-5,                # their Adam lr
    'epochs': 20,
    'samples_per_epoch': 100,  # 20 x 100 = their 2000 iterations, batch 1.
                               # Selection = lowest epoch-mean train loss, no early
                               # stop (their effective rule; wrapper docstring).
    'val_ratio': 0.0,          # their loop has no genuine held-out split (the wrapper
                               # still carves 1 sample as a minimal convergence log)
    'num_train_queries': 256,  # anchor queries: basis construction + 'ce' gold labels
}

# Experiment settings
config['return_logits'] = True
config['run_num'] = 5          # 5 seeds -> the mean +- std reported in the table
config['seed'] = 42
config['demo_seed'] = 12
config['load_in_8bit'] = True
config['use_cache'] = False

# Data settings
config['sample_method'] = 'uniform'  # 'random', 'uniform'
config['add_extra_query'] = True
config['example_separator'] = '\n'

# Evaluation settings
config['test_data_num'] = 500
config['compute_d_NTP'] = True
config['compute_L_mse'] = True  # L_MSE (eq. 11): E||h_icl - h_tv||^2 at final-layer label position
config['metric'] = 'acc'  # 'acc', 'macro_f1'
 
# LTV extraction
config['num_train_queries'] = [256]  # Number of training queries for task vector learning
config['ridge_lambda'] = [5.0] 
config['extraction_batch_size'] = 1
config['save_task_vectors'] = False
config['save_logits'] = True
