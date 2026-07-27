config = {}

### Main Configuration ###
config['exp_name'] = 'results/loreft'  # feat/loreft-baseline sweep (init_exp_path refuses to reuse an existing dir)
config['gpus'] = ['0']
config['models'] = ['meta-llama/Llama-3.1-8B'] # 'meta-llama/Llama-2-7b-hf', 'meta-llama/Llama-2-13b-hf', 'Qwen/Qwen3-8B', 'Qwen/Qwen2.5-7B'
config['datasets'] = ['sst2','sst5','mr','subj','trec','hate_speech18','agnews','dbpedia'] 
config['num_shot'] = 30 # number of shots, which ensures label balance by including the maximum possible examples per label
config['bs'] = 16  # batch size

# Run LTV
config['run_baseline'] = True
config['run_ltv'] = True

# Learned-TV baseline (Yang et al., ICLR 2026): gradient-trained single vector,
# injected at the input of one decoder layer at the label position.
# Off on feat/loreft-baseline: the Yang/Saglam sweep belongs to PR #4 and
# rerunning it here would double the cost. Flip on to run everything at once.
config['run_learned_tv'] = False
config['learned_tv'] = {
    # Yang et al. is NOT an ICL method (they train on zero-shot prompts with gold
    # labels), so the row we report is the ICL-based 'lmse' variant: their
    # architecture, our label-free eq.-11 objective against 30-shot ICL teacher
    # hiddens. Add 'ce' to also reproduce their own gold-label objective.
    'losses': ['lmse'],
    'layer': 'mid',            # their best configuration: middle decoder layer
    'lr': 5e-3,                # their released code (train_ltv.py AdamW(..., lr=5e-3));
                               # we follow the code rather than the paper text's 1e-3.
    'weight_decay': 0.01,
    'epochs': 8,
    'samples_per_epoch': 100,  # 800 batch-1 steps, same as Learnable-TV so every
                               # gradient baseline gets an identical update budget.
                               # Unlike Learnable-TV this budget is NOT scale-limited:
                               # reachable travel is 800 x 5e-3 = 4.0 against an init of
                               # U(-0.1,0.1) (mean |theta| 0.05), so theta is genuinely
                               # learned rather than set by its init.
    'patience': 2,
    'val_ratio': 0.2,          # 80/20 train/val split of the anchor pool
    'num_train_queries': 256,  # same anchor budget as LTV ('ce' additionally consumes gold labels)
    'init_scale': 0.1,
}

# Learnable-TV baseline (Saglam et al., ACL Findings 2025): a learnable
# (n_layers x n_heads) mixing matrix over per-layer ICL-activation bases,
# added to every decoder layer's output at the label position.
# Off on feat/loreft-baseline — see run_learned_tv note above.
config['run_learnable_tv'] = False
config['learnable_tv'] = {
    # Saglam et al. IS an ICL method (its basis comes from ICL activations), so we
    # report their own objective: CE on label-shuffled 30-shot ICL prompts.
    # Add 'lmse' to also run the label-free variant.
    # BOTH objectives are reported for Saglam: 'ce' is their own (the faithful row),
    # 'lmse' is the label-free variant run under our eq.-11 objective, which makes
    # the two baselines comparable on the same objective as well.
    'losses': ['ce', 'lmse'],
    'k_shot': 30,              # shuffled CE prompts reuse our 30-shot demonstration
    'weight_decay': 0.0,       # their Adam has no weight decay
    # Phi is only (n_layers x n_heads) = 32x32 = 1024 scalars, so their 2000 iters
    # are NOT a capacity requirement. Adam bounds per-entry travel by ~lr/step, so
    # their own budget moves Phi just 2000 x 5e-5 = 0.10 against a randn init of
    # std 1 — i.e. the INIT sets Phi's scale and training only perturbs it ~2%
    # (verified numerically). Two consequences:
    #   1. raising the step count is nearly useless here (800->2400 takes the
    #      reachable scale from 0.04 to 0.12, both << 1);
    #   2. zero-init at their lr is a DEAD setting: |Phi| ~ 0.04, v ~ 0, and the
    #      method degenerates to zero-shot (observed on sst2: L_mse_logit 254 vs
    #      zero-shot ref 255).
    # So we hold steps x lr at the scale Phi must reach instead of inflating steps.
    'init': 'zero',            # neutral start (v=0) rather than a random draw...
    'lr': 1e-3,                # ...paired with lr raised so 800 x 1e-3 = 0.8 ~ the
                               # 0.81 mean magnitude their randn init hands the method
                               # for free. Set init='randn', lr=5e-5 to reproduce the
                               # paper's own combination instead.
    'epochs': 8,
    'samples_per_epoch': 100,  # 800 batch-1 steps; selection = lowest epoch train loss (paper rule).
                               # 'ce' prompts are 30-shot (~600 tokens), i.e. ~20x longer than the
                               # zero-shot prompts used elsewhere. `curve` logs phi_l2/phi_shift_l2
                               # per epoch — check those before changing this.
    'val_ratio': 0.2,          # held-out slice used ONLY for the convergence diagnostic
    'num_train_queries': 256,  # same anchor budget as LTV; 'ce' also uses their gold labels
}

# LoReFT-style baseline (Wu et al., NeurIPS 2024, pyreft): explicitly
# optimized low-rank representation intervention, requested by reviewer EHiD
# to isolate the contribution of LTV's linear, closed-form (optimization-free)
# formulation. Phi(h) = h + R^T((Wh+b) - Rh) at the label position, input of
# the chosen decoder layer(s). Harness identical to Learned-TV, so the two
# rows differ ONLY in the operator (query-dependent low-rank edit vs constant
# vector), and both differ from LTV only in HOW the mapping is obtained
# (gradient descent vs closed-form ridge).
config['run_loreft'] = True
config['loreft'] = {
    # Reported row: 'lmse' — LoReFT is not an ICL method (supervised FT), so
    # like Learned-TV the comparable row is the label-free eq.-11 proxy
    # against ICL teacher hiddens: the same information budget as our LTV,
    # but explicitly gradient-optimized (the axis reviewer EHiD asks about).
    # 'ce' (gold first-token CE on zero-shot prompts, + the 30 demo labels
    # via extra_queries) stays implemented — add it here to run it.
    'losses': ['lmse'],
    'layers': 'all',           # original recipe: intervene at EVERY decoder
                               # layer, trained jointly in one run ('mid'
                               # isolates the operator at the Learned-TV site)
    'rank': 8,                 # their commonsense-recipe rank; params =
                               # 32 x (2 x 4096 x 8 + 8) ~ 2.1M — the original
                               # parameterization (vs Yang 4,096 / Saglam 1,024)
    'lr': 9e-4,                # the ReFT paper's standard LM learning rate.
                               # Not scale-limited: 800 x 9e-4 = 0.72 peak travel
                               # (0.36 under the warmup+decay schedule) vs init
                               # entry scales ~1/sqrt(4096) = 0.016
    'weight_decay': 0.0,
    'dropout': 0.05,           # their recipe (train-time only; injection uses
                               # .eval(), so evaluation stays deterministic)
    'warmup_ratio': 0.1,       # linear warmup then linear decay (their setup)
    'epochs': 8,
    'samples_per_epoch': 100,  # 800 batch-1 steps — identical update budget to
                               # Learned-TV and Learnable-TV
    'patience': 2,
    'val_ratio': 0.2,
    'num_train_queries': 256,  # same anchor budget as LTV
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
