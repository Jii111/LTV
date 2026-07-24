config = {}

### Main Configuration ###
config['exp_name'] = 'results/ltv'  
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
config['run_learned_tv'] = True
config['learned_tv'] = {
    'losses': ['lmse', 'ce'],  # 'lmse': label-free, trains on our eq.-11 proxy vs ICL hiddens
                               # 'ce'  : paper-faithful gold-label cross-entropy (reviewer-facing row)
    'layer': 'mid',            # their best configuration: middle decoder layer
    'lr': 1e-3,                # paper text (their released code uses 5e-3)
    'weight_decay': 0.01,
    'epochs': 10,
    'samples_per_epoch': 100,
    'patience': 2,
    'val_ratio': 0.2,          # 80/20 train/val split of the anchor pool
    'num_train_queries': 256,  # same anchor budget as LTV ('ce' additionally consumes gold labels)
    'init_scale': 0.1,
}

# Experiment settings
config['return_logits'] = True
config['run_num'] = 1
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
