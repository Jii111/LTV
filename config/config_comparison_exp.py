config = {}

### Main Configuration ###
config['exp_name'] = 'results/comparison_exp'
config['gpus'] = ['0']
config['models'] = ['meta-llama/Llama-3.1-8B']
config['datasets'] = ['agnews', 'dbpedia', 'sst5', 'trec', 'sst2', 'subj', 'mr', 'hate_speech18']

# Few-shot / data
config['num_shot'] = 30
config['bs'] = 1
config['test_data_num'] = 500
config['sample_method'] = 'uniform'
config['add_extra_query'] = True
config['example_separator'] = '\n'

# Run settings
config['run_num'] = 1
config['seed'] = 42
config['demo_seed'] = 12
config['metric'] = 'acc'
config['use_cache'] = False
config['load_in_8bit'] = False
config['compute_L_mse'] = True  # L_MSE (eq. 11) for zero-ref / LTV / MLP variants

# LTV extraction
config['num_train_queries'] = 256
config['ridge_lambda'] = 5.0
config['extraction_batch_size'] = 1

# MLP comparison
config['mlp_layers'] = [2, 4, 8, 16]
config['mlp'] = {
    'enabled': True,
    'num_train_queries': 256,
    'batch_size': 8,
    'epochs': 20,
    'lr': 1e-3,
    'warmup_ratio': 0.1,
    'eval_interval_epochs': 0.5,
    'weight_decay': 0.01,
    # 'weight_decay_per_layer': {2: 0.0, 4: 0.001, 8: 0.01, 16: 0.1},
}
