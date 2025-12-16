config = {}

# Experiment naming / paths
config['exp_name'] = 'exps/error_exp'
config['gpus'] = ['0']

# Models / datasets
config['models'] = ['Qwen/Qwen2.5-7B']
config['datasets'] = [
    'agnews',
    'dbpedia',
    'sst5',
    'trec',
    'sst2',
    'subj',
    'mr',
    'hate_speech18',
]

# Data / evaluation
config['shot_per_class'] = 30
config['bs'] = 1
config['val_data_num'] = 32
config['test_data_num'] = 500
config['sample_method'] = 'uniform'
config['use_cache'] = False
config['add_extra_query'] = True
config['example_separator'] = '\n'

# Runs / seeds
config['run_num'] = 1
config['seed'] = 42
config['demo_seed'] = 12

# Metric placeholder (unused for MSE-only flow)
config['metric'] = 'acc'

# M2 settings
config['num_train_queries_m2'] = 256
config['ridge_lambda'] = 0.1
config['extraction_batch_size'] = 8

# MLP settings (2-layer square MLP)
config['mlp'] = {
    'enabled': True,
    'num_train_queries': 256,  # anchors for MLP
    'batch_size': 8,
    'epochs': 20,
    'lr': 1e-3,
    'warmup_ratio': 0.1,
    'eval_interval_epochs': 0.5,
}

# Plot settings
config['plot'] = {
    'font_family': 'Times New Roman',
    'colors': {
        'm2': '#1f77b4',            # Static
        'm2_adaptive': '#ff7f0e',   # Task Matrix
        'mlp': '#2ca02c',           # MLP curve
    }
}
