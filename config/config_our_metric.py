config = {}

# Paths (format with {model}, {dataset})
config['model_name'] = 'meta-llama/Llama-3.1-8B'
config['datasets'] = ['sst2']

# Logit directories produced by run_our_ltv.py with save_logits=True
config['result_dir'] = 'results/ltv'

# Output
config['method_name'] = 'LTV'
config['save_dir'] = 'results/ltv_metrics'
config['plot_path'] = 'results/ltv_metrics/d_NTP.png'