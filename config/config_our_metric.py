config = {}

# Paths (format with {model}, {dataset})
config['model_name'] = 'meta-llama/Llama-3.1-8B'
config['datasets'] = ['sst2','sst5','mr','subj','trec','hate_speech18','agnews','dbpedia']

# Logit directories produced by run_our_ltv.py with save_logits=True
config['icl_logit_dir_tpl'] = 'results/ltv/{model}/{dataset}/logits_icl'
config['tv_logit_dir_tpl'] = 'results/ltv/{model}/{dataset}/logits_tv'

# Output
config['method_name'] = 'Ours'
config['save_dir'] = 'results/ltv_metrics'
config['plot_path'] = 'results/ltv_metrics/d_NTP.png'
config['dataset_order'] = ['agnews','dbpedia','hate_speech18','mr','sst2','sst5','subj','trec']
