config = {}

### Main Configuration ###
config['exp_name'] = 'exps/cross_model'
config['gpus'] = ['0']
config['large_model'] = 'Qwen/Qwen2.5-72B'
config['small_model'] = 'Qwen/Qwen2.5-7B'
config['datasets'] = ['trec','sst2','sst5','mr','subj','agnews','hate_speech18','dbpedia']
config['num_shot'] = 30 # number of shots, which ensures label balance by including the maximum possible examples per label
config['bs'] = 2  # batch size

# Experiment settings
config['run_num'] = 5
config['seed'] = 42
config['demo_seed'] = 12
config['load_in_8bit'] = False
config['load_in_4bit'] = True
config['use_cache'] = False

# Data settings
config['sample_method'] = 'uniform'
config['add_extra_query'] = True
config['example_separator'] = '\n'

# Evaluation settings
config['test_data_num'] = 500
config['metric'] = 'acc' # 'acc', 'macro_f1'
config['return_logits'] = True
config['compute_d_NTP'] = True
config['compare_lm_head'] = True # LM head label vector comparison between large and small model

# LTV extraction
config['num_train_queries'] = [256]
config['ridge_lambda'] = [5.0]
config['extraction_batch_size'] = 1