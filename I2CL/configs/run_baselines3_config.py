config = {}

### Main Configuration ###
config['exp_name'] = 'exps/subj_check'  # M2 experiment directory
config['gpus'] = ['0']
config['models'] = ['meta-llama/Llama-2-13b-hf'] #,'meta-llama/Llama-2-7b-hf']  # list of model names
config['datasets'] = ['subj']         #'hate_speech18','dbpedia']
config['shot_per_class'] = 30 # number of shots (NOT per class) qwen 7b dbpedia, agnews
config['bs'] = 8  # batch size

# Experiment settings
config['return_logits'] = True
config['logits_mode'] = 'first'
config['run_num'] = 1
config['seed'] = 42
config['demo_seed'] = 12
config['run_baseline'] = True
config['metric'] = 'acc'  # 'acc', 'macro_f1'
config['load_in_8bit'] = True
config['use_cache'] = False

config['run_i2cl'] = True
config['run_tv'] = True
config['run_fv'] = True
config['run_sv'] = True

# Task vector extraction
config['num_train_queries'] = [256]  # Number of training queries for task vector learning
config['ridge_lambda'] = [5.0] 
config['extraction_batch_size'] = 1
config['inference_batch_size'] = 1

# Which layers to inject (None = all layers)
config['target_layers'] = None  # None means all layers

# Module to use for extraction/injection
config['module'] = ['hidden']  # Use hidden states (full layer output)
config['tok_pos'] = 'last'  # Token position for extraction

# Data settings
config['val_data_num'] = 32
config['test_data_num'] = 500
config['sample_method'] = 'uniform'  # 'random', 'uniform'
config['use_instruction'] = False
config['add_extra_query'] = True
config['example_separator'] = '\n'

# Evaluation settings
config['compute_kl_divergence'] = True
config['save_task_vectors'] = False
config['evaluate_reconstruction'] = False  # Evaluate reconstruction quality on val set

# SV baseline settings
config['question_prompt'] = "Input: {input}\n"
config["eos"] = "\n"
config["proj_tokens"] = "→" 


# I2CL baseline settings
#config['train_cali'] = False
config['init_value'] = [0.1, 1.0]  # linear and constraint: [0.1, 1.0], add: [0.1]

config['layer'] = 'all' # all, early, mid, late
config['tok_pos'] = 'last'  # 'random', 'first', 'last'
config['inject_method'] = 'linear'  # 'linear', 'constraint', 'add'
config['inject_pos'] = 'all'  # 'all', 'first', last', 'random'
config['module'] = ['mlp', 'attn']  # 'mlp', 'attn', 'hidden'
config['gen_cv_method'] = 'context'  # 'context', 'noise'
config['post_fuse_method'] = 'mean'  # 'mean', 'pca'
config['split_demon'] = True  # split demonstraiton into seperate examples
config['gen_example_method'] = 'normal' 

config['add_noise'] = True  # whether add noise
config['noise_scale'] = 0.001  # noise scale
config['epochs'] = 100  # number of epochs
config['optim'] = 'adamW'  # 'adam', 'adamW', 'sgd'
config['grad_bs'] = 2  # batch size for clibration
config['lr'] = 0.01
config['wd'] = 1e-3
config['cali_example_method'] = 'normal' # 'normal', 'random_label'

# function vector baseline settings
config['choose_layer']=False
config['universal_fv']=True
config['n_top_heads']=10
config['n_mean_activations_trials']=20 
config['separators']={"input": "", "output": "", "instructions": ""}
config['prefixes']={"input": "\n", "output": "", "instructions": ""}
config['revision']=None