config = {}

### Main Configuration ###
config['exp_name'] = 'exps/baseline_zs' 
config['gpus'] = ['0']
config['models'] = ['meta-llama/Llama-3.1-8B','Qwen/Qwen2.5-7B']
# 'Qwen/Qwen2.5-7B', 'Qwen/Qwen2.5-32B', 'Qwen/Qwen2.5-14B', 'meta-llama/Llama-2-7b-hf', 'gpt2-xl', 'EleutherAI/gpt-j-6B'
config['datasets'] = ['hate_speech18'] 
config['shot_per_class'] = 30 # number of shots (NOT per class)
config['bs'] = 4  # batch size

# Experiment settings
config['return_logits'] = True
config['logits_mode'] = 'first'
config['run_num'] = 5
config['seed'] = 42
config['demo_seed'] = 12
config['run_baseline'] = True
config['metric'] = 'acc'  # 'acc', 'macro_f1'
config['load_in_8bit'] = True
config['use_cache'] = False

# Task vector extraction
config['num_train_queries'] = [32, 64, 128, 256]  # Number of training queries for task vector learning
config['ridge_lambda'] = [0.1, 1.0, 5.0, 10.0] 
config['extraction_batch_size'] = 1
config['inference_batch_size'] = 1

# Which layers to inject (None = all layers)
config['target_layers'] = None  # None means all layers

# Module to use for extraction/injection
config['module'] = ['hidden']  # Use hidden states (full layer output)
config['tok_pos'] = 'last'  # Token position for extraction

# Data settings
config['val_data_num'] = 32
config['test_data_num'] = 8000
config['sample_method'] = 'uniform'  # 'random', 'uniform'
config['use_instruction'] = False
config['add_extra_query'] = True
config['example_separator'] = '\n'

# Evaluation settings
config['compute_kl_divergence'] = True
config['save_task_vectors'] = False
config['evaluate_reconstruction'] = False  # Evaluate reconstruction quality on val set

# SV baseline settings
config['question_prompt'] = "{input}"
config["eos"] = "\n\n"
config["proj_tokens"] = "→" #\n 같은디?

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
config['edit_layer']=-2 
config['n_top_heads']=10 
config['n_mean_activations_trials']=20 
config['separators']={"input": "Q:", "output": "A:", "instructions": ""}
config['prefixes']={"input": "\n", "output": "\n\n", "instructions": ""}
config['revision']=None