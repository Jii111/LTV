import sys
sys.path.append('../')
import my_datasets as md

config = {}

### 주요 config ###
config['exp_name'] = 'exps/taskvector_shot_1113_2'
config['gpus'] = ['0']
config['models'] = ['Qwen/Qwen2.5-7B']  # 'Qwen/Qwen2.5-7B', 'Qwen/Qwen2.5-32B', 'Qwen/Qwen2.5-14B', 'meta-llama/Llama-2-7b-hf', 'gpt2-xl', 'EleutherAI/gpt-j-6B'
config['datasets'] = ['sst5', 'trec']  # 'agnews', 'dbpedia', ' hate_speech18', 'mr', 'sst2', 'sst5', 'trec'
config['shot_per_class'] = 30 # number of shots. NOT per class
config['bs'] = 8  # batch size
###

config['return_logits'] = True
config['logits_mode'] = 'first' 
config['run_num'] = 10

config['seed'] = 21
config['demo_seed'] = 6
config['run_baseline'] = True
config['metric'] = 'acc'  # 'acc', 'macro_f1'
config['load_in_8bit'] = True
config['use_cache'] = False

# context vector
config['layer'] = 'all' # all, late, early, mid
config['tok_pos'] = 'last'
config['module'] = ['hidden']  # 'mlp', 'attn', 'hidden'
config['gen_cv_method'] = 'context'  # 'context', 'noise'
config['post_fuse_method'] = 'mean'  # 'mean', 'pca'
config['split_demon'] = False  # split demonstraiton into seperate examples

# data

config['val_data_num'] = 32
config['test_data_num'] = 500
config['sample_method'] = 'uniform'  # 'random', 'uniform'
config['use_instruction'] = False
config['add_extra_query'] = True
config['example_separator'] = '\n'