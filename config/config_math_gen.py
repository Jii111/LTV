"""
Config for math generation datasets: GSM8K, MultiArith.
Use with: run_math_gen.py

python run_math_gen.py --config_path configs/config_math_gen.py
"""

config = {}

config['exp_name'] = 'exps/math_gen'
config['gpus'] = ['0']
config['models'] = ['meta-llama/Llama-3.1-8B-Instruct'] #meta-llama/Llama-3.1-8B-Instruct
config['datasets'] = ['gsm8k']

# 1 seed run as requested
config['run_num'] = 5
config['seed'] = 42

config['num_shot'] = 30         # few-shot demo examples
config['num_train_queries'] = 256 # queries for task vector extraction
config['val_data_num'] = 32
config['test_data_num'] = 100

config['ridge_lambda'] = [5.0] #, 10, 1e3, 1e5, 1e7
config['extraction_batch_size'] = 1
config['bs'] = 1
config['max_new_tokens'] = 256

config['load_in_8bit'] = True
config['do_sample'] = True
