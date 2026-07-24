"""
Smoke-test config for run/run_translation_gen.py — tiny sizes to check the
pipeline runs end-to-end and to time one run before scaling up to
config_translation_gen.py's full sizes.
"""

config = {}

config['exp_name'] = 'exps/translation_gen_smoke'
config['gpus'] = ['0']
config['models'] = ['meta-llama/Llama-3.1-8B','meta-llama/Llama-3.1-8B-Instruct']
config['datasets'] = ['wmt14_enfr']

config['run_num'] = 1
config['seed'] = 42

config['num_shot'] = 4
config['num_train_queries'] = [8]   # sweep, to smoke-test the multi-value path too
config['val_data_num'] = 8
config['test_data_num'] = 10

config['bs'] = 1
config['max_new_tokens'] = 48
config['stop_pattern'] = r'\nEnglish:|\n\n'

config['early_range'] = (1, 4)
config['ridge_lambdas'] = [1.0, 5.0]
config['n_cv_folds'] = 2
config['repeat_ngram'] = 3
config['repeat_min_repeats'] = 4

config['load_in_8bit'] = True
config['do_sample'] = False
config['temperature'] = 1.0
config['top_p'] = 1.0
