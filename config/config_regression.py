config = {}

### Main Configuration ###
config['exp_name'] = 'exps/regression'
config['model']    = 'meta-llama/Llama-3.1-8B' 
config['gpu']      = '0'                  
config['datasets'] = ['linear_regression', 'relu_regression']
config['d'] = 6
config['num_shot'] = 30
config['bs'] = 8 # batch size

# Run LTV
config['run_baseline'] = True # run zero-shot and few-shot baselines
config['run_ltv']  = True

# Experiment settings
config['run_num'] = 10
config['seed'] = 47
config['load_in_8bit'] = False

# Evaluation settings
config['test_data_num'] = 500

# LTV extraction
config['num_train_queries'] = 256
config['ridge_lambda'] = [5]