config = {}

### Main Configuration ###
config['exp_name'] = 'exps/mainexp2'  
config['models'] = ['Qwen/Qwen2.5-14B']
config['datasets'] = ['dbpedia'] 
config['shot_per_class'] = 30 # number of shots (NOT per class)
config['bs'] = 1  # batch size

config['run_num'] = 5
config['seed'] = 42

# M2-Adaptive analysis figures
config['save_m2a_figures'] = True
config['plot_permuted_control'] = True
