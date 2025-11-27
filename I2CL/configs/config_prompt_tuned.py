config = {}

### 주요 config ### # ✅ 수정
config['exp_name'] = 'exps/method1_exp_5.0e-2-2000-bs16'
config['return_logits'] = True
config['logits_mode'] = 'first'

config['gpus'] = ['0']
config['models'] = ['Qwen/Qwen2.5-7B']
#['Qwen/Qwen2.5-32B','Qwen/Qwen2.5-7B','Qwen/Qwen2.5-14B']
# 'meta-llama/Llama-2-7b-hf', 'gpt2-xl', 'EleutherAI/gpt-j-6B', 'meta-llama/Llama-3.1-8B'
config['datasets'] = ['agnews'] 
config['peft_names'] = [
    "Mayfull/qwen-2.5-7b-agnews-5.0e-2-2000-bs16",
    
    

]

config['run_num'] = 10  # number of runs
config['bs'] = 16  # batch size
config['use_cache'] = False

config['return_logits'] = True
config['logits_mode'] = 'first'
config['use_cache'] = False  

config['test_data_num'] = 500  # number of test data
config['sample_method'] = 'uniform'
config['metric'] = 'acc'
config['seed'] = 42 
