import os
import sys
import json
import random
import functools
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import matplotlib.pyplot as plt
import yaml
import wrapper
import my_datasets as md
from peft import PromptTuningConfig, PeftModel
from transformers import BitsAndBytesConfig
import re

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def set_device(gpu_id):
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    return device


def init_exp_path(args, exp_name, separate_dataset=True):
    if separate_dataset:
        save_dir = os.path.join(exp_name, args.model_name, args.dataset_name)
    else:
        save_dir = os.path.join(exp_name, args.model_name)
    args.save_dir = save_dir
    if os.path.exists(save_dir) and 'debug' not in exp_name:
        raise ValueError(f"Experiment {exp_name} already exists! please delete it or change the name!")
    os.makedirs(save_dir, exist_ok=True)
    # save config_dict
    with open(f'{save_dir}/config.json', 'w') as f:
        json.dump(args.config, f, indent=4)
    # save args as txt file, I want to make it beautiful
    with open(f'{save_dir}/args.txt', 'w') as f:
        for key, value in vars(args).items():
            f.write(f'{key}: {value}\n')


def load_model_tokenizer(model_name, device, output_hidden_states=True, load_in_8bit=False):
    # load tokenizer and model
    if 'Llama-2-7b' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name,use_faset=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    if 'Qwen' in model_name:
        if load_in_8bit:
            quant_cfg = BitsAndBytesConfig(load_in_8bit=load_in_8bit) if load_in_8bit else None
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.bfloat16,
                quantization_config=quant_cfg,
                device_map="auto", low_cpu_mem_usage=True)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, load_in_8bit=load_in_8bit,
                dtype=torch.bfloat16)
        
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 output_hidden_states=output_hidden_states,
                                                 load_in_8bit=load_in_8bit, #oad_in_8bit,
                                                 torch_dtype=torch.float16) # 32 -> 16 수정

    print("loaded_in_8bit:", getattr(model, "is_loaded_in_8bit", False))
    # Ensure hidden states are available when requested
    if hasattr(model, "config"):
        model.config.output_hidden_states = output_hidden_states


    if not load_in_8bit:
        model = model.to(device)
        
    config = AutoConfig.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    tokenizer.padding_side = 'left'
    
    if "qwen" in model_name.lower() or "llama" in model_name.lower():
        fv_config = {
            "n_heads": model.config.num_attention_heads,
            "n_layers": model.config.num_hidden_layers,
            "resid_dim": model.config.hidden_size,
            "name_or_path": model.config._name_or_path,
            "attn_hook_names": [
                f"model.layers.{layer}.self_attn.o_proj"
                for layer in range(model.config.num_hidden_layers)
            ],
            "layer_hook_names": [
                f"model.layers.{layer}"
                for layer in range(model.config.num_hidden_layers)
            ],
            "prepend_bos": True
        }
    else:
        fv_config = {
        "n_heads": model.config.num_attention_heads,
        "n_layers": model.config.num_hidden_layers,
        "resid_dim": model.config.hidden_size,
        "name_or_path": model.config._name_or_path,
        "attn_hook_names": [],
        "layer_hook_names": [],
        "prepend_bos": False
    }
    config.fv = fv_config
    
    return model, tokenizer, config


def load_custom_model_tokenizer(model_name, device, output_hidden_states=True, load_in_8bit=False, method='M3'):
    """
    Legacy entrypoint for custom M2/M3 models.
    Now falls back to the standard loader since we no longer rely on custom deltas.
    """
    return load_model_tokenizer(
        model_name=model_name,
        device=device,
        output_hidden_states=output_hidden_states,
        load_in_8bit=load_in_8bit,
    )


def load_peft_model_tokenizer(model_name, peft_name, device, output_hidden_states=True, load_in_8bit=False):
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # load base model (without quantization)
    if 'Qwen' in model_name:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            output_hidden_states=output_hidden_states,
            torch_dtype=torch.bfloat16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            output_hidden_states=output_hidden_states,
            load_in_8bit=load_in_8bit,
            torch_dtype=torch.float16
        )

    # Load PEFT adapter (for prompt tuning, do NOT merge)
    if peft_name:
        print(f"Loading PEFT adapter from: {peft_name}")
        model = PeftModel.from_pretrained(
            model,
            peft_name,
            device_map="auto",
            is_trainable=False  # inference only
        )
        model.eval()  # evaluation mode
        print("PEFT adapter loaded successfully (soft prompts will be applied during inference)")
        
    if not load_in_8bit:
        model = model.to(device)
    config = AutoConfig.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    
    if 'llama' in model_name:
        config.n_heads = model.config.num_attention_heads
        config.n_layers = model.config.num_hidden_layers
        config.resid_dim = model.config.hidden_size
        config.name_or_path = model.config._name_or_path
        config.attn_hook_names = [f"model.layers.{layer}.self_attn.o_proj" for layer in range(config.n_layers)]
        config.layer_hook_names = [f"model.layers.{layer}" for layer in range(config.n_layers)]
        config.prepend_bos = True
        
    elif 'Qwen' in model_name: 
        tokenizer.add_special_tokens({
            "bos_token": "<|beginoftext|>",
            "eos_token": "<|endoftext|>"
        })
        tokenizer.add_bos_token = True
        tokenizer.add_eos_token = True
        tokenizer.clean_up_tokenization_spaces = False
        
        config.n_heads = model.config.num_attention_heads
        config.n_layers = model.config.num_hidden_layers
        config.resid_dim = model.config.hidden_size
        config.name_or_path = model.config._name_or_path

        config.attn_hook_names = [f"model.layers.{layer}.self_attn.o_proj" for layer in range(config.n_layers)]
        config.layer_hook_names = [f"model.layers.{layer}" for layer in range(config.n_layers)]

        config.prepend_bos = True
        config.use_fast_tokenizer = False
        config.add_special_tokens = False
        config.return_dict = True
    
    return model, tokenizer, config


def get_model_wrapper(model_name, model, tokenizer, model_config, device):
    if 'llama' in model_name:
        model_wrapper = wrapper.LlamaWrapper(model, tokenizer, model_config, device)
    elif 'gpt' in model_name:
        model_wrapper = wrapper.GPTWrapper(model, tokenizer, model_config, device)
    elif 'GPT' in model_name:
        model_wrapper = wrapper.GPTWrapper(model, tokenizer, model_config, device)
    elif 'Qwen' in model_name:
        model_wrapper = wrapper.Qwen3Wrapper(model, tokenizer, model_config, device)
    else:
        raise ValueError("only support llama or gpt!")
    return model_wrapper


def load_config(file_path):
    if not file_path:
        raise ValueError("No file path provided")
    file_dir = os.path.dirname(file_path)
    if file_dir not in sys.path:
        sys.path.append(file_dir)
    file_name = os.path.basename(file_path)
    module_name = os.path.splitext(file_name)[0]
    module = __import__(module_name)
    try:
        my_variable = getattr(module, 'config')
        print(my_variable)
        return my_variable
    except AttributeError:
        print(f"The module does not have a variable named 'config'")


def load_yaml(yaml_path):
    """
    Load YAML configuration file and return as dictionary.

    Args:
        yaml_path: Path to the YAML configuration file

    Returns:
        dict: Configuration dictionary
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML config file not found: {yaml_path}")

    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def get_shot_num(dataset, shot_per_class, shot_num=5):
    if hasattr(dataset, 'class_num') and dataset.class_num is not None:
        shot_num = dataset.class_num * shot_per_class
    else:
        shot_num = shot_num
    # if shot_num < 0, then use all data
    if shot_num < 0:
        shot_num = -1
    return shot_num


def first_one_indices(tensor):
    """
    Finds the index of the first 1 in each row of a 2D tensor.

    Args:
      tensor (torch.Tensor): A 2D tensor of size (N, M) containing only 0 and 1 entries.

    Returns:
      torch.Tensor: A tensor of size N containing the index of the first 1 in each row.
                    If a row contains only 0s, the index will be set to -1 (or a sentinel value of your choice).
    """
    # Check for rows containing only zeros.
    is_all_zero = tensor.sum(dim=1) == 0
    # Get the index of the first occurrence of the maximum value (1) along each row.
    indices = tensor.argmax(dim=1)
    # Handle rows with all zeros.
    indices[is_all_zero] = -1  # Set to -1 to indicate no '1' found in these rows
    return indices


def last_one_indices(tensor):
    """
    Finds the index of the last 1 in each row of a 2D tensor.

    Args:
      tensor (torch.Tensor): A 2D tensor of size (N, M) containing only 0 and 1 entries.

    Returns:
      torch.Tensor: A tensor of size N containing the index of the last 1 in each row.
                    If a row contains only 0s, the index will be set to -1 (or a sentinel value of your choice).
    """
    # Reverse each row to find the last occurrence of 1 (which becomes the first in the reversed row)
    reversed_tensor = torch.flip(tensor, [1])
    # Check for rows containing only zeros in the reversed tensor
    is_all_zero = reversed_tensor.sum(dim=1) == 0
    # Get the index of the first occurrence of the maximum value (1) along each row in the reversed tensor
    indices = reversed_tensor.argmax(dim=1) 
    # Adjust the indices for the original order of each row
    indices = tensor.size(1) - 1 - indices
    # Handle rows with all zeros
    indices[is_all_zero] = -1  # Set to -1 to indicate no '1' found in these rows
    return indices


def plot_loss_curve(loss_list, save_path):
    plt.plot(loss_list)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(save_path)
    plt.close()


def svd_flip(u, v):
    # columns of u, rows of v
    max_abs_cols = torch.argmax(torch.abs(u), 0)
    i = torch.arange(u.shape[1]).to(u.device)
    signs = torch.sign(u[max_abs_cols, i])
    u *= signs
    v *= signs.view(-1, 1)
    return u, v


class PCA(nn.Module):
    def __init__(self, n_components):
        super().__init__()
        self.n_components = n_components

    @torch.no_grad()
    def fit(self, X):
        n, d = X.size()
        if self.n_components is not None:
            d = min(self.n_components, d)
        self.register_buffer("mean_", X.mean(0, keepdim=True))
        Z = X - self.mean_ # center
        U, S, Vh = torch.linalg.svd(Z, full_matrices=False)
        Vt = Vh
        U, Vt = svd_flip(U, Vt)
        self.register_buffer("components_", Vt[:d])
        return self

    def forward(self, X):
        return self.transform(X)

    def transform(self, X):
        assert hasattr(self, "components_"), "PCA must be fit before use."
        return torch.matmul(X - self.mean_, self.components_.t())

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, Y):
        assert hasattr(self, "components_"), "PCA must be fit before use."
        return torch.matmul(Y, self.components_) + self.mean_
    
class ContextSolver:
    def __init__(self, task_name, tokenizer=None):
        # assert task_name in ['sst2', 'trec', 'agnews', 'emo']
        self.task_name = task_name
        self.tokenizer = tokenizer
        self.task_dataset = md.get_dataset(task_name, split='train', max_data_num=10)
        self.format_s = self.task_dataset.get_dmonstration_template()['input']
        self.parse_format_s()

    def parse_format_s(self):
        self.X_prefix = self.format_s.split('\n')[0].split(':')[0] + ':'
        self.Y_prefix = self.format_s.split('\n')[1].split(':')[0] + ':'

    def get_empty_demo_context(self, context: str, only_demo_part=True):
        context = context.split('\n')
        for i, line in enumerate(context[:-2]):
            if self.X_prefix in line:
                line = self.X_prefix
            elif self.Y_prefix in line:
                line = line
            else:
                raise warnings.warn('Global prefix or other str exists!')
            context[i] = line
        if only_demo_part:
            context = context[:-2]
        context = '\n'.join(context)
        return context

    def get_mask_strings_and_match_before(self, context, input_ids, tokenizer=None):
        if tokenizer is None:
            tokenizer = self.tokenizer
        print('debug tokenizer name :', tokenizer.__class__.__name__)
        if 'Llama' in tokenizer.__class__.__name__:
            sap_token = tokenizer.encode('\n', add_special_tokens=False)[1]
            poss = torch.where(input_ids == sap_token)[0]
        else:
            sap_token = tokenizer.encode('\n', add_special_tokens=False)[0]
            poss = torch.where(input_ids == sap_token)[0]
        print('debug sap_token:', sap_token)
        print('debug poss:', poss)
        if len(poss) >= 2:
            match_before = poss[-2] + 1
        else:
            match_before = None

        list_s = []
        list_s.append(self.X_prefix)
        list_s.append('\n' + self.X_prefix)
        context = context.split('\n')
        for i, line in enumerate(context[:-2]):
            if self.X_prefix in line:
                pass
            elif self.Y_prefix in line:
                list_s.append('\n' + line)
                list_s.append('\n' + line + '\n')
            else:
                raise warnings.warn('Global prefix or other str exists!')
        return list_s, match_before

    def get_mask(self, input_ids, tokenizer=None):
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids)
        if len(input_ids.shape) == 2:
            assert input_ids.shape[0] == 1
            input_ids = input_ids[0]
        if tokenizer is None:
            tokenizer = self.tokenizer
        context = tokenizer.decode(input_ids)
        list_s, match_before = self.get_mask_strings_and_match_before(context, input_ids=input_ids,
                                                                      tokenizer=tokenizer)
        print('debug context:', context)
        print('debug list_s:', list_s)
        print('debug match_before:', match_before)
        tensor_str_finder = TensorStrFinder(tokenizer=tokenizer)
        mask = tensor_str_finder.get_strs_mask_in_tensor(list_s=list_s, t=input_ids,
                                                         match_before=match_before)
        return mask
    
class TensorStrFinder:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def find_tensor_in_tensor(self, a_tensor: Union[torch.Tensor, list], b_tensor: torch.Tensor,
                              return_mask=True, match_before: Optional[int] = None):
        if len(b_tensor.shape) == 2:
            assert b_tensor.shape[0] == 1
            b_tensor = b_tensor[0]
        if isinstance(a_tensor, list):
            a_tensor = torch.tensor(a_tensor)
        if a_tensor.device != b_tensor.device:
            a_tensor = a_tensor.to(b_tensor.device)

        window_size = len(a_tensor)
        b_windows = b_tensor.unfold(0, window_size, 1)

        matches = torch.all(b_windows == a_tensor, dim=1)

        positions = torch.nonzero(matches, as_tuple=True)[0]

        if return_mask:
            mask = torch.zeros_like(b_tensor, dtype=torch.bool)
            for pos in positions:
                if match_before is None or pos + window_size <= match_before:
                    mask[pos:pos + window_size] = True
            return mask

        return positions

    def find_str_in_tensor(self, s: str, t: torch.Tensor, return_mask=True, match_before=None):
        s_tokens = self.tokenizer.encode(s, add_special_tokens=False)
        s_tensor = torch.LongTensor(s_tokens)
        return self.find_tensor_in_tensor(s_tensor, t, return_mask=return_mask,
                                          match_before=match_before)

    def get_strs_mask_in_tensor(self, list_s: List[str], t: torch.Tensor, match_before=None):
        list_s_tokens = [self.tokenizer.encode(s, add_special_tokens=False) for s in list_s]
        if 'Llama' in self.tokenizer.__class__.__name__:
            list_s_tokens = [s_tokens[1:] if s_tokens[0] == 29871 else s_tokens for s_tokens in list_s_tokens]
        list_s_tensor = [torch.LongTensor(s_tokens) for s_tokens in list_s_tokens]
        print('debug list_s_tensor:', list_s_tensor)
        mask_tensor_list = [
            self.find_tensor_in_tensor(s_tensor, t, return_mask=True, match_before=match_before) for
            s_tensor in list_s_tensor]
        mask_tensor = functools.reduce(torch.logical_or, mask_tensor_list)
        return mask_tensor

def compute_kl_divergence(logits_p, logits_q, is_qwen=False):
      probs_p = F.softmax(logits_p, dim=-1)
      probs_q = F.softmax(logits_q, dim=-1)

      print()
      print("few logits mean/std:", logits_p.mean().item(), logits_q.std().item())
      print("method logits mean/std:", logits_p.mean().item(), logits_q.std().item())

      kl_elem = probs_p * (torch.log(probs_p) - torch.log(probs_q))
      kl = kl_elem.sum(dim=-1).float()  # (num_test_query,)

      print()
      print("[DEBUG] ALL KL values", kl)
      print()

      valid = torch.isfinite(kl)
      if valid.any():
          kl_mean = kl[valid].mean()
      else:
          kl_mean = torch.tensor(float("nan"), device=kl.device)

      print("====> KL : ", kl_mean.item())

      q1, q2, q3 = torch.quantile(
          kl[valid],
          torch.tensor([0.25, 0.5, 0.75], device=kl.device)
      )

      print("[DEBUG] KL stats")
      print("  min :", kl[valid].min().item())
      print("  Q1  :", q1.item())
      print("  Q2  :", q2.item())
      print("  Q3  :", q3.item())
      print("  max :", kl[valid].max().item())

      return kl_mean.item()
    
def compute_kl_divergence1(logits_p, logits_q, is_qwen=False):
      probs_p = F.softmax(logits_p, dim=-1)
      probs_q = F.softmax(logits_q, dim=-1)

      print()
      print("few logits mean/std:", logits_p.mean().item(), logits_q.std().item())
      print("method logits mean/std:", logits_p.mean().item(), logits_q.std().item())

      kl_elem = probs_p * (torch.log(probs_p) - torch.log(probs_q))

      # inf/nan 제외용 마스크
      valid = torch.isfinite(kl_elem)

      # 유효한 값만 합산 (invalid는 0으로 처리)
      kl_sum = torch.where(valid, kl_elem, torch.zeros_like(kl_elem)).sum(dim=-1).float()
      valid_count = valid.sum(dim=-1).clamp_min(1).float()

      kl_per_sample = kl_sum / valid_count

      # 샘플 단위에서도 inf/nan 제외
      valid_sample = torch.isfinite(kl_per_sample) & (valid_count > 0)
      kl_mean = kl_per_sample[valid_sample].mean()

      print()
      print("[DEBUG] ALL KL values", kl_per_sample)
      print()
      print("====> KL : ", kl_mean.item())

      q1, q2, q3 = torch.quantile(
          kl_per_sample[valid_sample],
          torch.tensor([0.25, 0.5, 0.75], device=kl_per_sample.device)
      )

      print("[DEBUG] KL stats")
      print("  min :", kl_per_sample[valid_sample].min().item())
      print("  Q1  :", q1.item())
      print("  Q2  :", q2.item())
      print("  Q3  :", q3.item())
      print("  max :", kl_per_sample[valid_sample].max().item())

      return kl_mean.item()

def compute_kl_divergence(logits_p, logits_q, is_qwen=False):
    probs_p = F.softmax(logits_p, dim=-1)
    probs_q = F.softmax(logits_q, dim=-1)

    support_mismatch = (probs_p > 0) & (probs_q == 0)
    mismatch_count = support_mismatch.sum().item()
    
    print()
    print("few logits mean/std:", logits_p.mean().item(), logits_q.std().item())
    print("method logits mean/std:", logits_p.mean().item(), logits_q.std().item())

    if mismatch_count > 0:
        bad_ids = torch.where(support_mismatch)[-1]
        print("[DEBUG] support mismatch count:", mismatch_count)
        print("[DEBUG] bad ids:", bad_ids.tolist())
        idx = torch.where(support_mismatch)
        test_idx = idx[0]
        label_space_index = idx[1]

        bad_logits_q = logits_q[test_idx, label_space_index]
        bad_logits_p = logits_p[test_idx, label_space_index]    
        
    kl = probs_p * (torch.log(probs_p) - torch.log(probs_q))
    kl[support_mismatch] = 0.0   # 문제되는 항만 제거
    kl = kl.sum(dim=-1).float()
    
    
    print()
    print("[DEBUG] ALL KL values",kl)
    print()
    print("====> KL : ", kl.mean().item())
    
    q1, q2, q3 = torch.quantile(
    kl,
    torch.tensor([0.25, 0.5, 0.75], device=kl.device)
)

    print("[DEBUG] KL stats")
    print("  min :", kl.min().item())
    print("  Q1  :", q1.item())
    print("  Q2  :", q2.item()) 
    print("  Q3  :", q3.item())
    print("  max :", kl.max().item())

    if is_qwen == True:
        return kl.float().mean().item()
    else:
        return kl.mean().item()

def plot_kl_hist(kl_values, mean_kl, save_path):
    """KL divergence histogram plot"""
    plt.figure(figsize=(8,5))
    plt.hist(kl_values, bins=30, color='steelblue', alpha=0.7)
    plt.axvline(mean_kl, color='red', linestyle='--', label=f"Mean KL = {mean_kl:.4f}")
    plt.title("KL Divergence Distribution (ICL vs I2CL)")
    plt.xlabel("KL divergence per sample")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    
def nested_set(d, keys, value):
    for k in keys[:-1]:
        if k not in d:
            d[k] = {}
        d = d[k]
    d[keys[-1]] = value
    
def convert_to_svdataset1(split_demon, train_dataset, demon_indices, val_dataset, test_dataset, tokenizer, run_id, args):
      def split_query_and_label_prompt(input_str):
          # 마지막 라벨 줄(예: "Type:" / "Label:" / "Sentiment:") 찾기
          m = re.search(r"\n([A-Za-z _-]+:)\s*$", input_str)
          if m:
              label_prompt = m.group(1)
              query_part = input_str[:m.start()].rstrip()
          else:
              label_prompt = "Label:"
              query_part = input_str

          # 항상 공백 포함으로 정규화
          if not label_prompt.endswith(" "):
              label_prompt = label_prompt + " "
          return query_part, label_prompt

      def strip_label_prefix(text, label_prompt):
          # "Type:World" / "Type: World" 둘 다 제거
          if label_prompt:
              prefix_no_space = label_prompt.rstrip()
              if text.startswith(label_prompt):
                  return text[len(label_prompt):].strip()
              if text.startswith(prefix_no_space):
                  return text[len(prefix_no_space):].lstrip()
          # fallback
          if re.match(r"^[A-Za-z _-]+:\s*", text):
              return re.sub(r"^[A-Za-z _-]+:\s*", "", text).strip()
          return text

      def clean_input(raw_list):
          out = []
          label_prompt = None
          for e in raw_list:
              if isinstance(e, dict):
                  inp = e.get("input", "").strip()
                  outp = e.get("output", "").strip()
              elif isinstance(e, str):
                  # 멀티라인이면 전체에서 마지막 라벨 줄 추출
                  lines = [l.rstrip() for l in e.split("\n") if l.strip() != ""]
                  label_idx = None
                  for i in range(len(lines)-1, -1, -1):
                      if re.match(r"^[A-Za-z _-]+:\s*.*$", lines[i]):
                          label_idx = i
                          break
                  if label_idx is None:
                      inp = e.strip()
                      outp = ""
                  else:
                      inp = "\n".join(lines[:label_idx]).strip()
                      outp = lines[label_idx].strip()
              else:
                  continue

              query_part, lp = split_query_and_label_prompt(inp)
              if label_prompt is None:
                  label_prompt = lp

              outp = strip_label_prefix(outp, label_prompt)
              out.append({"input": query_part, "output": outp})
          return out, label_prompt

      _, valid_split_demon = val_dataset.gen_few_shot_demonstration(
          tokenizer=tokenizer,
          shot_num=min(len(val_dataset), args.config['val_data_num']),
          max_demonstration_tok_len=min(args.val_max_token, args.test_max_token),
          add_extra_query=args.config['add_extra_query'],
          example_separator=args.config['example_separator'],
          return_data_index=False,
          seed=args.config['demo_seed'] + run_id,
          index_info=None,
          sv_data=True
      )
      valid_split_demon, label_prompt = clean_input(valid_split_demon)

      _, test_split_demon = test_dataset.gen_few_shot_demonstration(
          tokenizer=tokenizer,
          shot_num=min(len(test_dataset), args.config['test_data_num']),
          max_demonstration_tok_len=min(args.val_max_token, args.test_max_token),
          add_extra_query=args.config['add_extra_query'],
          example_separator=args.config['example_separator'],
          return_data_index=False,
          seed=args.config['demo_seed'] + run_id,
          index_info=None,
          sv_data=True
      )
      test_split_demon, _ = clean_input(test_split_demon)
      test_split_demon = test_split_demon[:-1]

      all_idx = list(range(len(train_dataset.all_data)))
      candidate_idx = [i for i in all_idx if i not in demon_indices]
      if len(candidate_idx) < 10:
          raise ValueError("train_dummy_split_demon 후보 index 부족")
      dummy_idx = random.sample(candidate_idx, 10)

      train_dummy_split_demon = []
      for idx in dummy_idx:
          input_str, ans_str, label = train_dataset.apply_template(train_dataset.all_data[idx])
          query_part, lp = split_query_and_label_prompt(input_str)
          if label_prompt is None:
              label_prompt = lp
          outp = strip_label_prefix(ans_str[label], label_prompt)
          train_dummy_split_demon.append({"input": query_part, "output": outp})

      split_demon, _ = clean_input(split_demon)
      split_demon = split_demon[:-1]

      if label_prompt is None:
          label_prompt = "Label: "

      return split_demon, train_dummy_split_demon, valid_split_demon, test_split_demon, label_prompt
    
def convert_to_svdataset2(split_demon, train_dataset, demon_indices, val_dataset, test_dataset, tokenizer, run_id, args):
      def split_query_and_label_prompt(input_str):
          # 끝줄이 "Label:" / "Sentiment:" 같은 라벨 프롬프트면 분리
          m = re.search(r"\n([A-Za-z _-]+:)\s*$", input_str)
          if m:
              label_prompt = m.group(1)
              query_part = input_str[:m.start()].rstrip()
          else:
              label_prompt = "Label:"
              query_part = input_str
          return query_part, label_prompt

      def strip_label_prefix(text, label_prompt):
          # output이 "Label: company"면 제거
          if label_prompt and text.startswith(label_prompt):
              return text[len(label_prompt):].strip()
          # 혹시 "Label:company"처럼 붙어있을 때도 처리
          if re.match(r"^[A-Za-z _-]+:\s*", text):
              return re.sub(r"^[A-Za-z _-]+:\s*", "", text).strip()
          return text

      def clean_input(raw_list):
          out = []
          label_prompt = None
          for e in raw_list:
              if isinstance(e, dict):
                  inp = e.get("input", "").strip()
                  outp = e.get("output", "").strip()
              elif isinstance(e, str):
                  parts = e.split("\n")
                  inp = parts[0].strip()
                  outp = parts[1].strip() if len(parts) > 1 else ""
              else:
                  continue

              query_part, lp = split_query_and_label_prompt(inp)
              if label_prompt is None:
                  label_prompt = lp

              outp = strip_label_prefix(outp, label_prompt)
              out.append({"input": query_part, "output": outp})
          return out, label_prompt

      _, valid_split_demon = val_dataset.gen_few_shot_demonstration(
          tokenizer=tokenizer,
          shot_num=min(len(val_dataset), args.config['val_data_num']),
          max_demonstration_tok_len=min(args.val_max_token, args.test_max_token),
          add_extra_query=args.config['add_extra_query'],
          example_separator=args.config['example_separator'],
          return_data_index=False,
          seed=args.config['demo_seed'] + run_id,
          index_info=None,
          sv_data=True
      )
      valid_split_demon, label_prompt = clean_input(valid_split_demon)

      _, test_split_demon = test_dataset.gen_few_shot_demonstration(
          tokenizer=tokenizer,
          shot_num=min(len(test_dataset), args.config['test_data_num']),
          max_demonstration_tok_len=min(args.val_max_token, args.test_max_token),
          add_extra_query=args.config['add_extra_query'],
          example_separator=args.config['example_separator'],
          return_data_index=False,
          seed=args.config['demo_seed'] + run_id,
          index_info=None,
          sv_data=True
      )
      test_split_demon, _ = clean_input(test_split_demon)
      test_split_demon = test_split_demon[:-1]

      # dummy train query
      all_idx = list(range(len(train_dataset.all_data)))
      candidate_idx = [i for i in all_idx if i not in demon_indices]
      if len(candidate_idx) < 10:
          raise ValueError("train_dummy_split_demon 후보 index 부족")
      dummy_idx = random.sample(candidate_idx, 10)

      train_dummy_split_demon = []
      for idx in dummy_idx:
          input_str, ans_str, label = train_dataset.apply_template(train_dataset.all_data[idx])
          query_part, lp = split_query_and_label_prompt(input_str)
          if label_prompt is None:
              label_prompt = lp
          outp = strip_label_prefix(ans_str[label], label_prompt)
          train_dummy_split_demon.append({"input": query_part, "output": outp})

      split_demon, _ = clean_input(split_demon)
      split_demon = split_demon[:-1]

      if label_prompt is None:
          label_prompt = "Label:"

      return split_demon, train_dummy_split_demon, valid_split_demon, test_split_demon, label_prompt
  