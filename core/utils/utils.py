"""Utilities for loading models, configs, and analysis helpers."""

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
import wrapper
import our_datasets as md
from transformers import BitsAndBytesConfig

from typing import Any, Dict, List, Optional, Tuple, Union
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, PreTrainedModel, PreTrainedTokenizerBase, PretrainedConfig

def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and torch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def set_device(gpu_id: Union[int, str]) -> torch.device:
    """Select and set a CUDA device if available."""
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    return device


def init_exp_path(args: Any, exp_name: str, separate_dataset: bool = True) -> None:
    """Create experiment output directory and persist config/args."""
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
    # save args as txt file
    with open(f'{save_dir}/args.txt', 'w') as f:
        for key, value in vars(args).items():
            f.write(f'{key}: {value}\n')


def load_model_tokenizer(
    model_name: str,
    device: torch.device,
    output_hidden_states: bool = True,
    load_in_8bit: bool = False,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase, PretrainedConfig]:
    """Load a model/tokenizer pair with optional quantization."""

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
                                                 load_in_8bit=load_in_8bit, 
                                                 torch_dtype=torch.float16) 

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

def get_model_wrapper(
    model_name: str,
    model: nn.Module,
    tokenizer: Any,
    model_config: Any,
    device: torch.device,
) -> "wrapper.ModelWrapper":
    """Create the correct wrapper class for a model family."""
    
    if 'llama' in model_name:
        model_wrapper = wrapper.LlamaWrapper(model, tokenizer, model_config, device)
    elif 'Qwen' in model_name:
        model_wrapper = wrapper.Qwen3Wrapper(model, tokenizer, model_config, device)
    else:
        raise ValueError("only support llama or gpt!")
    return model_wrapper


def load_config(file_path: str) -> Optional[Dict[str, Any]]:
    """Import a config module and return its `config` dict."""
    
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

def get_shot_num(dataset: Any, shot_per_class: int, shot_num: int = 5) -> int:
    """Compute total shots based on dataset class count."""
    
    if hasattr(dataset, 'class_num') and dataset.class_num is not None:
        shot_num = dataset.class_num * shot_per_class
    else:
        shot_num = shot_num
    # if shot_num < 0, then use all data
    if shot_num < 0:
        shot_num = -1
    return shot_num


def build_train_queries(
    train_dataset: Any,
    max_queries: int,
    exclude_indices: Optional[set[int]] = None,
) -> Tuple[List[str], List[int]]:
    """Sample train queries excluding demonstration indices."""
    total = len(train_dataset.all_data)
    if total == 0:
        return [], []
    if exclude_indices is None:
        exclude_indices = set()
    candidate_indices = [idx for idx in range(total) if idx not in exclude_indices]
    if len(candidate_indices) == 0:
        return [], []
    sample_size = min(max_queries, len(candidate_indices))
    query_indices = random.sample(candidate_indices, sample_size)
    queries = []
    for idx in query_indices:
        ques_str, _, _ = train_dataset.apply_template(train_dataset.all_data[idx])
        queries.append(ques_str)
    return queries, query_indices


def get_acc(entry: Any) -> float:
    """Normalize accuracy extraction across result formats."""
    if isinstance(entry, tuple):
        data = entry[0]
        if isinstance(data, dict):
            return data.get("acc", -1)

    if isinstance(entry, dict):
        return entry.get("acc", -1)

    return -1



def last_one_indices(tensor: torch.Tensor) -> torch.Tensor:
    """
    Finds the index of the last 1 in each row of a 2D tensor.

    Args:
      tensor (torch.Tensor): A 2D tensor of size (N, M) containing only 0 and 1 entries.

    Returns:
      torch.Tensor: A tensor of size N containing the index of the last 1 in each row.
                    If a row contains only 0s, the index will be set to -1 (or a sentinel value of your choice).
    """
    reversed_tensor = torch.flip(tensor, [1])
    is_all_zero = reversed_tensor.sum(dim=1) == 0
    indices = reversed_tensor.argmax(dim=1) 
    indices = tensor.size(1) - 1 - indices
    indices[is_all_zero] = -1  # Set to -1 to indicate no '1' found in these rows
    return indices
    
class ContextSolver:
    """Utility for extracting masks and contexts from demonstrations."""
    def __init__(self, task_name: str, tokenizer: Optional[Any] = None) -> None:
        self.task_name = task_name
        self.tokenizer = tokenizer
        self.task_dataset = md.get_dataset(task_name, split='train', max_data_num=10)
        self.format_s = self.task_dataset.get_dmonstration_template()['input']
        self.parse_format_s()

    def parse_format_s(self) -> None:
        """Parse dataset template prefixes for masking."""
        self.X_prefix = self.format_s.split('\n')[0].split(':')[0] + ':'
        self.Y_prefix = self.format_s.split('\n')[1].split(':')[0] + ':'

    def get_empty_demo_context(self, context: str, only_demo_part: bool = True) -> str:
        """Strip demonstration content down to label prefixes."""
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

    def get_mask_strings_and_match_before(
        self,
        context: str,
        input_ids: torch.Tensor,
        tokenizer: Optional[Any] = None,
    ) -> Tuple[List[str], Optional[int]]:
        """Build mask strings and a cutoff position for matching."""
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

    def get_mask(self, input_ids: Union[List[int], torch.Tensor], tokenizer: Optional[Any] = None) -> torch.Tensor:
        """Compute a boolean mask for demo tokens in the input."""
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
    """Find token string matches inside tokenized tensors."""
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def find_tensor_in_tensor(
        self,
        a_tensor: Union[torch.Tensor, List[int]],
        b_tensor: torch.Tensor,
        return_mask: bool = True,
        match_before: Optional[int] = None,
    ) -> Union[torch.Tensor, torch.Tensor]:
        """Find a sub-tensor in a tensor and optionally return a mask."""
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

    def find_str_in_tensor(
        self,
        s: str,
        t: torch.Tensor,
        return_mask: bool = True,
        match_before: Optional[int] = None,
    ) -> torch.Tensor:
        """Tokenize a string and find it inside a tensor."""
        s_tokens = self.tokenizer.encode(s, add_special_tokens=False)
        s_tensor = torch.LongTensor(s_tokens)
        return self.find_tensor_in_tensor(s_tensor, t, return_mask=return_mask,
                                          match_before=match_before)

    def get_strs_mask_in_tensor(
        self,
        list_s: List[str],
        t: torch.Tensor,
        match_before: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute a combined mask for multiple strings."""
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

def compute_kl_divergence(logits_p: torch.Tensor, logits_q: torch.Tensor, is_qwen: bool = False) -> float:
    """Compute per-sample KL divergence between two logit tensors."""
    probs_p = F.softmax(logits_p, dim=-1)
    probs_q = F.softmax(logits_q, dim=-1)

    kl_elem = probs_p * (torch.log(probs_p) - torch.log(probs_q))
    kl = kl_elem.sum(dim=-1).float()  # (num_test_query,)

    valid = torch.isfinite(kl)
    if valid.any():
        kl_mean = kl[valid].mean()
    else:
        kl_mean = torch.tensor(float("nan"), device=kl.device)

    q1, q2, q3 = torch.quantile(
        kl[valid],
        torch.tensor([0.25, 0.5, 0.75], device=kl.device)
    )

    return kl_mean.item()
    
def nested_set(d: Dict[str, Any], keys: List[str], value: Any) -> None:
    """Set a nested dict value given a key path."""
    for k in keys[:-1]:
        if k not in d:
            d[k] = {}
        d = d[k]
    d[keys[-1]] = value
