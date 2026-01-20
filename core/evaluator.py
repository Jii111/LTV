import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import global_vars as gv
import utils
import pandas as pd

from fv_utils.intervention_utils import *
from sv_utils.TVframework import SVEvaluator

class Evaluator(nn.Module):

    def __init__(self, dataset, batch_size):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size

    def evaluate(self, model_wrapper, tokenizer, demonstration='', use_cache=False, return_logits=False, 
                 return_head_outputs=False, fv_vector=None, sv_logit=None, edit_layer=None, model_config=None,
                 return_q_states=False): 

        return self._evaluate_text_classification_batch(
            model_wrapper, tokenizer,
            demonstration, use_cache=use_cache, return_logits=return_logits,
            return_head_outputs=return_head_outputs, fv_vector=fv_vector, sv_logit=sv_logit, edit_layer=edit_layer, model_config=model_config,
            return_q_states=return_q_states
        )  
        
    def _evaluate_text_classification_batch(self, model_wrapper, tokenizer,
                                            demonstration, use_cache=False, return_logits=False, 
                                            fv_vector=None, sv_logit=None, edit_layer=None, model_config=None,
                                            return_head_outputs=False, return_q_states=False):

        model = model_wrapper.model

        # ======================================================
        # 1. Prepare label tokens
        # ======================================================
        all_base_logits = []
        label_info = self.get_label_info(tokenizer)

        # ======================================================
        # 2. Prepare data
        # ======================================================
        all_inputs, all_labels = [], []
        for data in self.dataset.all_data:
            ques_str, _, label = self.dataset.apply_template(data)
            context = ques_str if (use_cache or len(demonstration) == 0) else demonstration + ques_str
            all_inputs.append(context)
            all_labels.append(label)

        use_cache = False

        # ======================================================
        # 4. Evaluation loop
        # ======================================================
        for batch_idx, i in enumerate(range(0, len(all_inputs), self.batch_size)):
            cur_inputs = all_inputs[i:i + self.batch_size]

            input_tok = tokenizer(cur_inputs, return_tensors="pt", padding=True)
            input_ids = input_tok['input_ids'].to(model.device)
            attn_mask = input_tok['attention_mask'].to(model.device)

            pred_loc = utils.last_one_indices(attn_mask).to(model.device)
            gv.ATTN_MASK_START = torch.zeros_like(pred_loc)
            gv.ATTN_MASK_END = pred_loc
                        
            if fv_vector is not None:
                intervention_idx = -1
                intervention_fn = add_function_vector(edit_layer, fv_vector.reshape(1, model_config.fv['resid_dim']), model.device, idx=intervention_idx)
                with TraceDict(model, layers=model_config.fv['layer_hook_names'], edit_output=intervention_fn) as td:    
                    output = model(input_ids=input_ids,attention_mask=attn_mask,
                    use_cache=False,return_head_outputs=return_head_outputs,return_q_states=return_q_states)
                logits = output.logits
                
            elif sv_logit is not None:                
                logits = sv_logit
                        
            else:
                output = model(input_ids=input_ids,attention_mask=attn_mask, 
                    use_cache=False,return_head_outputs=return_head_outputs,return_q_states=return_q_states)
                logits = output.logits

            base_logits = logits[torch.arange(logits.size(0)), pred_loc]
            all_base_logits.append(base_logits.detach().cpu())
            
            #log_probs = F.log_softmax(base_logits, dim=-1) 

            # (A) space-token
            #space_preds = log_probs[:, label_info['space_first']].argmax(dim=-1)
            #space_logits = base_logits[:, label_info['space_first']]

            # (B) semantic first token
            #semantic_preds = log_probs[:, label_info['semantic_first']].argmax(dim=-1)
            #semantic_logits = base_logits[:, label_info['semantic_first']]

            #all_space_preds.extend(space_preds.cpu().tolist())
            #all_semantic_preds.extend(semantic_preds.cpu().tolist())
            #all_space_logits.append(space_logits.detach().cpu())
            #all_semantic_logits.append(semantic_logits.detach().cpu())

            del logits
            torch.cuda.empty_cache()

        print("[DEBUG] : ", torch.cat(all_base_logits, dim=0).shape)
        
        if return_logits:
            #all_pred_logits = torch.cat(final_logits, dim=0)
            metrics, all_pred_logits = self.evaluate_logits(
                all_base_logits, all_labels, tokenizer, model_wrapper.model.config._name_or_path, return_logits=return_logits)
            return metrics, all_pred_logits, all_labels
        else:
            metrics = self.evaluate_logits(all_base_logits, all_labels, tokenizer, model_wrapper.model.config._name_or_path, return_logits=return_logits)
            return metrics
        
    def evaluate_logits(self, base_logits, labels, tokenizer, model_name, return_logits=False):
      label_info = self.get_label_info(tokenizer)

      logits = torch.cat(base_logits, dim=0)    # (N, V)
      log_probs = F.log_softmax(logits, dim=-1)

      space_preds = log_probs[:, label_info["space_first"]].argmax(dim=-1)
      space_logits = logits[:, label_info["space_first"]]

      semantic_preds = log_probs[:, label_info["semantic_first"]].argmax(dim=-1)
      semantic_logits = logits[:, label_info["semantic_first"]]

      # ===== debug/print =====
      print("\n[DEBUG] Space-token prediction count")
      print(pd.Series(space_preds.cpu().tolist()).value_counts().sort_index())
      print("\n[DEBUG] No-space first-token prediction count")
      print(pd.Series(semantic_preds.cpu().tolist()).value_counts().sort_index())
      print("\n[DEBUG] Ground-truth label count")
      print(pd.Series(labels).value_counts().sort_index())

      acc_space = (space_preds.cpu() == torch.tensor(labels)).float().mean().item()
      acc_first = (semantic_preds.cpu() == torch.tensor(labels)).float().mean().item()

      print("=" * 60)
      print(f"[ACC] space-token      : {acc_space:.4f}")
      print(f"[ACC] semantic first  : {acc_first:.4f}")
      print("=" * 60)

      # ===== final choice =====
      if "llama-2" in model_name.lower():
          pred_strategy = "first semantic"
          final_preds = semantic_preds.cpu().tolist()
          final_logits = semantic_logits.cpu()
          for i, text in enumerate(label_info["ans_txt_list"]):
              tok = tokenizer.convert_ids_to_tokens([label_info["semantic_first"][i]])[0]
              print("[Double Check] used label")
              print(f"[{label_info['label_id'][i]}] {text:<10} | token_id = {label_info['semantic_first'][i]:<6} | token = {tok}")
      else:
          pred_strategy = "space"
          final_preds = space_preds.cpu().tolist()
          final_logits = space_logits.cpu()
          for i, text in enumerate(label_info["ans_txt_list"]):
              tok = tokenizer.convert_ids_to_tokens([label_info["space_first"][i]])[0]
              print("[Double Check] used label")
              print(f"[{label_info['label_id'][i]}] {text:<10} | token_id = {label_info['space_first'][i]:<6} | token = {tok}")

      metrics = self.compute_performance(final_preds, labels)
      metrics["pred_strategy"] = pred_strategy

      if return_logits:
          return metrics, final_logits
      return metrics

    def compute_performance(self, predictions, answers):
        
        num_classes = self.dataset.class_num
        TP = [0] * num_classes
        FP = [0] * num_classes
        FN = [0] * num_classes

        for y_true, y_pred in zip(answers, predictions):
            if y_true == y_pred:
                TP[y_true] += 1
            else:
                FP[y_pred] += 1
                FN[y_true] += 1

        f1 = []
        for i in range(num_classes):
            p = TP[i] / (TP[i] + FP[i]) if TP[i] + FP[i] > 0 else 0
            r = TP[i] / (TP[i] + FN[i]) if TP[i] + FN[i] > 0 else 0
            f1.append(2 * p * r / (p + r) if p + r > 0 else 0)

        acc = sum(int(p == t) for p, t in zip(predictions, answers)) / len(answers)
        macro_f1 = sum(f1) / num_classes

        metrics = {"acc": acc, "macro_f1": macro_f1}
        return metrics
    
    def get_label_info(self, tokenizer):
        ans_txt_list = self.dataset.get_dmonstration_template()['options']

        space_first = []
        semantic_first = []
        label_id = []
        
        for lid, label_text in enumerate(ans_txt_list):
            space_first.append(tokenizer.encode(" " + label_text, add_special_tokens=False)[0])
            semantic_first.append(tokenizer.encode(label_text, add_special_tokens=False)[0])
            label_id.append(lid)
            
        return {
                    "label_id": label_id,
                    "ans_txt_list": ans_txt_list,
                    "space_first": space_first,
                    "semantic_first": semantic_first
                }