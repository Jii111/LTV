"""Evaluation utilities for classification-style ICL experiments."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple

import global_vars as gv
import utils
import pandas as pd

class Evaluator(nn.Module):
    """Batch evaluator for text classification tasks."""

    def __init__(self, dataset: Any, batch_size: int) -> None:
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size

    def evaluate(
        self,
        model_wrapper: Any,
        tokenizer: Any,
        demonstration: str = '',
        use_cache: bool = False,
        return_logits: bool = False,
        return_head_outputs: bool = False,
        fv_vector: Optional[torch.Tensor] = None,
        sv_logit: Optional[torch.Tensor] = None,
        edit_layer: Optional[int] = None,
        model_config: Optional[Any] = None,
        return_q_states: bool = False,
    ):
        """Evaluate a model wrapper on the dataset."""
        return self._evaluate_text_classification_batch(
            model_wrapper, tokenizer,
            demonstration, use_cache=use_cache, return_logits=return_logits,
            return_head_outputs=return_head_outputs, fv_vector=fv_vector, sv_logit=sv_logit, edit_layer=edit_layer, model_config=model_config,
            return_q_states=return_q_states
        )  
        
    def _evaluate_text_classification_batch(
        self,
        model_wrapper: Any,
        tokenizer: Any,
        demonstration: str,
        use_cache: bool = False,
        return_logits: bool = False,
        fv_vector: Optional[torch.Tensor] = None,
        sv_logit: Optional[torch.Tensor] = None,
        edit_layer: Optional[int] = None,
        model_config: Optional[Any] = None,
        return_head_outputs: bool = False,
        return_q_states: bool = False,
    ):
        """Run batched evaluation and collect logits."""

        model = model_wrapper.model
        all_base_logits = []
        all_inputs, all_labels = [], []
        for data in self.dataset.all_data:
            ques_str, _, label = self.dataset.apply_template(data)
            context = ques_str if (use_cache or len(demonstration) == 0) else demonstration + ques_str
            all_inputs.append(context)
            all_labels.append(label)

        use_cache = False

        for batch_idx, i in enumerate(range(0, len(all_inputs), self.batch_size)):
            cur_inputs = all_inputs[i:i + self.batch_size]

            input_tok = tokenizer(cur_inputs, return_tensors="pt", padding=True)
            input_ids = input_tok['input_ids'].to(model.device)
            attn_mask = input_tok['attention_mask'].to(model.device)

            pred_loc = utils.last_one_indices(attn_mask).to(model.device)
            gv.ATTN_MASK_START = torch.zeros_like(pred_loc)
            gv.ATTN_MASK_END = pred_loc
                      
            output = model(input_ids=input_ids,attention_mask=attn_mask, 
                use_cache=False,return_head_outputs=return_head_outputs,return_q_states=return_q_states)
            logits = output.logits

            base_logits = logits[torch.arange(logits.size(0)), pred_loc]
            all_base_logits.append(base_logits.detach().cpu())

            del logits
            torch.cuda.empty_cache()
        
        if return_logits:
            metrics, all_pred_logits = self.evaluate_logits(
                all_base_logits, all_labels, tokenizer, model_wrapper.model.config._name_or_path, return_logits=return_logits)
            return metrics, all_pred_logits, all_labels
        else:
            metrics = self.evaluate_logits(all_base_logits, all_labels, tokenizer, model_wrapper.model.config._name_or_path, return_logits=return_logits)
            return metrics
        
    def evaluate_logits(
        self,
        base_logits: List[torch.Tensor],
        labels: List[int],
        tokenizer: Any,
        model_name: str,
        return_logits: bool = False,
    ):
      """Compute metrics from collected logits."""
      label_info = self.get_label_info(tokenizer)

      logits = torch.cat(base_logits, dim=0)    # (N, V)
      log_probs = F.log_softmax(logits, dim=-1)

      space_preds = log_probs[:, label_info["space_first"]].argmax(dim=-1)
      space_logits = logits[:, label_info["space_first"]]

      semantic_preds = log_probs[:, label_info["semantic_first"]].argmax(dim=-1)
      semantic_logits = logits[:, label_info["semantic_first"]]

      if "llama-2" in model_name.lower():
          pred_strategy = "first semantic"
          final_preds = semantic_preds.cpu().tolist()
          final_logits = semantic_logits.cpu()
          for i, text in enumerate(label_info["ans_txt_list"]):
              tok = tokenizer.convert_ids_to_tokens([label_info["semantic_first"][i]])[0]
      else:
          pred_strategy = "space"
          final_preds = space_preds.cpu().tolist()
          final_logits = space_logits.cpu()
          for i, text in enumerate(label_info["ans_txt_list"]):
              tok = tokenizer.convert_ids_to_tokens([label_info["space_first"][i]])[0]

      metrics = self.compute_performance(final_preds, labels)
      metrics["pred_strategy"] = pred_strategy

      if return_logits:
          return metrics, final_logits
      return metrics

    def compute_performance(self, predictions: List[int], answers: List[int]) -> Dict[str, float]:
        """Compute accuracy and macro-F1 for predictions."""
        
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
    
    def get_label_info(self, tokenizer: Any) -> Dict[str, Any]:
        """Build label token mappings for the dataset."""
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
