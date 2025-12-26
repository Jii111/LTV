import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import global_vars as gv
import utils
import pandas as pd

from fv_utils.intervention_utils import *

class Evaluator(nn.Module):

    def __init__(self, dataset, batch_size):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size

    def evaluate(self, model_wrapper, tokenizer, demonstration='', use_cache=False, return_logits=False, 
                 return_head_outputs=False, fv_vector=None, fv_edit_layer=None, model_config=None,
                 return_q_states=False): 

        return self._evaluate_text_classification_batch(
            model_wrapper, tokenizer,
            demonstration, use_cache=use_cache, return_logits=return_logits,
            return_head_outputs=return_head_outputs, fv_vector=fv_vector, fv_edit_layer=fv_edit_layer, model_config=model_config,
            return_q_states=return_q_states
        )  
        
    def _evaluate_text_classification_batch(self, model_wrapper, tokenizer,
                                            demonstration, use_cache=False, return_logits=False, 
                                            fv_vector=None, fv_edit_layer=None, model_config=None,
                                            return_head_outputs=False, return_q_states=False):

        model = model_wrapper.model

        # ======================================================
        # 1. Prepare label tokens
        # ======================================================
        ans_txt_list = self.dataset.get_dmonstration_template()['options']

        space_label_tokens = []
        semantic_label_tokens = []
        nospace_token_lists = [] 
        all_space_preds = []
        all_semantic_preds = []
        all_space_logits = []
        all_semantic_logits = []

        for label_text in ans_txt_list:
            toks_space = tokenizer.encode(" " + label_text, add_special_tokens=False)
            space_label_tokens.append(toks_space[0])

            toks_nospace = tokenizer.encode(label_text, add_special_tokens=False)
            semantic_label_tokens.append(toks_nospace[0])
            nospace_token_lists.append(toks_nospace)  

        # ======================================================
        # 2. Prepare data
        # ======================================================
        all_inputs, all_labels = [], []
        for data in self.dataset.all_data:
            ques_str, _, label = self.dataset.apply_template(data)
            context = ques_str if (use_cache or len(demonstration) == 0) else demonstration + ques_str
            all_inputs.append(context)
            all_labels.append(label)

        # ======================================================
        # 3. Cache demonstration if needed
        # ======================================================
        if len(demonstration) > 0 and use_cache:
            demon_token = tokenizer(demonstration, return_tensors="pt").to(model.device)
            with torch.no_grad():
                demon_outputs = model(**demon_token, use_cache=True)

            demon_past_key_values = tuple(
                tuple(t.repeat(self.batch_size, 1, 1, 1) for t in layer)
                for layer in demon_outputs.past_key_values
            )
            demon_attn_mask = demon_token['attention_mask'].repeat(self.batch_size, 1)
        else:
            demon_past_key_values = None
            demon_attn_mask = None
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
                        
            if fv_vector is None:
                output = model(input_ids=input_ids,attention_mask=attn_mask,
                    use_cache=False,return_head_outputs=return_head_outputs,return_q_states=return_q_states)
            else:
                intervention_idx = -1
                intervention_fn = add_function_vector(fv_edit_layer, fv_vector.reshape(1, model_config.fv['resid_dim']), model.device, idx=intervention_idx)
                with TraceDict(model, layers=model_config.fv['layer_hook_names'], edit_output=intervention_fn) as td:    
                    output = model(input_ids=input_ids,attention_mask=attn_mask,
                    use_cache=False,return_head_outputs=return_head_outputs,return_q_states=return_q_states)
            logits = output.logits
            base_logits = logits[torch.arange(logits.size(0)), pred_loc]
            log_probs = F.log_softmax(base_logits, dim=-1)   # [ADD]

            # (A) space-token
            space_preds = log_probs[:, space_label_tokens].argmax(dim=-1)
            space_logits = base_logits[:, space_label_tokens]

            # (B) semantic first token
            semantic_preds = log_probs[:, semantic_label_tokens].argmax(dim=-1)
            semantic_logits = base_logits[:, semantic_label_tokens]

            all_space_preds.extend(space_preds.cpu().tolist())
            all_semantic_preds.extend(semantic_preds.cpu().tolist())
            all_space_logits.append(space_logits.detach().cpu())
            all_semantic_logits.append(semantic_logits.detach().cpu())

            # -------------------------------
            # Debug: first batch only
            # -------------------------------
            if batch_idx == 0:
                full_ids = tokenizer.encode(cur_inputs[0], add_special_tokens=False)
                print("\n=== FULL TOKENIZATION (tail) ===")
                print(full_ids[-50:])
                print(tokenizer.convert_ids_to_tokens(full_ids[-50:]))
                print("\n=== Label strings ===")
                print(ans_txt_list)

                loc = pred_loc[0].item()
                print(pred_loc)
                print("pred_loc context:",
                      tokenizer.decode(input_ids[0][loc-10:loc+10]))

                print("\n=== Label tokenization check ===")
                for lid, text in enumerate(ans_txt_list):
                    print(f"[{lid}] {text}")
                    print("  space   :", tokenizer.encode(" " + text, add_special_tokens=False))
                    print("  nospace :", tokenizer.encode(text, add_special_tokens=False))
                    
                print("\n=== Prediction ↔ Label meaning check (first batch) ===")
                for b in range(min(5, len(cur_inputs))):
                    gt = all_labels[i + b]

                    sp = space_preds[b].item()
                    se = semantic_preds[b].item()

                    print(f"\n[Sample {b}]")
                    print("GT label id :", gt, "->", ans_txt_list[gt])
                    print("space pred :", sp, "->", ans_txt_list[sp])
                    print("first pred :", se, "->", ans_txt_list[se])

                # 1. full prompt tokenization
                full_ids = tokenizer.encode(cur_inputs[0], add_special_tokens=False)
                full_toks = tokenizer.convert_ids_to_tokens(full_ids)

                print("\n=== Label token sanity check ===")
                print(f"{'label':<12} {'semantic':>10} {'tok':>12} | {'space':>10} {'tok':>12} | {'ICL':>10} {'tok':>12}")
                print("-" * 80)

                for label in ans_txt_list:
                    # semantic
                    sem_ids = tokenizer.encode(label, add_special_tokens=False)
                    sem_id = sem_ids[0]
                    sem_tok = tokenizer.convert_ids_to_tokens([sem_id])[0]

                    # space
                    sp_ids = tokenizer.encode(" " + label, add_special_tokens=False)
                    sp_id = sp_ids[0]
                    sp_tok = tokenizer.convert_ids_to_tokens([sp_id])[0]
                    # ICL
                    icl_id = full_ids[-1]
                    icl_tok = tokenizer.convert_ids_to_tokens([icl_id])[0]

                    print(f"{label:<12} {sem_id:>10} {sem_tok:>12} | {sp_id:>10} {sp_tok:>12} | {icl_id:>10} {icl_tok:>12}")

            del logits
            torch.cuda.empty_cache()

        # ======================================================
        # 5. Print prediction distributions
        # ======================================================
        print("\n[DEBUG] Space-token prediction count")
        print(pd.Series(all_space_preds).value_counts().sort_index())

        print("\n[DEBUG] No-space first-token prediction count")
        print(pd.Series(all_semantic_preds).value_counts().sort_index())

        print("\n[DEBUG] Ground-truth label count")
        print(pd.Series(all_labels).value_counts().sort_index())
        
        # ======================================================
        # 6. Print accuracy comparison (no return change)
        # ======================================================
        labels = torch.tensor(all_labels)

        acc_space = (torch.tensor(all_space_preds) == labels).float().mean().item()
        acc_first = (torch.tensor(all_semantic_preds) == labels).float().mean().item()

        print("=" * 60)
        print(f"[ACC] space-token      : {acc_space:.4f}")
        print(f"[ACC] semantic first  : {acc_first:.4f}")
        print("=" * 60)

        # ======================================================
        # 7. ORIGINAL RETURN (UNCHANGED)
        # ======================================================
        if "llama" in model.config._name_or_path.lower():
            pred_strategy = "space"
            final_preds = all_space_preds
            final_logits = all_space_logits
        else:
            pred_strategy = "first semantic"
            final_preds = all_semantic_preds
            final_logits = all_semantic_logits
        
        del all_space_preds, all_semantic_preds, all_space_logits, all_semantic_logits
        
        num_classes = self.dataset.class_num
        TP = [0] * num_classes
        FP = [0] * num_classes
        FN = [0] * num_classes

        for y_true, y_pred in zip(all_labels, final_preds):
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

        acc = sum(int(p == t) for p, t in zip(final_preds, all_labels)) / len(all_labels)
        macro_f1 = sum(f1) / num_classes

        metrics = {"acc": acc, "macro_f1": macro_f1, "pred_strategy": pred_strategy}

        if return_logits:
            all_pred_logits = torch.cat(final_logits, dim=0)
            return metrics, all_pred_logits, all_labels
        else:
            return metrics
