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

    def _evaluate_text_classification_batch2(self, model_wrapper, tokenizer,
                                            demonstration, use_cache=False, return_logits=False,
                                            return_head_outputs=False, return_q_states=False): 

        model = model_wrapper.model
        # prepare label dict          
        label_map = {}
        ans_txt_list = self.dataset.get_dmonstration_template()['options']
        label_texts = []
        label_tok_list = []

        for label, ans_txt in enumerate(ans_txt_list):
            if 'gpt' in tokenizer.__class__.__name__.lower():
                ans_txt = ' ' + ans_txt  # add space to the beginning of answer
            if 'qwen' in tokenizer.__class__.__name__.lower():
                ans_txt = ' ' + ans_txt  # add space to the beginning of answer
            toks = tokenizer.encode(ans_txt, add_special_tokens=False)
            ans_tok = toks[0]  # use the first token if more than one token
            print(f"ans_txt: {ans_txt}, ans_tok: {ans_tok}")
            label_map[ans_tok] = label  # index is the label
            label_texts.append(ans_txt)
            label_tok_list.append(toks)
        print(f"label_map: {label_map}")

        # prepare all data
        all_pred_labels = []
        all_pred_logits = [] if return_logits else None 
        all_inputs, all_labels = [], []
        for data in self.dataset.all_data:
            ques_str, _, label = self.dataset.apply_template(data)
            if use_cache or len(demonstration) == 0:
                context = ques_str
            else:
                context = demonstration + ques_str
            all_inputs.append(context)
            all_labels.append(label)

        # cache the demonstration
        if len(demonstration) > 0 and use_cache:
            demon_token = tokenizer(demonstration, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                demon_outputs = model(**demon_token, use_cache=True)
            demon_past_key_values = demon_outputs.past_key_values
            demon_attn_mask = demon_token['attention_mask']
            demon_past_key_values = tuple(
                tuple(
                    t.repeat(self.batch_size, 1, 1, 1) for
                    t in tup
                ) for tup in demon_past_key_values
            )
            demon_attn_mask = demon_attn_mask.repeat(self.batch_size, 1)
            if len(all_inputs) % self.batch_size != 0:  # last batch
                sp_demon_past_key_values = tuple(
                    tuple(
                        t.repeat(len(all_inputs) % self.batch_size, 1, 1, 1)
                        for t in tup
                    ) for tup in demon_outputs.past_key_values
                )
                sp_demon_attn_mask = demon_attn_mask[-(len(all_inputs) % self.batch_size):]
            use_cache = True
        else:
            demon_past_key_values = None
            sp_demon_past_key_values = None
            sp_demon_attn_mask = None
            use_cache = False

        # loop over all data
        with torch.no_grad():
            batch_iter = tqdm(
                enumerate(range(0, len(all_inputs), self.batch_size)),
                total=(len(all_inputs) + self.batch_size - 1) // self.batch_size,
                desc="Evaluating"
            )
            for batch_idx, i in batch_iter:
                cur_inputs = all_inputs[i:i + self.batch_size]
                cur_labels = all_labels[i:i + self.batch_size]
                # accommodate for the last batch
                if len(cur_inputs) != self.batch_size:
                    demon_past_key_values = sp_demon_past_key_values
                    demon_attn_mask = sp_demon_attn_mask
                input_tok = tokenizer(cur_inputs, return_tensors="pt", padding=True, truncation=False)
                input_ids = input_tok['input_ids'].to(model.device)
                if input_ids.shape[1] > 4096:
                    raise ValueError("Too many inputs")
                attn_mask = input_tok['attention_mask'].to(model.device)

                # DEBUGGING: Print shape info for first batch only
                if batch_idx == 0:
                    print(f"\n{'=' * 60}")
                    print("DEBUGGING: Input/Output Shapes (First Batch, I2CL evaluator)")
                    print(f"{'=' * 60}")
                    print(f"Input IDs shape (after tokenization): {input_ids.shape}")

                # get index for prediction logits, need to be applied before concatenating demon_attn_mask with attn_mask
                pred_loc = utils.last_one_indices(attn_mask).to(model.device)

                # set global variables
                gv.ATTN_MASK_START = torch.zeros_like(pred_loc)
                gv.ATTN_MASK_END = pred_loc
                if use_cache:
                    attn_mask = torch.cat([demon_attn_mask, attn_mask], dim=1)
                    with torch.no_grad():
                        output = model(
                            input_ids=input_ids, attention_mask=attn_mask,
                            past_key_values=demon_past_key_values, use_cache=use_cache,
                            return_head_outputs=return_head_outputs,
                            return_q_states=return_q_states
                        )
                else:
                    with torch.no_grad():
                        output = model(
                            input_ids=input_ids, attention_mask=attn_mask, use_cache=False,
                            return_head_outputs=return_head_outputs,
                            return_q_states=return_q_states
                        )
                logits = output.logits


                pred_logits = logits[torch.arange(logits.size(0)), pred_loc]  # (B,V)
                interest_index = list(label_map.keys())
                pred_logits = pred_logits[:, interest_index]  # (B,K)

                scores = F.softmax(pred_logits, dim=-1) 
                pred_labels = scores.argmax(dim=-1)

                # decode pred_labels to text
                pred_labels_list = pred_labels.cpu().numpy().tolist()
                pred_labels_text = [ans_txt_list[label] for label in pred_labels_list]
                cur_labels_text = [ans_txt_list[label] for label in cur_labels]
                
                                # DEBUGGING: Print output shape for first batch only
                if batch_idx == 0:
                    print("cur inputs : ",cur_inputs)
                    print(f"Output logits shape: {logits.shape}")
                    print(f"Prediction location (pred_loc): {pred_loc[0].item()}")
                    print("interest_index:", interest_index)
                    print("label_map:", label_map)
                    print("answer token check:", tokenizer.decode(input_ids[0][-20:]))
                    label_token_ids = tokenizer.encode("Label:", add_special_tokens=False)
                    print("label_token_ids:", label_token_ids)
                    for b in range(min(2, input_ids.size(0))):
                        loc = pred_loc[b].item()
                        print("pred_loc context:",
                            tokenizer.decode(input_ids[b][loc-5:loc+5]))
                    print("\n=== Label map vs full tokenization ===")
                    for label_id, label_text in enumerate(ans_txt_list):
                        toks = tokenizer.encode(" " + label_text, add_special_tokens=False)
                        first_tok = toks[0]
                        first_tok_dec = tokenizer.decode([first_tok])

                        print(f"Label {label_id}: '{label_text}'")
                        print(f"  full tokens : {toks}")
                        print(f"  decoded     : {[tokenizer.decode([t]) for t in toks]}")
                        print(f"  used token  : {first_tok} -> '{first_tok_dec}'")
                        print(f"  num_tokens  : {len(toks)}\n")
                    print(f"{'=' * 60}\n")

                if return_logits:
                    all_pred_logits.append(scores.detach().cpu())
                all_pred_labels.extend(pred_labels.cpu().numpy().tolist())
                
                
                del logits, pred_logits, output
                torch.cuda.empty_cache()

        print("예측 : ")
        print(pd.Series(all_pred_labels).value_counts())
        print("정답 : ")
        print(pd.Series(all_labels).value_counts())
        assert len(all_pred_labels) == len(all_labels)
        # both all_results and all_labels are list containing label index, can you help me to calculate accuracy and macro f1?
        # initialize TP, FP, FN
        acc = []
        num_classes = self.dataset.class_num
        TP = [0] * num_classes
        FP = [0] * num_classes
        FN = [0] * num_classes
        for i, true_label in enumerate(all_labels):
            pred_label = all_pred_labels[i]
            pred = pred_label == true_label
            acc.append(pred)
            # Update TP, FP, FN
            if pred:
                TP[true_label] += 1
            else:
                FP[pred_label] += 1
                FN[true_label] += 1
        # Calculate precision, recall, F1 for each class and macro F1
        precision = [0] * num_classes
        recall = [0] * num_classes
        f1 = [0] * num_classes
        for i in range(num_classes):
            precision[i] = TP[i] / (TP[i] + FP[i]) if (TP[i] + FP[i]) > 0 else 0
            recall[i] = TP[i] / (TP[i] + FN[i]) if (TP[i] + FN[i]) > 0 else 0
            f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0
        macro_f1 = sum(f1) / num_classes
        acc = sum(acc) / len(acc)

        macro_acc = 0
        class_acc = [0] * num_classes
        for i in range(num_classes):
            class_acc[i] = TP[i] / (TP[i] + FN[i]) if (TP[i] + FN[i]) > 0 else 0
        macro_acc = sum(class_acc) / num_classes
        print("macro_acc : ", macro_acc)
        metrics = {'acc': acc, 'macro_f1': macro_f1}

        if return_logits:  # ✅ 수정
            all_pred_logits = torch.cat(all_pred_logits, dim=0)  # (N, num_classes)
            return metrics, all_pred_logits, all_labels
        else:
            return metrics