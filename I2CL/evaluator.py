import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import global_vars as gv
import utils


class Evaluator(nn.Module):

    def __init__(self, dataset, batch_size):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size

    def evaluate(self, model_wrapper, tokenizer, demonstration='', use_cache=False,
                 return_logits=False, logits_mode='first', return_head_outputs=False,
                 return_q_states=False):  # ✅ 수정

        return self._evaluate_text_classification_batch(
            model_wrapper, tokenizer,
            demonstration, use_cache=use_cache, return_logits=return_logits,
            logits_mode=logits_mode, return_head_outputs=return_head_outputs,
            return_q_states=return_q_states
        )  # ✅ 수정

    def _evaluate_text_classification_batch(self, model_wrapper, tokenizer,
                                            demonstration, use_cache=False, return_logits=False,
                                            logits_mode='first', return_head_outputs=False,
                                            return_q_states=False):  # ✅ 수정

        model = model_wrapper.model
        # prepare label dict          
        label_map = {}
        ans_txt_list = self.dataset.get_dmonstration_template()['options']
        label_texts = [];
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
        all_pred_logits = [] if return_logits else None  # ✅ 수정
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
                    print("DEBUGGING: Input/Output Shapes (First Batch)")
                    print(f"{'=' * 60}")
                    print(f"Input IDs shape (after tokenization): {input_ids.shape}")
                    print(f"First input text: {cur_inputs[0][:100]}...")

                    # Check if model is PeftModel
                    from peft import PeftModel
                    if isinstance(model, PeftModel):
                        print(f"Model type: PeftModel (soft prompts should be prepended)")
                        if hasattr(model, 'peft_config') and 'default' in model.peft_config:
                            num_virtual_tokens = getattr(model.peft_config['default'], 'num_virtual_tokens', 0)
                            print(f"Num Virtual Tokens (Soft Prompts): {num_virtual_tokens}")
                            print(
                                f"Expected output seq length: {input_ids.shape[1]} + {num_virtual_tokens} = {input_ids.shape[1] + num_virtual_tokens}"
                            )
                    else:
                        print(f"Model type: {type(model).__name__} (NOT PeftModel - no soft prompts)")

                # get index for prediction logits, need to be applied before concatenating demon_attn_mask with attn_mask
                pred_loc = utils.last_one_indices(attn_mask).to(model.device)

                # ✅ CRITICAL FIX: Adjust pred_loc for soft prompts!
                from peft import PeftModel
                import wrapper
                num_virtual_tokens = 0

                # Get actual model: if model_wrapper is ModelWrapper, use model_wrapper.model
                # Otherwise use the model directly (already unwrapped at line 24)
                actual_model = model_wrapper.model if isinstance(model_wrapper, wrapper.ModelWrapper) else model

                if isinstance(actual_model, PeftModel):
                    if hasattr(actual_model, 'peft_config') and 'default' in actual_model.peft_config:
                        num_virtual_tokens = getattr(actual_model.peft_config['default'], 'num_virtual_tokens', 0)
                        if num_virtual_tokens > 0:
                            # Soft prompts are prepended, so shift pred_loc
                            pred_loc = pred_loc + num_virtual_tokens
                            if batch_idx == 0:
                                print(f"⚠️  ADJUSTED pred_loc by {num_virtual_tokens} (soft prompts)")

                # set global variables
                gv.ATTN_MASK_START = torch.zeros_like(pred_loc)
                gv.ATTN_MASK_END = pred_loc
                if use_cache:
                    attn_mask = torch.cat([demon_attn_mask, attn_mask], dim=1)
                    output = model(
                        input_ids=input_ids, attention_mask=attn_mask,
                        past_key_values=demon_past_key_values, use_cache=use_cache,
                        return_head_outputs=return_head_outputs,
                        return_q_states=return_q_states
                    )
                else:
                    output = model(
                        input_ids=input_ids, attention_mask=attn_mask, use_cache=False,
                        return_head_outputs=return_head_outputs,
                        return_q_states=return_q_states
                    )
                logits = output.logits

                # DEBUGGING: Print output shape for first batch only
                if batch_idx == 0:
                    print(f"Output logits shape: {logits.shape}")
                    print(f"Prediction location (pred_loc): {pred_loc[0].item()}")
                    if isinstance(model, PeftModel) and hasattr(
                            model, 'peft_config'
                    ) and 'default' in model.peft_config:
                        num_virtual_tokens = getattr(model.peft_config['default'], 'num_virtual_tokens', 0)
                        print(f"⚠️  CRITICAL: Logits seq length should be input + virtual tokens!")
                        print(f"   Input seq: {input_ids.shape[1]}, Virtual tokens: {num_virtual_tokens}")
                        print(f"   Output seq: {logits.shape[1]}")
                        if logits.shape[1] == input_ids.shape[1]:
                            print(f"   ❌ ERROR: Soft prompts NOT applied (seq length unchanged)!")
                        elif logits.shape[1] == input_ids.shape[1] + num_virtual_tokens:
                            print(f"   ✅ OK: Soft prompts ARE applied (seq length increased)!")
                        else:
                            print(f"   ⚠️  WARNING: Unexpected seq length difference!")
                    print(f"{'=' * 60}\n")

                # ✅ 수정
                if logits_mode == 'first':
                    pred_logits = logits[torch.arange(logits.size(0)), pred_loc]  # (B,V)
                    interest_index = list(label_map.keys())
                    pred_logits = pred_logits[:, interest_index]  # (B,K)

                    scores = F.softmax(pred_logits, dim=-1)  # 굳이 안해도 된다는데 체크
                    pred_labels = scores.argmax(dim=-1)

                    # decode pred_labels to text
                    pred_labels_list = pred_labels.cpu().numpy().tolist()
                    pred_labels_text = [ans_txt_list[label] for label in pred_labels_list]
                    cur_labels_text = [ans_txt_list[label] for label in cur_labels]

                else:
                    raise ValueError(f"Unknown logits_mode: {logits_mode}")

                if return_logits:
                    all_pred_logits.append(scores.detach().cpu())
                all_pred_labels.extend(pred_labels.cpu().numpy().tolist())

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
        metrics = {'acc': acc, 'macro_f1': macro_f1}

        if return_logits:  # ✅ 수정
            all_pred_logits = torch.cat(all_pred_logits, dim=0)  # (N, num_classes)
            return metrics, all_pred_logits, all_labels
        else:
            return metrics
