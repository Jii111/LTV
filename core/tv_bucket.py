"""
Position-bucketed, teacher-forced task vector extraction + per-step injection.

Upgrades the single-point (t=0 only) M2-Adaptive extraction used in
run/run_math_gen.py along three axes:

  1. Multi-position samples: for each anchor query, the ICL context greedy-
     decodes a teacher answer y-hat, then two teacher-forced forwards
     ((demo,x,y-hat) and (x,y-hat)) collect h_icl(t)/h_zs(t) at every answer
     position t, not just the t=0 boundary. N anchors -> up to N*T samples.
  2. Bucket ridge: Delta_t = h_icl(t) - h_zs(t) decays with t (the zero-shot
     forward is teacher-forced with the same y-hat prefix, so both contexts
     converge as t grows). A single pooled W would be dominated by the
     near-zero large-t targets and wash out the t=0 correction, so samples
     are split into three buckets (t=0 / early / late) with independent
     ridge fits. Lambda is chosen per bucket via anchor-level CV (whole
     trajectories held out together — position-level CV leaks strongly
     correlated adjacent-t samples across folds and under-regularizes).
  3. Token-ID concatenation: the teacher continuation y-hat is appended to
     each context as token IDs, never by re-tokenizing concatenated
     strings, so token boundaries never shift between the extraction pass
     and the greedy-decode pass that produced y-hat.

Injection (`inject_bucket_stepwise`) mirrors this: every forward through the
last decoder layer (prefill and each KV-cache decode step) adds the
bucket-appropriate W to the hidden state at the last sequence position, so
the correction persists across the whole generation instead of being
applied once before the first token.
"""

import random
import re
from contextlib import contextmanager
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm
from transformers import StoppingCriteria, StoppingCriteriaList

import core.utils.utils_method as um

BUCKETS = ('w0', 'early', 'late')


class _StopStringCriteria(StoppingCriteria):
    """Stop generation once the sequence matches stop_pattern (a regex)."""
    def __init__(self, tokenizer, stop_pattern: str, prompt_len: int):
        self.tokenizer = tokenizer
        self.stop_pattern = stop_pattern
        self.prompt_len = prompt_len

    def __call__(self, input_ids: torch.LongTensor, scores, **_kwargs) -> bool:
        text = self.tokenizer.decode(input_ids[0, self.prompt_len:], skip_special_tokens=True)
        return re.search(self.stop_pattern, text) is not None


def _bucket_for(t: int, early_range: Tuple[int, int]) -> str:
    if t == 0:
        return 'w0'
    elif early_range[0] <= t <= early_range[1]:
        return 'early'
    return 'late'


def _is_degenerate(token_ids: List[int], ngram: int = 3, min_repeats: int = 4) -> bool:
    """Detect a length-`ngram` window that repeats >= min_repeats times in a row."""
    span = ngram * min_repeats
    if len(token_ids) < span:
        return False
    for i in range(len(token_ids) - span + 1):
        window = token_ids[i:i + ngram]
        if all(token_ids[i + k * ngram:i + (k + 1) * ngram] == window for k in range(min_repeats)):
            return True
    return False


def _truncate_at_stop(answer_ids: torch.Tensor, tokenizer, stop_pattern: Optional[str]) -> torch.Tensor:
    """Cut `answer_ids` (1, L) at the first EOS token, and at the first regex
    match of `stop_pattern` in the decoded answer text. This re-tokenizes
    only the answer's own prefix (to count how many of its own tokens to
    keep) — never the demo/query/answer boundary, so it doesn't reintroduce
    the string-concat drift the token-ID-concat design avoids elsewhere."""
    ids = answer_ids[0]
    if tokenizer.eos_token_id is not None:
        eos_pos = (ids == tokenizer.eos_token_id).nonzero()
        if len(eos_pos) > 0:
            ids = ids[:eos_pos[0].item()]
    if stop_pattern:
        text = tokenizer.decode(ids, skip_special_tokens=True)
        m = re.search(stop_pattern, text)
        if m:
            prefix_ids = tokenizer(text[:m.start()], add_special_tokens=False).input_ids
            ids = ids[:len(prefix_ids)]
    return ids.unsqueeze(0)


def _anchor_level_ridge_lambda(
    features: torch.Tensor,   # (n, d)
    targets: torch.Tensor,    # (n, d)
    anchor_ids: List[int],    # (n,)
    lambda_candidates: Sequence[float],
    n_folds: int,
    device,
) -> float:
    """Pick lambda by k-fold CV where whole anchors (all their t-samples) are
    held out together, never split across train/val."""
    unique_anchors = sorted(set(anchor_ids))
    if len(unique_anchors) < 2 or len(lambda_candidates) == 1:
        return lambda_candidates[0]

    n_folds = max(2, min(n_folds, len(unique_anchors)))
    rng = random.Random(0)
    shuffled = unique_anchors[:]
    rng.shuffle(shuffled)
    folds = [set(shuffled[i::n_folds]) for i in range(n_folds)]

    anchor_arr = torch.as_tensor(anchor_ids)
    best_lambda, best_score = lambda_candidates[0], float('inf')
    for lam in lambda_candidates:
        fold_scores = []
        for fold in folds:
            held_mask = torch.tensor([a in fold for a in anchor_ids])
            if held_mask.sum() == 0 or (~held_mask).sum() == 0:
                continue
            F_tr = features[~held_mask].T.to(device)  # (d, n_tr)
            T_tr = targets[~held_mask].T.to(device)
            W = um.ridge_regression(T_tr, F_tr, lam, device)
            F_val = features[held_mask].to(device)
            T_val = targets[held_mask].to(device)
            pred = (W @ F_val.T).T
            fold_scores.append(((pred - T_val) ** 2).mean().item())
        if fold_scores:
            avg = sum(fold_scores) / len(fold_scores)
            if avg < best_score:
                best_score, best_lambda = avg, lam
    return best_lambda


@torch.no_grad()
def extract_bucket_task_vectors(
    model_wrapper,
    tokenizer,
    demo: str,
    train_queries: List[str],
    max_new_tokens: int = 48,
    stop_pattern: Optional[str] = None,
    early_range: Tuple[int, int] = (1, 4),
    ridge_lambdas: Sequence[float] = (0.1, 1.0, 5.0, 10.0, 50.0),
    n_cv_folds: int = 5,
    repeat_ngram: int = 3,
    repeat_min_repeats: int = 4,
    verbose: bool = True,
) -> Tuple[Dict[str, Optional[torch.Tensor]], Dict[str, Optional[float]], dict]:
    """
    Returns:
      Ws: {'w0'|'early'|'late' -> W tensor (d, d) or None if bucket got 0 samples}
      chosen_lambdas: same keys -> the lambda selected by anchor-level CV
      diagnostics: sample/anchor counts per bucket + degenerate-rollout count
    """
    model, device = model_wrapper.model, model_wrapper.device

    # Demo is tokenized once, with the leading BOS. Each query is tokenized
    # twice: with BOS (as a standalone zero-shot prompt) and without BOS (to
    # be concatenated onto the demo without duplicating BOS in the middle).
    demo_ids = tokenizer(demo, return_tensors='pt', add_special_tokens=True).input_ids.to(device)

    samples = {b: {'feat': [], 'tgt': [], 'anchor': []} for b in BUCKETS}
    n_degenerate = 0

    for anchor_id, q in enumerate(tqdm(train_queries, desc="Extract bucket TVs", disable=not verbose)):
        q_bos = tokenizer(q, return_tensors='pt', add_special_tokens=True).input_ids.to(device)
        q_nobos = tokenizer(q, return_tensors='pt', add_special_tokens=False).input_ids.to(device)

        icl_prompt_ids = torch.cat([demo_ids, q_nobos], dim=1)
        gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False,
                          pad_token_id=tokenizer.eos_token_id)
        if stop_pattern:
            gen_kwargs['stopping_criteria'] = StoppingCriteriaList([
                _StopStringCriteria(tokenizer, stop_pattern, icl_prompt_ids.shape[1])
            ])
        out = model.generate(icl_prompt_ids, **gen_kwargs)
        answer_ids = out[:, icl_prompt_ids.shape[1]:]
        answer_ids = _truncate_at_stop(answer_ids, tokenizer, stop_pattern)
        if answer_ids.shape[1] == 0:
            continue
        if _is_degenerate(answer_ids[0].tolist(), repeat_ngram, repeat_min_repeats):
            n_degenerate += 1
            continue

        full_icl_ids = torch.cat([demo_ids, q_nobos, answer_ids], dim=1)
        full_zs_ids = torch.cat([q_bos, answer_ids], dim=1)

        h_icl_all = model(input_ids=full_icl_ids, output_hidden_states=True,
                          use_cache=False).hidden_states[-1][0]
        h_zs_all = model(input_ids=full_zs_ids, output_hidden_states=True,
                         use_cache=False).hidden_states[-1][0]

        ans_len = answer_ids.shape[1]
        icl_start = demo_ids.shape[1] + q_nobos.shape[1]  # index of answer token 0 in full_icl_ids
        zs_start = q_bos.shape[1]                          # index of answer token 0 in full_zs_ids

        for t in range(ans_len):
            h_icl_t = h_icl_all[icl_start - 1 + t]
            h_zs_t = h_zs_all[zs_start - 1 + t]
            bucket = _bucket_for(t, early_range)
            samples[bucket]['feat'].append(h_zs_t.cpu())
            samples[bucket]['tgt'].append((h_icl_t - h_zs_t).cpu())
            samples[bucket]['anchor'].append(anchor_id)

        del h_icl_all, h_zs_all, out, answer_ids
        torch.cuda.empty_cache()

    Ws: Dict[str, Optional[torch.Tensor]] = {}
    chosen_lambdas: Dict[str, Optional[float]] = {}
    diagnostics = {'n_degenerate': n_degenerate, 'n_anchors': len(train_queries)}

    for bucket in BUCKETS:
        feat, tgt, anc = samples[bucket]['feat'], samples[bucket]['tgt'], samples[bucket]['anchor']
        diagnostics[f'{bucket}_n_samples'] = len(feat)
        diagnostics[f'{bucket}_n_anchors'] = len(set(anc))
        if not feat:
            Ws[bucket], chosen_lambdas[bucket] = None, None
            continue
        F_mat = torch.stack(feat)  # (n, d)
        T_mat = torch.stack(tgt)   # (n, d)
        lam = _anchor_level_ridge_lambda(F_mat, T_mat, anc, list(ridge_lambdas), n_cv_folds, device)
        W = um.ridge_regression(T_mat.T, F_mat.T, lam, device).cpu()
        Ws[bucket], chosen_lambdas[bucket] = W, lam

    return Ws, chosen_lambdas, diagnostics


@contextmanager
@torch.no_grad()
def inject_bucket_stepwise(
    model_wrapper,
    W_w0: Optional[torch.Tensor] = None,
    W_early: Optional[torch.Tensor] = None,
    W_late: Optional[torch.Tensor] = None,
    early_range: Tuple[int, int] = (1, 4),
):
    """
    Adds W_bucket(t) @ h to the last-position hidden state of every forward
    through the last decoder layer, where t is the count of answer tokens
    generated so far (t=0 at prefill, t=1 after the first generated token,
    ...). Passing only W_w0 (early/late=None) reproduces the old
    prefill-only injection, since bucket_for(t>=1) then resolves to None and
    the hook is a no-op on every decode step.
    """
    device = model_wrapper.device
    Ws = {
        'w0': W_w0.to(device) if W_w0 is not None else None,
        'early': W_early.to(device) if W_early is not None else None,
        'late': W_late.to(device) if W_late is not None else None,
    }
    layer_idx = model_wrapper.num_layers - 1
    layer_module = model_wrapper._get_nested_attr(
        model_wrapper._get_arribute_path(layer_idx, "hidden")
    )
    step_counter = [0]  # value of t for the *next* decode-step call

    def hook(_module, _inputs, outputs):
        hidden = outputs[0] if isinstance(outputs, tuple) else outputs
        is_prefill = hidden.shape[1] > 1
        t = 0 if is_prefill else step_counter[0]
        W = Ws[_bucket_for(t, early_range)]
        last = hidden.shape[1] - 1
        if W is not None:
            h = hidden[:, last, :]
            hidden[:, last, :] = h + (W @ h.T).T
        step_counter[0] = t + 1
        if isinstance(outputs, tuple):
            return (hidden,) + outputs[1:]
        return hidden

    handle = layer_module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
