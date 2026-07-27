"""
LoReFT-style baseline — Wu et al., "ReFT: Representation Finetuning for
Language Models" (NeurIPS 2024; github.com/stanfordnlp/pyreft).

Requested by reviewer EHiD: an EXPLICITLY OPTIMIZED task-vector intervention
to isolate how much of LTV's advantage comes from the linear, closed-form
formulation. LoReFT edits the residual-stream hidden h through a learned
low-rank projection (verbatim from pyreft's LoreftIntervention, linear act):

    Phi(h) = h + R^T ( (W h + b) - R h ),      R in R^{r x d}, rows orthonormal
                                               W in R^{r x d}, b in R^r

i.e. inside the r-dim subspace spanned by R, the component of h is REPLACED
by a learned affine source W h + b; the orthogonal complement is untouched.
Unlike Learned-TV's constant theta, the edit is query-dependent (a function
of h), and unlike our LTV it is gradient-optimized rather than closed-form.

Harness: identical to the Learned-TV row so the two differ ONLY in the
operator — same anchor queries, same objectives, same injection site
(input of the chosen decoder layer(s), label position only), same batch-1
step budget, same val-based selection with patience. Deviations from the
original LoReFT recipe (documented in Baseline_Exp_Details.md):
  - single position (the label token) instead of prefix+suffix position
    sets, because every method in this suite intervenes at the label
    position only;
  - layer set is configurable ('mid' default to match Learned-TV; the
    original intervenes at several layers);
  - objectives: 'ce' (their supervised LM-loss analogue on the gold label's
    first token) and 'lmse' (our label-free eq.-11 proxy), instead of
    seq2seq LM loss over a completion.

Optimizer follows the pyreft training setup: AdamW, linear warmup then
linear decay to zero (HF default schedule), dropout configurable (their
tasks use 0.0-0.05; we default 0.0 so the learned intervention is
deterministic at eval).
"""

import random
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from tqdm import tqdm

import global_vars as gv
from core.utils import utils_method as um
from core.wrapper_base import Qwen3Wrapper
from core.wrapper_learned_tv import label_first_token_ids


class LowRankRotateLayer(torch.nn.Module):
    """A linear transformation with orthogonal initialization (pyreft)."""

    def __init__(self, n: int, m: int, init_orth: bool = True):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(n, m), requires_grad=True)
        if init_orth:
            torch.nn.init.orthogonal_(self.weight)

    def forward(self, x):
        return torch.matmul(x.to(self.weight.dtype), self.weight)


class LoreftIntervention(torch.nn.Module):
    """pyreft's LoreftIntervention with the default linear activation:
    Phi(h) = h + R^T((W h + b) - R h). Computed in the parameter dtype
    (fp32) and cast back to the input dtype."""

    def __init__(self, embed_dim: int, rank: int, dropout: float = 0.0):
        super().__init__()
        rotate_layer = LowRankRotateLayer(embed_dim, rank, init_orth=True)
        # Keeps R's columns exactly orthonormal throughout training; AdamW
        # updates the unconstrained parametrization underneath.
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.learned_source = torch.nn.Linear(embed_dim, rank)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, base: torch.Tensor) -> torch.Tensor:
        R = self.rotate_layer.weight                     # (d, r), orthonormal columns
        base32 = base.to(R.dtype)
        out = base32 + torch.matmul(
            self.learned_source(base32) - torch.matmul(base32, R), R.T
        )
        return self.dropout(out).to(base.dtype)


class LoReFTWrapper(Qwen3Wrapper):
    """Gradient-trained low-rank representation intervention (LoReFT-style),
    applied at the input of the chosen decoder layer(s), label position only."""

    def __init__(self, model, tokenizer, model_config, device) -> None:
        super().__init__(model, tokenizer, model_config, device)
        self.interventions: Optional[torch.nn.ModuleList] = None
        self.layer_idxs: Optional[List[int]] = None

    # ------------------------------------------------------------------
    def resolve_layers(self, layers: Union[str, int, Sequence[int]]) -> List[int]:
        if layers == 'mid':
            return [self.num_layers // 2]
        if layers == 'last':
            return [self.num_layers - 1]
        if layers == 'all':
            return list(range(self.num_layers))
        if isinstance(layers, (int, str)):
            return [int(layers)]
        return [int(l) for l in layers]

    def _layer_module(self, layer_idx: int):
        return self._get_nested_attr(self._get_arribute_path(layer_idx, "hidden"))

    # ------------------------------------------------------------------
    @torch.no_grad()
    def collect_icl_hidden(
        self,
        demo: str,
        queries: List[str],
        tokenizer,
        batch_size: int = 1,
        verbose: bool = True,
    ) -> torch.Tensor:
        """ICL teacher targets for the 'lmse' objective: final-layer hidden at
        the label position of (demo + query), float32 on CPU. Same collection
        as LTV extraction, so both methods see the identical teacher."""
        targets = []
        batches = [queries[i:i + batch_size] for i in range(0, len(queries), batch_size)]
        for batch in tqdm(batches, desc="LoReFT: ICL targets", disable=not verbose):
            tokens = tokenizer([demo + q for q in batch], return_tensors="pt",
                               padding=True, truncation=False).to(self.device)
            out = self.model(**tokens, output_hidden_states=True, use_cache=False)
            h = um.extract_label_position_hidden(out.hidden_states[-1], tokens["attention_mask"])
            targets.append(h.detach().float().cpu())
            del tokens, out, h
            torch.cuda.empty_cache()
        return torch.cat(targets, dim=0)

    # ------------------------------------------------------------------
    def train_loreft(
        self,
        queries: List[str],
        labels: List[int],
        tokenizer,
        model_name: str,
        options: List[str],
        loss: str = 'lmse',
        icl_targets: Optional[torch.Tensor] = None,
        layers: Union[str, int, Sequence[int]] = 'mid',
        rank: int = 4,
        lr: float = 9e-4,
        weight_decay: float = 0.0,
        dropout: float = 0.0,
        warmup_ratio: float = 0.1,
        epochs: int = 8,
        samples_per_epoch: int = 100,
        patience: int = 2,
        val_ratio: float = 0.2,
        seed: int = 0,
        verbose: bool = True,
    ) -> Tuple[torch.nn.ModuleList, Dict[str, Any]]:
        """Train the intervention(s) with the LLM frozen.
        Returns (interventions, info)."""
        assert loss in ('ce', 'lmse'), loss
        if loss == 'lmse':
            assert icl_targets is not None and len(icl_targets) == len(queries), \
                "'lmse' needs one ICL teacher hidden per anchor query"

        layer_idxs = self.resolve_layers(layers)
        modules = [self._layer_module(l) for l in layer_idxs]
        device = self.device

        for p in self.model.parameters():
            p.requires_grad_(False)

        rng = random.Random(seed)
        order = list(range(len(queries)))
        rng.shuffle(order)
        n_val = max(1, int(len(order) * val_ratio))
        val_idx, train_idx = order[:n_val], order[n_val:]

        gold_ids = label_first_token_ids(tokenizer, options, model_name)

        torch.manual_seed(seed)
        interventions = torch.nn.ModuleList([
            LoreftIntervention(self.embed_dim, rank, dropout=dropout)
            for _ in layer_idxs
        ]).to(device=device, dtype=torch.float32)
        init_R = [iv.rotate_layer.weight.detach().clone() for iv in interventions]
        init_W = [iv.learned_source.weight.detach().clone() for iv in interventions]

        # The rotation params MUST be excluded from weight decay: torch's
        # orthogonal parametrization (householder map) computes
        # Q = householder_product(...) * X.diagonal().int(), relying on the
        # underlying diagonal staying exactly +-1 (its gradient is zero).
        # Decoupled weight decay shrinks it to +-0.9999, the .int() cast
        # truncates that to 0, and R silently collapses to all-zeros
        # (verified on torch 2.7).
        rotate_params, source_params = [], []
        for iv in interventions:
            rotate_params += list(iv.rotate_layer.parameters())
            source_params += list(iv.learned_source.parameters())
        optimizer = torch.optim.AdamW([
            {'params': rotate_params, 'weight_decay': 0.0},
            {'params': source_params, 'weight_decay': weight_decay},
        ], lr=lr)
        actual_steps = max(1, epochs * min(samples_per_epoch, max(len(train_idx), 1)))
        warmup_steps = int(actual_steps * warmup_ratio)

        def lr_lambda(s: int) -> float:
            if warmup_steps > 0 and s < warmup_steps:
                return (s + 1) / warmup_steps
            return max(0.0, 1.0 - (s - warmup_steps) / max(1, actual_steps - warmup_steps))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        def make_pre_hook(interv):
            def pre_hook(mod, args, kwargs):
                hidden = args[0] if args else kwargs['hidden_states']
                # batch size 1, no padding -> label position is the last token.
                # Rebuilt via cat rather than in-place assignment: Phi's backward
                # saves its input slice, and an index_put into the same tensor
                # would bump its autograd version (unlike Learned-TV's addition,
                # whose backward needs no saved input).
                new_last = interv(hidden[:, -1:, :])
                hidden = torch.cat([hidden[:, :-1, :], new_last], dim=1)
                if args:
                    return (hidden,) + args[1:], kwargs
                kwargs['hidden_states'] = hidden
                return args, kwargs
            return pre_hook

        handles = [m.register_forward_pre_hook(make_pre_hook(iv), with_kwargs=True)
                   for m, iv in zip(modules, interventions)]

        def forward_one(idx: int, need_hidden: bool):
            tokens = tokenizer(queries[idx], return_tensors="pt").to(device)
            out = self.model(**tokens, use_cache=False, output_hidden_states=need_hidden)
            return out

        def sample_loss(idx: int) -> torch.Tensor:
            if loss == 'lmse':
                out = forward_one(idx, need_hidden=True)
                h = out.hidden_states[-1][0, -1, :].float()
                return (h - icl_targets[idx].to(device)).pow(2).sum()
            out = forward_one(idx, need_hidden=False)
            logit = out.logits[0, -1, :].float().unsqueeze(0)
            gold = torch.tensor([gold_ids[labels[idx]]], device=device)
            return F.cross_entropy(logit, gold)

        @torch.no_grad()
        def val_score() -> float:
            """Higher is better for both objectives."""
            interventions.eval()
            try:
                if loss == 'lmse':
                    total = 0.0
                    for idx in val_idx:
                        out = forward_one(idx, need_hidden=True)
                        h = out.hidden_states[-1][0, -1, :].float()
                        total += (h - icl_targets[idx].to(device)).pow(2).sum().item()
                    return -total / len(val_idx)
                correct = 0
                for idx in val_idx:
                    out = forward_one(idx, need_hidden=False)
                    if out.logits[0, -1, :].argmax().item() == gold_ids[labels[idx]]:
                        correct += 1
                return correct / len(val_idx)
            finally:
                interventions.train()

        def snapshot() -> List[Dict[str, torch.Tensor]]:
            return [{k: v.detach().clone() for k, v in iv.state_dict().items()}
                    for iv in interventions]

        best_score, best_state, no_improve, epochs_ran = None, None, 0, 0
        curve = []
        interventions.train()
        try:
            with torch.enable_grad():
                for epoch in range(epochs):
                    epochs_ran = epoch + 1
                    picks = rng.sample(train_idx, min(samples_per_epoch, len(train_idx)))
                    running = 0.0
                    pbar = tqdm(picks, desc=f"LoReFT/{loss} ep{epoch + 1}/{epochs}",
                                leave=False, disable=not verbose)
                    for step, idx in enumerate(pbar):
                        optimizer.zero_grad(set_to_none=True)
                        loss_t = sample_loss(idx)
                        loss_t.backward()
                        if epoch == 0 and step == 0:
                            # Tripwire: a broken hook/graph would leave the
                            # intervention untrained and silently ship a
                            # zero-shot-like baseline.
                            gsum = sum(p.grad.abs().sum().item()
                                       for p in interventions.parameters()
                                       if p.grad is not None)
                            assert gsum > 0, \
                                "no gradient reached the LoReFT parameters — hook not in the graph"
                        optimizer.step()
                        scheduler.step()
                        running += loss_t.item()
                        pbar.set_postfix(loss=f"{running / (step + 1):.3f}")
                        if (step + 1) % 32 == 0:
                            torch.cuda.empty_cache()
                    train_loss = running / len(picks)
                    score = val_score()
                    with torch.no_grad():
                        source_l2 = sum(iv.learned_source.weight.norm().item() ** 2
                                        for iv in interventions) ** 0.5
                        source_drift = sum(
                            (iv.learned_source.weight - W0).norm().item() ** 2
                            for iv, W0 in zip(interventions, init_W)) ** 0.5
                        rotate_drift = sum(
                            (iv.rotate_layer.weight - R0).norm().item() ** 2
                            for iv, R0 in zip(interventions, init_R)) ** 0.5
                        ortho_err = max(
                            (iv.rotate_layer.weight.T @ iv.rotate_layer.weight
                             - torch.eye(rank, device=device)).abs().max().item()
                            for iv in interventions)
                    assert ortho_err < 1e-3, \
                        f"R lost orthonormality (err {ortho_err}) — weight decay reached the rotation params?"
                    curve.append({'epoch': epoch + 1, 'train_loss': train_loss,
                                  'val_score': score,
                                  'lr': scheduler.get_last_lr()[0],
                                  # the two *_drift fields are ||param - init||_F
                                  # across sites — the evidence that both halves
                                  # of the operator actually trained (source_l2
                                  # is an absolute norm, ~1.16 already at init)
                                  'source_l2': source_l2,
                                  'source_drift': source_drift,
                                  'rotate_drift': rotate_drift,
                                  'ortho_err': ortho_err})
                    if verbose:
                        print(f"[LoReFT/{loss}] epoch {epoch + 1}/{epochs} "
                              f"train_loss {train_loss:.4f} val_score {score:.4f}")
                    if best_score is None or score > best_score:
                        best_score, no_improve = score, 0
                        best_state = snapshot()
                    else:
                        no_improve += 1
                        if no_improve >= patience:
                            break
        finally:
            for h in handles:
                h.remove()
            torch.cuda.empty_cache()

        if best_state is not None:
            for iv, st in zip(interventions, best_state):
                iv.load_state_dict(st)
        interventions.eval()

        self.interventions = interventions
        self.layer_idxs = layer_idxs
        n_params = sum(p.numel() for iv in interventions
                       for p in (iv.rotate_layer.weight, iv.learned_source.weight,
                                 iv.learned_source.bias))
        info = {'layer_idxs': layer_idxs, 'loss': loss, 'rank': rank, 'lr': lr,
                'dropout': dropout, 'warmup_ratio': warmup_ratio,
                'selection': 'best val score, patience early stop (same rule as Learned-TV row)',
                'epochs_ran': epochs_ran, 'best_val_score': best_score, 'curve': curve,
                'num_train': len(train_idx), 'num_val': len(val_idx),
                'param_numel': int(n_params), 'total_steps': actual_steps,
                'warmup_steps': warmup_steps,
                'final_lr': scheduler.get_last_lr()[0],
                # not scale-limited: reachable travel vs init entry scales of
                # ~1/sqrt(d) for both R (orthonormal) and W (kaiming).
                # max_travel is the peak-lr bound (steps x lr); scheduled_travel
                # is the tighter bound under the warmup+decay schedule actually
                # used (~half of max) — still >> the ~0.016 init scale.
                'max_travel_per_entry': actual_steps * lr,
                'scheduled_travel_per_entry': lr * sum(
                    lr_lambda(s) for s in range(actual_steps))}
        return interventions, info

    # ------------------------------------------------------------------
    @contextmanager
    @torch.no_grad()
    def inject_loreft(
        self,
        interventions: Optional[torch.nn.ModuleList] = None,
        layer_idxs: Optional[List[int]] = None,
    ) -> Iterator[None]:
        """Apply Phi to the residual stream entering each chosen decoder layer
        at the label position (gv.ATTN_MASK_END, set per batch by Evaluator)."""
        interventions = self.interventions if interventions is None else interventions
        layer_idxs = self.layer_idxs if layer_idxs is None else layer_idxs
        if interventions is None or layer_idxs is None:
            raise ValueError("No trained LoReFT intervention. Run train_loreft first.")

        interventions.eval()

        def make_pre_hook(interv):
            def pre_hook(mod, args, kwargs):
                hidden = args[0] if args else kwargs['hidden_states']
                pos = gv.ATTN_MASK_END.to(hidden.device)
                rows = torch.arange(hidden.size(0), device=hidden.device)
                hidden[rows, pos, :] = interv(hidden[rows, pos, :])
                if args:
                    return (hidden,) + args[1:], kwargs
                kwargs['hidden_states'] = hidden
                return args, kwargs
            return pre_hook

        handles = [self._layer_module(l).register_forward_pre_hook(
                       make_pre_hook(iv), with_kwargs=True)
                   for l, iv in zip(layer_idxs, interventions)]
        try:
            yield
        finally:
            for h in handles:
                h.remove()
