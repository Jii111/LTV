"""
Learnable Task Vector baseline — Saglam et al., "Learning Task
Representations from In-Context Learning" (ACL Findings 2025;
arXiv:2502.05390, github.com/baturaysaglam/ICL-task-repr).

Faithful reimplementation of the paper's LANGUAGE variant, verified against
the released code (language/src/LTV_utils/*.py, language/src/train.py):

- Parameter: a single (n_layers x n_heads) weight matrix Phi, randn init
  (ltv.py:17). The LLM is frozen; only Phi is trained.
- Per-layer basis P[l]: the mean, over clean ICL prompts, of layer l's
  attention output at the last token — i.e. the o_proj OUTPUT at position
  -1, averaged (extract_utils.py get_attn_out projects the mean per-head
  INPUT once through o_proj; since o_proj is bias-free this equals the mean
  o_proj OUTPUT, which we capture directly).
- Injected vector v[l] = P[l] ⊙ repeat_interleave(Phi[l], head_dim): each
  head_dim block of the resid_dim basis is scaled by one scalar Phi[l, h]
  (ltv.forward:20-30). This is the language variant's block-wise scaling —
  NOT the regression code's per-head projection + sum.
- Injection: v[l] added to the OUTPUT of EVERY decoder layer at the
  last-token (label) position (intervention_utils.py:16-24), at train and
  inference alike.

Objectives (one switch):
- 'ce'   paper-faithful: gold-label first-token cross-entropy on
         label-SHUFFLED k-shot ICL prompts (corrupted demos force the
         injected vector to carry the task; train.py:92-95, data.py:53-67).
- 'lmse' label-free variant (OUR addition, not in the paper): the eq.-11
         proxy ||h_icl - h(Phi)||^2 on zero-shot prompts against ICL
         teacher hiddens — the same information budget as our LTV.

Selection FOLLOWS the released code: a fixed step count with the
lowest-training-loss checkpoint kept, and no early stopping. In train.py the
name `lowest_val_loss` is the current training batch's loss (there is no
validation set anywhere), and the `early_stoppage_tolerance` branch only
writes an extra checkpoint — it never breaks the loop, so all `n_iter` steps
always run. We compare the epoch-mean training loss rather than a single
step's because we run batch 1, where per-step loss is far noisier than their
batch of 32. The held-out slice carved out by `val_ratio` is used ONLY to log
a convergence diagnostic and never influences selection.

Deviations from the released training loop, kept intentional and matched to
our LTV protocol (documented so the rebuttal caption is accurate):
- basis computed once from our standard demonstration + 256 anchor queries
  and cached, rather than re-derived every step from a fresh sample of 100
  clean ICL prompts (train.py:90 `sample_attn_out(..., batch_size=100)`).
  This is where most of their compute goes — 2000 x (100 + 32) forward
  passes — and it is the deviation that actually shrinks our cost;
- budget: `epochs x samples_per_epoch` batch-1 steps in place of their
  2000 iterations x batch 32. Note their step count is NOT what makes the
  method work: Adam bounds per-entry travel by ~lr per step, so even their
  own budget moves Phi only 2000 x 5e-5 = 0.10 against a randn init of
  std 1. Phi's scale is set by the init, not by the optimizer.
"""

import random
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

import global_vars as gv
from core.utils import utils_method as um
from core.wrapper_base import Qwen3Wrapper


def label_first_token_ids(tokenizer, options: List[str], model_name: str) -> List[int]:
    """First token id per class label, matching Evaluator's prediction
    strategy ('first semantic' for Llama-2, leading-space token otherwise)."""
    if "llama-2" in model_name.lower():
        return [tokenizer.encode(opt, add_special_tokens=False)[0] for opt in options]
    return [tokenizer.encode(" " + opt, add_special_tokens=False)[0] for opt in options]


def build_shuffled_prompt(pool, query_input, k, rng, separator='\n'):
    """k-shot prompt whose demonstration labels are permuted within the
    prompt (their label-shuffling corruption), followed by the query with no
    answer (prompt_utils.py:410-419; data.py sample_data shuffle_labels)."""
    picks = rng.sample(range(len(pool)), min(k, len(pool)))
    labels = [pool[i][2] for i in picks]
    rng.shuffle(labels)
    lines = []
    for i, lab in zip(picks, labels):
        input_str, ans_str, _ = pool[i]
        lines.append(input_str + ' ' + ans_str[lab])
    return separator.join(lines) + separator + query_input


class LearnableTVWrapper(Qwen3Wrapper):
    """Saglam-style learnable head-mixing task vector (all-layer injection)."""

    def __init__(self, model, tokenizer, model_config, device) -> None:
        super().__init__(model, tokenizer, model_config, device)
        self.phi: Optional[torch.Tensor] = None
        self.basis: Optional[torch.Tensor] = None  # (L, d) fp32 cpu

    # ------------------------------------------------------------------
    def _decoder_layers(self):
        return [self._get_nested_attr(self._get_arribute_path(l, "hidden"))
                for l in range(self.num_layers)]

    def _o_proj_modules(self):
        mods = dict(self.model.named_modules())
        return [mods[f"model.layers.{l}.self_attn.o_proj"] for l in range(self.num_layers)]

    def _n_heads(self) -> int:
        return self.model_config.num_attention_heads

    def _head_dim(self) -> int:
        n_heads = self._n_heads()
        assert self.embed_dim % n_heads == 0
        return self.embed_dim // n_heads

    def compose_vectors(self, basis: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """(L, d) injected vectors: block-wise scaling of the basis by Phi."""
        return basis * phi.repeat_interleave(self._head_dim(), dim=1)

    @staticmethod
    def _empty_cache():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def collect_head_basis(
        self,
        demo: str,
        queries: List[str],
        tokenizer,
        batch_size: int = 1,
        verbose: bool = True,
    ) -> torch.Tensor:
        """Per-layer basis P[l] = mean over ICL prompts of layer l's o_proj
        OUTPUT at the last token. o_proj is bias-free, so this equals the
        paper's project-once-of-mean-input construction. Left padding puts
        the last real token at index -1, so batched capture is safe.
        Computed once and cached (their code recomputes per step; caching is
        exact for our fixed demonstration)."""
        o_projs = self._o_proj_modules()
        sums = [torch.zeros(self.embed_dim, dtype=torch.float64) for _ in range(self.num_layers)]
        count = 0
        captured: Dict[int, torch.Tensor] = {}

        def make_hook(layer_idx):
            def hook(module, inputs, output):
                out = output[0] if isinstance(output, tuple) else output
                captured[layer_idx] = out[:, -1, :].detach().float().cpu()
            return hook

        handles = [m.register_forward_hook(make_hook(l)) for l, m in enumerate(o_projs)]
        try:
            batches = [queries[i:i + batch_size] for i in range(0, len(queries), batch_size)]
            for batch in tqdm(batches, desc="Learnable-TV: head basis", disable=not verbose):
                tokens = tokenizer([demo + q for q in batch], return_tensors="pt",
                                   padding=True, truncation=False).to(self.device)
                self.model(**tokens, use_cache=False)
                for l in range(self.num_layers):
                    sums[l] += captured[l].double().sum(dim=0)
                count += len(batch)
                del tokens
                self._empty_cache()
        finally:
            for h in handles:
                h.remove()

        self.basis = torch.stack([(sums[l] / count).float() for l in range(self.num_layers)], dim=0)
        return self.basis

    @torch.no_grad()
    def collect_icl_hidden(
        self,
        demo: str,
        queries: List[str],
        tokenizer,
        batch_size: int = 1,
        verbose: bool = True,
    ) -> torch.Tensor:
        """ICL teacher targets for the 'lmse' objective: final-layer hidden
        at the label position of (demo + query), float32 on CPU."""
        targets = []
        batches = [queries[i:i + batch_size] for i in range(0, len(queries), batch_size)]
        for batch in tqdm(batches, desc="Learnable-TV: ICL targets", disable=not verbose):
            tokens = tokenizer([demo + q for q in batch], return_tensors="pt",
                               padding=True, truncation=False).to(self.device)
            out = self.model(**tokens, output_hidden_states=True, use_cache=False)
            h = um.extract_label_position_hidden(out.hidden_states[-1], tokens["attention_mask"])
            targets.append(h.detach().float().cpu())
            del tokens, out, h
            self._empty_cache()
        return torch.cat(targets, dim=0)

    # ------------------------------------------------------------------
    def train_learnable_tv(
        self,
        queries: List[str],
        labels: List[int],
        pool: List[Tuple[str, List[str], int]],
        tokenizer,
        model_name: str,
        options: List[str],
        basis: torch.Tensor,
        loss: str = 'ce',
        icl_targets: Optional[torch.Tensor] = None,
        k_shot: int = 10,
        lr: float = 5e-5,
        weight_decay: float = 0.0,
        epochs: int = 10,
        samples_per_epoch: int = 100,
        val_ratio: float = 0.2,
        init: str = 'zero',
        example_separator: str = '\n',
        seed: int = 0,
        verbose: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Train Phi (n_layers x n_heads) with the LLM frozen.

        Selection follows the released code: fixed number of steps, keep the
        lowest-training-loss checkpoint, no early stopping (their
        `lowest_val_loss` is the current training batch loss and
        `early_stoppage_tolerance` never breaks the loop). `val_ratio` carves
        out a held-out slice used ONLY to log a convergence diagnostic.

        init: 'randn' reproduces the repo's Gaussian init (ltv.py:17), std 1.
        This matters more than the step count. Adam moves each entry by at most
        ~lr per step, so total travel is bounded by steps x lr: their own budget
        gives 2000 x 5e-5 = 0.10, i.e. training only perturbs the random draw by
        ~10% and it is the INIT, not the optimizer, that sets Phi's scale.
        Consequently 'zero' init cannot work at their lr (800 x 5e-5 = 0.04, so
        v stays ~0 and the method degenerates to zero-shot) unless lr is raised
        to keep steps x lr ~ O(1). Both are documented deviations; `curve`
        records phi_l2 / phi_shift_l2 per epoch so this is measured, not assumed.
        """
        assert loss in ('ce', 'lmse'), loss
        assert init in ('zero', 'randn'), init
        if loss == 'lmse':
            assert icl_targets is not None and len(icl_targets) == len(queries), \
                "'lmse' needs one ICL teacher hidden per anchor query"

        device = self.device
        n_heads = self._n_heads()
        for p in self.model.parameters():
            p.requires_grad_(False)

        rng = random.Random(seed)
        order = list(range(len(queries)))
        rng.shuffle(order)
        n_val = max(1, int(len(order) * val_ratio))
        val_idx, train_idx = order[:n_val], order[n_val:]

        gold_ids = label_first_token_ids(tokenizer, options, model_name)
        basis_dev = basis.to(device=device, dtype=torch.float32)

        if init == 'randn':
            gen = torch.Generator(device='cpu').manual_seed(seed)
            phi = torch.randn(self.num_layers, n_heads, generator=gen)
        else:
            phi = torch.zeros(self.num_layers, n_heads)
        phi = phi.to(device=device, dtype=torch.float32).requires_grad_(True)
        phi_init = phi.detach().clone()
        optimizer = torch.optim.Adam([phi], lr=lr, weight_decay=weight_decay)

        state: Dict[str, Optional[torch.Tensor]] = {'v': None}

        def make_hook(layer_idx):
            def hook(module, inputs, outputs):
                hidden = outputs[0] if isinstance(outputs, tuple) else outputs
                v = state['v']
                assert v is not None
                hidden = hidden.clone()
                hidden[:, -1, :] = hidden[:, -1, :] + v[layer_idx].to(hidden.dtype)
                if isinstance(outputs, tuple):
                    return (hidden,) + outputs[1:]
                return hidden
            return hook

        handles = [m.register_forward_hook(make_hook(l))
                   for l, m in enumerate(self._decoder_layers())]

        def forward_prompt(prompt: str, need_hidden: bool):
            state['v'] = self.compose_vectors(basis_dev, phi)
            tokens = tokenizer(prompt, return_tensors="pt").to(device)
            return self.model(**tokens, use_cache=False, output_hidden_states=need_hidden)

        def sample_loss(idx: int) -> torch.Tensor:
            if loss == 'lmse':
                out = forward_prompt(queries[idx], need_hidden=True)
                h = out.hidden_states[-1][0, -1, :].float()
                assert icl_targets is not None
                # Per-dim MSE: same optimum as the eq.-11 sum, but a stable
                # gradient scale — the sum (~O(1e4)) overflows fp16 backward
                # through the 32-layer injection. Reported L_mse is computed
                # separately, so this training scale does not affect metrics.
                return (h - icl_targets[idx].to(device)).pow(2).mean()
            prompt = build_shuffled_prompt(pool, queries[idx], k_shot, rng, example_separator)
            out = forward_prompt(prompt, need_hidden=False)
            logit = out.logits[0, -1, :].float().unsqueeze(0)
            gold = torch.tensor([gold_ids[labels[idx]]], device=device)
            return F.cross_entropy(logit, gold)

        @torch.no_grad()
        def val_diagnostic() -> float:
            """Held-out diagnostic ONLY — never used for model selection (the
            paper selects by training loss). For 'ce' this is label-restricted
            accuracy, i.e. the same decision rule as our Evaluator, so the curve
            is comparable to the reported numbers; for 'lmse' it is negative
            per-dim MSE. Higher is better in both cases."""
            if not val_idx:
                return float('nan')
            if loss == 'lmse':
                assert icl_targets is not None
                total = 0.0
                for idx in val_idx:
                    out = forward_prompt(queries[idx], need_hidden=True)
                    h = out.hidden_states[-1][0, -1, :].float()
                    total += (h - icl_targets[idx].to(device)).pow(2).mean().item()
                return -total / len(val_idx)
            correct = 0
            gold_tensor = torch.tensor(gold_ids, device=device)
            for idx in val_idx:
                out = forward_prompt(queries[idx], need_hidden=False)
                label_logits = out.logits[0, -1, :].index_select(0, gold_tensor)
                if label_logits.argmax().item() == labels[idx]:
                    correct += 1
            return correct / len(val_idx)

        # Model selection follows the released code (train.py:114-128): keep the
        # checkpoint with the lowest TRAINING loss and run a fixed number of
        # steps — their `lowest_val_loss` is the current training batch loss and
        # `early_stoppage_tolerance` never breaks the loop. We compare epoch-mean
        # training loss rather than a single step's, because we run batch 1 where
        # per-step loss is far noisier than their batch of 32.
        best_train_loss, best_phi, epochs_ran = None, None, 0
        curve, grad_checked = [], False
        try:
            with torch.enable_grad():
                for epoch in range(epochs):
                    epochs_ran = epoch + 1
                    picks = rng.sample(train_idx, min(samples_per_epoch, len(train_idx)))
                    running, counted = 0.0, 0
                    pbar = tqdm(picks, desc=f"Learnable-TV/{loss} ep{epoch + 1}/{epochs}",
                                leave=False, disable=not verbose)
                    for step, idx in enumerate(pbar):
                        optimizer.zero_grad(set_to_none=True)
                        loss_t = sample_loss(idx)
                        if not torch.isfinite(loss_t):
                            continue  # skip a pathological step rather than poison Adam state
                        loss_t.backward()
                        if not grad_checked:
                            # Tripwire on the first step that actually backwards (NOT
                            # step 0, which may be skipped above): a detached hook
                            # would leave Phi untrained and silently ship zero-shot.
                            assert phi.grad is not None and phi.grad.abs().sum().item() > 0, \
                                "no gradient reached Phi — injection hooks not in the graph"
                            grad_checked = True
                        torch.nn.utils.clip_grad_norm_([phi], max_norm=1.0)
                        optimizer.step()
                        running += loss_t.item(); counted += 1
                        pbar.set_postfix(loss=f"{running / counted:.3f}")
                        if (step + 1) % 32 == 0:
                            self._empty_cache()
                    if counted == 0:
                        # Every step was non-finite: no update landed. Scoring this
                        # epoch would hand it train_loss 0.0 and win selection.
                        if verbose:
                            print(f"[Learnable-TV/{loss}] epoch {epoch + 1}/{epochs} "
                                  f"all {len(picks)} steps non-finite — epoch skipped")
                        continue
                    train_loss = running / counted
                    val_diag = val_diagnostic()
                    phi_d = phi.detach()
                    curve.append({'epoch': epoch + 1, 'train_loss': train_loss,
                                  'val_diagnostic': val_diag,
                                  # Did Phi actually move? Adam bounds travel by
                                  # steps x lr, so compare phi_shift_l2 against the
                                  # scale Phi must reach (init std for 'randn').
                                  'phi_l2': phi_d.norm().item(),
                                  'phi_absmean': phi_d.abs().mean().item(),
                                  'phi_shift_l2': (phi_d - phi_init).norm().item()})
                    if verbose:
                        print(f"[Learnable-TV/{loss}] epoch {epoch + 1}/{epochs} "
                              f"train_loss {train_loss:.4f} val_diag {val_diag:.4f} "
                              f"|Phi| {phi_d.norm().item():.4f} "
                              f"|dPhi| {(phi_d - phi_init).norm().item():.4f}")
                    if best_train_loss is None or train_loss < best_train_loss:
                        best_train_loss = train_loss
                        best_phi = phi.detach().clone()
        finally:
            for h in handles:
                h.remove()
            self._empty_cache()

        if best_phi is None:
            best_phi = phi.detach().clone()
        self.phi = best_phi.cpu()
        info = {'loss': loss, 'lr': lr, 'init': init, 'epochs_ran': epochs_ran,
                'selection': 'lowest_epoch_train_loss (paper rule)',
                'best_train_loss': best_train_loss, 'curve': curve,
                'num_train': len(train_idx), 'num_val_diagnostic': len(val_idx),
                'phi_numel': int(phi.numel()),
                # Adam moves each entry by at most ~lr per step, so this bounds how
                # far Phi can travel from its init. Compare against the scale Phi
                # needs (1.0 under 'randn'); if it is far smaller, the budget —
                # not the step count — is what is limiting the baseline.
                'max_travel_per_entry': epochs * samples_per_epoch * lr,
                'k_shot_shuffled': k_shot if loss == 'ce' else None}
        return self.phi, info

    # ------------------------------------------------------------------
    @contextmanager
    @torch.no_grad()
    def inject_learnable_tv(
        self,
        phi: Optional[torch.Tensor] = None,
        basis: Optional[torch.Tensor] = None,
    ) -> Iterator[None]:
        """Add v[l] to every decoder layer's output at the label position
        (gv.ATTN_MASK_END, set per batch by Evaluator)."""
        phi = self.phi if phi is None else phi
        basis = self.basis if basis is None else basis
        if phi is None or basis is None:
            raise ValueError("No learnable task vector. Run collect_head_basis + train_learnable_tv first.")

        v = self.compose_vectors(basis.float(), phi.float()).to(self.device)

        def make_hook(layer_idx):
            def hook(module, inputs, outputs):
                hidden = outputs[0] if isinstance(outputs, tuple) else outputs
                pos = gv.ATTN_MASK_END.to(hidden.device)
                hidden[torch.arange(hidden.size(0), device=hidden.device), pos, :] += \
                    v[layer_idx].to(hidden.dtype)
                if isinstance(outputs, tuple):
                    return (hidden,) + outputs[1:]
                return hidden
            return hook

        handles = [m.register_forward_hook(make_hook(l))
                   for l, m in enumerate(self._decoder_layers())]
        try:
            yield
        finally:
            for h in handles:
                h.remove()
