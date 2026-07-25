"""
Learned Task Vector baseline — Yang et al., "Task Vectors, Learned Not
Extracted: Performance Gains and Mechanistic Insights" (ICLR 2026;
arXiv:2509.24169, github.com/HLYang2001/Learned_TV).

A single d-dimensional vector theta is trained by gradient descent with the
LLM entirely frozen, and added to the residual stream at the INPUT of one
decoder layer (their best configuration: the middle layer) at the last-token
(label) position. Their recipe, verified against the released code: random
init U(-init_scale, init_scale), AdamW(weight_decay=0.01), batch size 1, up
to `epochs` x `samples_per_epoch` steps, early stopping (patience on a
held-out anchor split). Paper text uses lr = 1e-3 (released code: 5e-3).

Two training objectives are supported:
- 'ce'   paper-faithful: cross-entropy of the gold label's first token on
         zero-shot prompts. Consumes gold labels for the anchor queries
         (an extra supervision budget our LTV does not use).
- 'lmse' label-free variant: minimizes our eq.-11 proxy
         || h_icl(x) - h_final(x; theta) ||^2 against ICL teacher hiddens
         at the final layer — the same information budget as our LTV
         (demonstrations + unlabeled anchors), but gradient-trained.
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
    """First token id per class label, matching Evaluator's prediction strategy
    ('first semantic' for Llama-2, leading-space token otherwise)."""
    if "llama-2" in model_name.lower():
        return [tokenizer.encode(opt, add_special_tokens=False)[0] for opt in options]
    return [tokenizer.encode(" " + opt, add_special_tokens=False)[0] for opt in options]


class LearnedTVWrapper(Qwen3Wrapper):
    """Gradient-trained single-vector baseline (residual-stream addition at
    the input of one decoder layer, label position only)."""

    def __init__(self, model, tokenizer, model_config, device) -> None:
        super().__init__(model, tokenizer, model_config, device)
        self.theta: Optional[torch.Tensor] = None
        self.layer_idx: Optional[int] = None

    # ------------------------------------------------------------------
    def resolve_layer(self, layer) -> int:
        if layer == 'mid':
            return self.num_layers // 2
        if layer == 'last':
            return self.num_layers - 1
        return int(layer)

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
        for batch in tqdm(batches, desc="Learned-TV: ICL targets", disable=not verbose):
            tokens = tokenizer([demo + q for q in batch], return_tensors="pt",
                               padding=True, truncation=False).to(self.device)
            out = self.model(**tokens, output_hidden_states=True, use_cache=False)
            h = um.extract_label_position_hidden(out.hidden_states[-1], tokens["attention_mask"])
            targets.append(h.detach().float().cpu())
            del tokens, out, h
            torch.cuda.empty_cache()
        return torch.cat(targets, dim=0)

    # ------------------------------------------------------------------
    def train_learned_tv(
        self,
        queries: List[str],
        labels: List[int],
        tokenizer,
        model_name: str,
        options: List[str],
        loss: str = 'lmse',
        icl_targets: Optional[torch.Tensor] = None,
        layer='mid',
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        epochs: int = 10,
        samples_per_epoch: int = 100,
        patience: int = 2,
        val_ratio: float = 0.2,
        init_scale: float = 0.1,
        seed: int = 0,
        verbose: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Train theta with the LLM frozen. Returns (theta_cpu, info)."""
        assert loss in ('ce', 'lmse'), loss
        if loss == 'lmse':
            assert icl_targets is not None and len(icl_targets) == len(queries), \
                "'lmse' needs one ICL teacher hidden per anchor query"

        layer_idx = self.resolve_layer(layer)
        module = self._layer_module(layer_idx)
        device = self.device

        for p in self.model.parameters():
            p.requires_grad_(False)

        rng = random.Random(seed)
        order = list(range(len(queries)))
        rng.shuffle(order)
        n_val = max(1, int(len(order) * val_ratio))
        val_idx, train_idx = order[:n_val], order[n_val:]

        gold_ids = label_first_token_ids(tokenizer, options, model_name)

        gen = torch.Generator(device='cpu').manual_seed(seed)
        theta = ((torch.rand(self.embed_dim, generator=gen) * 2 - 1) * init_scale)
        theta = theta.to(device=device, dtype=torch.float32).requires_grad_(True)
        optimizer = torch.optim.AdamW([theta], lr=lr, weight_decay=weight_decay)
        # Their trainer steps a linear-decay schedule once per sample
        # (train_ltv.py:110-115, hidden_states.py:1330).
        total_steps = max(1, epochs * min(samples_per_epoch, max(len(train_idx), 1)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda s: max(0.0, 1.0 - s / total_steps))

        def pre_hook(mod, args, kwargs):
            hidden = args[0] if args else kwargs['hidden_states']
            # batch size 1, no padding -> label position is the last token
            hidden = hidden.clone()
            hidden[:, -1, :] = hidden[:, -1, :] + theta.to(hidden.dtype)
            if args:
                return (hidden,) + args[1:], kwargs
            kwargs['hidden_states'] = hidden
            return args, kwargs

        handle = module.register_forward_pre_hook(pre_hook, with_kwargs=True)

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

        best_score, best_theta, no_improve, epochs_ran = None, None, 0, 0
        curve = []
        try:
            with torch.enable_grad():
                for epoch in range(epochs):
                    epochs_ran = epoch + 1
                    picks = rng.sample(train_idx, min(samples_per_epoch, len(train_idx)))
                    running = 0.0
                    pbar = tqdm(picks, desc=f"Learned-TV/{loss} ep{epoch + 1}/{epochs}",
                                leave=False, disable=not verbose)
                    for step, idx in enumerate(pbar):
                        optimizer.zero_grad(set_to_none=True)
                        loss_t = sample_loss(idx)
                        loss_t.backward()
                        if epoch == 0 and step == 0:
                            # Tripwire: a broken hook/graph would leave theta
                            # untrained and silently ship a zero-shot baseline.
                            assert theta.grad is not None and theta.grad.abs().sum().item() > 0, \
                                "no gradient reached theta — injection hook not in the graph"
                        optimizer.step()
                        scheduler.step()
                        running += loss_t.item()
                        pbar.set_postfix(loss=f"{running / (step + 1):.3f}")
                        if (step + 1) % 32 == 0:
                            torch.cuda.empty_cache()
                    train_loss = running / len(picks)
                    score = val_score()
                    curve.append({'epoch': epoch + 1, 'train_loss': train_loss,
                                  'val_score': score})
                    if verbose:
                        print(f"[Learned-TV/{loss}] epoch {epoch + 1}/{epochs} "
                              f"train_loss {train_loss:.4f} val_score {score:.4f}")
                    if best_score is None or score > best_score:
                        best_score, no_improve = score, 0
                        best_theta = theta.detach().clone()
                    else:
                        no_improve += 1
                        if no_improve >= patience:
                            break
        finally:
            handle.remove()
            torch.cuda.empty_cache()

        if best_theta is None:
            best_theta = theta.detach().clone()
        self.theta = best_theta.cpu()
        self.layer_idx = layer_idx
        info = {'layer_idx': layer_idx, 'loss': loss, 'lr': lr,
                'selection': 'best val accuracy, patience early stop (paper rule)',
                'epochs_ran': epochs_ran, 'best_val_score': best_score, 'curve': curve,
                'num_train': len(train_idx), 'num_val': len(val_idx)}
        return self.theta, info

    # ------------------------------------------------------------------
    @contextmanager
    @torch.no_grad()
    def inject_learned_tv(
        self,
        theta: Optional[torch.Tensor] = None,
        layer_idx: Optional[int] = None,
    ) -> Iterator[None]:
        """Add theta to the residual stream entering decoder layer `layer_idx`
        at the label position (gv.ATTN_MASK_END, set per batch by Evaluator)."""
        theta = self.theta if theta is None else theta
        layer_idx = self.layer_idx if layer_idx is None else layer_idx
        if theta is None or layer_idx is None:
            raise ValueError("No learned task vector. Run train_learned_tv first.")

        vec = theta.to(self.device)
        module = self._layer_module(layer_idx)

        def pre_hook(mod, args, kwargs):
            hidden = args[0] if args else kwargs['hidden_states']
            pos = gv.ATTN_MASK_END.to(hidden.device)
            hidden[torch.arange(hidden.size(0), device=hidden.device), pos, :] += vec.to(hidden.dtype)
            if args:
                return (hidden,) + args[1:], kwargs
            kwargs['hidden_states'] = hidden
            return args, kwargs

        handle = module.register_forward_pre_hook(pre_hook, with_kwargs=True)
        try:
            yield
        finally:
            handle.remove()
