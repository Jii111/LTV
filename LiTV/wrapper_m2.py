"""
Simplified M2 / Adaptive Task Vector wrapper (final-layer only).

M2 (constant Δ):
- Collect anchor queries, run ICL (demo+q) and zero-shot (q) to get last hidden label states.
- Δ = h_ICL(label) - h_zero(label); store mean Δ̄.
- Inference: add Δ̄ to zero-shot label hidden before the LM head.

M2-Adaptive (linear Δ):
- Same anchors, build H0 = h_zero(label), Y = Δ.
- Closed-form ridge: W = (H0^T H0 + λI)^-1 H0^T Y (implemented in utils_method).
- Inference: h_TV = h_zero(label) + W @ h_zero(label).

Hooking: we inject at the last decoder layer output (label position only) via forward hooks.
"""

from contextlib import contextmanager
from typing import List, Optional

import torch
from tqdm import tqdm

import global_vars as gv
import utils_method as um
from wrapper import Qwen3Wrapper


class M2Wrapper(Qwen3Wrapper):
    """Final-layer constant task vector."""

    def __init__(self, model, tokenizer, model_config, device):
        super().__init__(model, tokenizer, model_config, device)
        self.task_vector: Optional[torch.Tensor] = None  # Δ̄ (d,)

    @torch.no_grad()
    def extract_m2_task_vector(
        self,
        demo: str,
        train_queries: List[str],
        tokenizer,
        batch_size: int = 8,
        verbose: bool = True,
    ) -> torch.Tensor:
        """
        Compute mean Δ = h_L^{ICL}(label) − h_L^{Zero}(label) over anchor queries.
        """
        deltas = []
        batches = [
            train_queries[i : i + batch_size]
            for i in range(0, len(train_queries), batch_size)
        ]

        for batch_idx, batch in enumerate(tqdm(batches, desc="Extract M2", disable=not verbose)):
            # ICL hidden
            icl_inputs = [demo + q for q in batch]
            icl_tokens = tokenizer(
                icl_inputs, return_tensors="pt", padding=True, truncation=False
            ).to(self.device)
            icl_out = self.model(
                **icl_tokens, output_hidden_states=True, use_cache=False
            )
            icl_hidden = icl_out.hidden_states[-1]  # CausalLMOutputWithPast
            icl_label = um.extract_label_position_hidden(icl_hidden, icl_tokens["attention_mask"])

            # Zero-shot hidden
            zero_tokens = tokenizer(batch, return_tensors="pt", padding=True, truncation=False).to(
                self.device
            )
            zero_out = self.model(
                **zero_tokens, output_hidden_states=True, use_cache=False
            )
            zero_hidden = zero_out.hidden_states[-1]
            zero_label = um.extract_label_position_hidden(zero_hidden, zero_tokens["attention_mask"])

            delta = icl_label - zero_label  # (b, d)
            deltas.append(delta.cpu())

            del icl_tokens, icl_out, icl_hidden, icl_label
            del zero_tokens, zero_out, zero_hidden, zero_label
            torch.cuda.empty_cache()

        all_delta = torch.cat(deltas, dim=0) if deltas else torch.zeros(1, self.embed_dim)
        mean_delta = all_delta.mean(dim=0)
        self.task_vector = mean_delta
        return mean_delta

    @contextmanager
    @torch.no_grad()
    def inject_m2_task_vector(self, task_vector: Optional[torch.Tensor] = None):
        """
        Add constant Δ to label hidden at the last decoder layer.
        Assumes gv.ATTN_MASK_END is set before forward (Evaluator sets this).
        """
        if task_vector is None:
            task_vector = self.task_vector
        if task_vector is None:
            raise ValueError("No task vector available. Run extract_m2_task_vector first.")

        delta = task_vector.to(self.device)
        layer_idx = self.num_layers - 1
        layer_module = self._get_nested_attr(self._get_arribute_path(layer_idx, "hidden"))
        handles = []

        def hook(module, inputs, outputs):
            hidden = outputs[0] if isinstance(outputs, tuple) else outputs
            batch_size = hidden.size(0)
            label_pos = gv.ATTN_MASK_END.to(hidden.device)
            hidden[torch.arange(batch_size, device=hidden.device), label_pos, :] += delta
            if isinstance(outputs, tuple):
                return (hidden,) + outputs[1:]
            return hidden

        handles.append(layer_module.register_forward_hook(hook))
        try:
            yield
        finally:
            for h in handles:
                h.remove()


class M2AdaptiveWrapper(M2Wrapper):
    """Final-layer adaptive task vector with closed-form ridge regression."""

    def __init__(self, model, tokenizer, model_config, device):
        super().__init__(model, tokenizer, model_config, device)
        self.adaptive_matrix: Optional[torch.Tensor] = None  # W (d, d)

    @torch.no_grad()
    def extract_adaptive_task_vector(
        self,
        demo: str,
        train_queries: List[str],
        tokenizer,
        batch_size: int = 8,
        ridge_lambda: float = 0.01,
        verbose: bool = True,
    ) -> torch.Tensor:
        """
        Fit W so that Δ ≈ W · h_zero(label) using closed-form ridge regression.
        """
        feature_list = []
        target_list = []

        batches = [
            train_queries[i : i + batch_size]
            for i in range(0, len(train_queries), batch_size)
        ]

        for batch in tqdm(batches, desc="Extract M2-Adaptive", disable=not verbose):
            # ICL hidden
            icl_inputs = [demo + q for q in batch]
            icl_tokens = tokenizer(
                icl_inputs, return_tensors="pt", padding=True, truncation=False
            ).to(self.device)
            icl_out = self.model(
                **icl_tokens, output_hidden_states=True, use_cache=False
            )
            icl_label = um.extract_label_position_hidden(
                icl_out.hidden_states[-1], icl_tokens["attention_mask"]
            )

            # Zero hidden
            zero_tokens = tokenizer(batch, return_tensors="pt", padding=True, truncation=False).to(
                self.device
            )
            zero_out = self.model(
                **zero_tokens, output_hidden_states=True, use_cache=False
            )
            zero_label = um.extract_label_position_hidden(
                zero_out.hidden_states[-1], zero_tokens["attention_mask"]
            )

            delta = icl_label - zero_label  # (b, d)
            feature_list.append(zero_label.cpu())
            target_list.append(delta.cpu())

            del icl_tokens, icl_out, zero_tokens, zero_out, icl_label, zero_label, delta
            torch.cuda.empty_cache()

        if not feature_list:
            raise ValueError("No features collected for adaptive task vector.")

        features = torch.cat(feature_list, dim=0).T  # (d, n)
        targets = torch.cat(target_list, dim=0).T    # (d, n)
        W = um.ridge_regression(
            targets=targets,
            features=features,
            lambda_reg=ridge_lambda,
            device=self.device,
        )  # (d, d)
        self.adaptive_matrix = W.cpu()
        return self.adaptive_matrix

    @contextmanager
    @torch.no_grad()
    def inject_adaptive_task_vector(self, adaptive_matrix: Optional[torch.Tensor] = None):
        """
        Add W · h_zero(label) to label hidden at the last decoder layer.
        """
        if adaptive_matrix is None:
            adaptive_matrix = self.adaptive_matrix
        if adaptive_matrix is None:
            raise ValueError("No adaptive task vector. Run extract_adaptive_task_vector first.")

        W = adaptive_matrix.to(self.device)
        layer_idx = self.num_layers - 1
        layer_module = self._get_nested_attr(self._get_arribute_path(layer_idx, "hidden"))
        handles = []
        self.injected_deltas = []

        def hook(module, inputs, outputs):
            hidden = outputs[0] if isinstance(outputs, tuple) else outputs
            batch_size = hidden.size(0)
            label_pos = gv.ATTN_MASK_END.to(hidden.device)
            for i in range(batch_size):
                h_label = hidden[i, label_pos[i], :]
                delta = torch.matmul(W, h_label)
                hidden[i, label_pos[i], :] = h_label + delta
                
                self.injected_deltas.append({
                    "batch_idx": i,
                    "position": label_pos[i].item(),
                    "delta": delta.detach().cpu(),
                })

            if isinstance(outputs, tuple):
                return (hidden,) + outputs[1:]
            return hidden

        handles.append(layer_module.register_forward_hook(hook))
        try:
            yield
        finally:
            for h in handles:
                h.remove()
