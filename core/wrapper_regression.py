"""
Regression-specific LTV wrapper.

Overrides injection context managers to use prefill-only injection at the
last token position, without relying on gv.ATTN_MASK_END (which is set
externally by the classification Evaluator and not available here).

The extraction logic (extract_adaptive_task_vector, extract_m2_task_vector)
is inherited unchanged from LTVWrapper/M2Wrapper.
"""

from contextlib import contextmanager
from typing import Iterator, Optional

import torch

from core.wrapper_ltv import LTVWrapper


class RegressionLTVWrapper(LTVWrapper):
    """LTV wrapper adapted for regression (generation-based, prefill-only injection)."""

    @contextmanager
    @torch.no_grad()
    def inject_adaptive_task_vector(
        self, adaptive_matrix: Optional[torch.Tensor] = None
    ) -> Iterator[None]:
        """Inject W @ h at the last token position, prefill only (seq_len > 1)."""
        if adaptive_matrix is None:
            adaptive_matrix = self.adaptive_matrix
        if adaptive_matrix is None:
            raise ValueError("No adaptive task vector. Run extract_adaptive_task_vector first.")

        W = adaptive_matrix.to(self.device)
        layer_idx = self.num_layers - 1
        layer_module = self._get_nested_attr(self._get_arribute_path(layer_idx, "hidden"))

        def hook(module, inputs, outputs):
            hidden = outputs[0] if isinstance(outputs, tuple) else outputs
            if hidden.shape[1] <= 1:
                return outputs
            last = hidden.shape[1] - 1
            for i in range(hidden.shape[0]):
                h = hidden[i, last, :]
                hidden[i, last, :] = h + torch.matmul(W, h)
            if isinstance(outputs, tuple):
                return (hidden,) + outputs[1:]
            return hidden

        handle = layer_module.register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    @contextmanager
    @torch.no_grad()
    def inject_m2_task_vector(
        self, task_vector: Optional[torch.Tensor] = None
    ) -> Iterator[None]:
        """Inject constant delta at the last token position, prefill only."""
        if task_vector is None:
            task_vector = self.task_vector
        if task_vector is None:
            raise ValueError("No task vector. Run extract_m2_task_vector first.")

        delta = task_vector.to(self.device)
        layer_idx = self.num_layers - 1
        layer_module = self._get_nested_attr(self._get_arribute_path(layer_idx, "hidden"))

        def hook(module, inputs, outputs):
            hidden = outputs[0] if isinstance(outputs, tuple) else outputs
            if hidden.shape[1] <= 1:
                return outputs
            hidden[:, -1, :] = hidden[:, -1, :] + delta
            if isinstance(outputs, tuple):
                return (hidden,) + outputs[1:]
            return hidden

        handle = layer_module.register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()
