"""
Cross-model LTV wrapper.

Splits the extraction phase so h_ICL and h_ZS can come from different models.

Scenario 1: h_ICL (large), h_ZS (large) → W fitted on large-model space → inject into small model
Scenario 2: h_ICL (large), h_ZS (small) → W fitted on cross-model space → inject into small model

Also provides LM head label vector extraction for cross-model similarity analysis.
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Any, Dict, List, Optional

from core.utils import utils_method as um
from core.wrapper_ltv import LTVWrapper


class CrossModelLTVWrapper(LTVWrapper):
    """LTV wrapper with separated extraction steps for cross-model experiments."""

    @torch.no_grad()
    def extract_icl_hidden(
        self,
        demo: str,
        train_queries: List[str],
        tokenizer,
        batch_size: int = 1,
        verbose: bool = True,
    ) -> torch.Tensor:
        """Extract h_ICL (last-layer, label-position hidden state) from this model.

        Returns:
            h_icl: (n, d) tensor on CPU
        """
        icl_list = []
        batches = [train_queries[i:i + batch_size] for i in range(0, len(train_queries), batch_size)]

        for batch in tqdm(batches, desc="Extract h_ICL", disable=not verbose):
            icl_inputs = [demo + q for q in batch]
            icl_tokens = tokenizer(
                icl_inputs, return_tensors="pt", padding=True, truncation=False
            ).to(self.device)
            icl_out = self.model(**icl_tokens, output_hidden_states=True, use_cache=False)
            icl_label = um.extract_label_position_hidden(
                icl_out.hidden_states[-1], icl_tokens["attention_mask"]
            )
            icl_list.append(icl_label.cpu())
            del icl_tokens, icl_out, icl_label
            torch.cuda.empty_cache()

        return torch.cat(icl_list, dim=0)  # (n, d)

    @torch.no_grad()
    def extract_zs_hidden(
        self,
        train_queries: List[str],
        tokenizer,
        batch_size: int = 1,
        verbose: bool = True,
    ) -> torch.Tensor:
        """Extract h_ZS (last-layer, label-position hidden state) from this model.

        Returns:
            h_zs: (n, d) tensor on CPU
        """
        zs_list = []
        batches = [train_queries[i:i + batch_size] for i in range(0, len(train_queries), batch_size)]

        for batch in tqdm(batches, desc="Extract h_ZS", disable=not verbose):
            zero_tokens = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=False
            ).to(self.device)
            zero_out = self.model(**zero_tokens, output_hidden_states=True, use_cache=False)
            zero_label = um.extract_label_position_hidden(
                zero_out.hidden_states[-1], zero_tokens["attention_mask"]
            )
            zs_list.append(zero_label.cpu())
            del zero_tokens, zero_out, zero_label
            torch.cuda.empty_cache()

        return torch.cat(zs_list, dim=0)  # (n, d)

    def fit_adaptive_matrix(
        self,
        h_icl: torch.Tensor,
        h_zs: torch.Tensor,
        ridge_lambda: float,
    ) -> torch.Tensor:
        """Fit W such that (h_icl - h_zs) ≈ W @ h_zs.

        h_icl and h_zs can come from different models as long as hidden_dim matches.

        Args:
            h_icl: (n, d) - ICL hidden states (any model)
            h_zs:  (n, d) - zero-shot hidden states (any model)
            ridge_lambda: ridge regularization coefficient

        Returns:
            W: (d, d) adaptive matrix, stored in self.adaptive_matrix and returned on CPU
        """
        delta = (h_icl - h_zs).T   # (d, n)
        features = h_zs.T           # (d, n)
        W = um.ridge_regression(
            targets=delta,
            features=features,
            lambda_reg=ridge_lambda,
            device=self.device,
        )
        self.adaptive_matrix = W.cpu()
        return self.adaptive_matrix

    def get_lm_head_matrix(self) -> torch.Tensor:
        """Return full lm_head weight matrix (vocab_size, hidden_dim) on CPU as float32."""
        return self.model.lm_head.weight.detach().cpu().float()

    @torch.no_grad()
    def get_lm_head_label_vectors(
        self,
        dataset: Any,
        tokenizer,
    ) -> Dict[str, torch.Tensor]:
        """Extract LM head weight vectors for each label token.

        Since these are model parameters (not activations), no inference is needed.

        Returns:
            {label_text: (hidden_dim,) float32 tensor on CPU}
        """
        lm_head_weight = self.model.lm_head.weight  # (vocab_size, hidden_dim)
        options = dataset.get_dmonstration_template()['options']
        label_vectors = {}
        for label_text in options:
            token_id = tokenizer.encode(label_text, add_special_tokens=False)[0]
            label_vectors[label_text] = lm_head_weight[token_id].detach().cpu().float()
        return label_vectors


def compare_lm_head_label_vectors(
    large_vectors: Dict[str, torch.Tensor],
    small_vectors: Dict[str, torch.Tensor],
) -> Dict[str, Any]:
    """Compute cosine similarity between large and small model LM head label vectors.

    Args:
        large_vectors: {label_text: (d_large,)} from large model
        small_vectors: {label_text: (d_small,)} from small model

    Returns:
        {label_text: cosine_similarity (float)}, plus 'mean' key
    """
    results = {}
    similarities = []
    for label_text in large_vectors:
        if label_text not in small_vectors:
            continue
        v_large = large_vectors[label_text]
        v_small = small_vectors[label_text]
        cos_sim = F.cosine_similarity(v_large.unsqueeze(0), v_small.unsqueeze(0)).item()
        results[label_text] = cos_sim
        similarities.append(cos_sim)
    if similarities:
        results['mean'] = sum(similarities) / len(similarities)
    return results
