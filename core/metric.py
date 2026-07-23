"""Metric utilities for evaluating task-vector extraction quality."""

import torch
import torch.nn.functional as F
from typing import Dict, List, Union
import numpy as np
import matplotlib.pyplot as plt

def compute_d_NTP(
    logits_p: "Union[torch.Tensor, List[torch.Tensor]]",
    logits_q: "Union[torch.Tensor, List[torch.Tensor]]",
    is_qwen: bool = False,
    eps: float = 1e-8,
) -> float:
    """
    Our proposed metric is d_NTP, which measures the next-token distribution gap
    between ICL and task-vector. It is defined as
        dNTP(f; Z) = E_x[ KL( P_icl(.|x, Z) || P_tv(.|x, f(Z)) ) ].
    Lower d_NTP indicates P_tv aligns more closely with P_icl.
    """
    if isinstance(logits_p, list):
        logits_p = torch.cat(logits_p, dim=0)
    if isinstance(logits_q, list):
        logits_q = torch.cat(logits_q, dim=0)

    if logits_p.numel() == 0 or logits_q.numel() == 0:
        return 0.0

    if is_qwen:
        logits_p = logits_p.float()
        logits_q = logits_q.float()

    log_p = F.log_softmax(logits_p, dim=-1)
    log_q = F.log_softmax(logits_q, dim=-1)
    p = log_p.exp()

    kl = (p * (log_p - log_q)).sum(dim=-1)
    return kl.mean().item()

def compute_L_mse(
    hidden_icl: "Union[torch.Tensor, List[torch.Tensor]]",
    hidden_tv: "Union[torch.Tensor, List[torch.Tensor]]",
) -> Dict[str, float]:
    """
    Proxy objective L_MSE (paper eq. 11), reported as a metric (cf. Table 8):
        L_MSE(f) = E_x[ || h_icl(x) - h_tv(x) ||_2^2 ],
    where h_icl / h_tv are the final-layer hidden states at the label (last
    prompt token) position under ICL and TV inference, respectively.

    Returns both the eq.-11 value (squared L2 norm summed over hidden dim,
    averaged over queries) and a per-dimension variant for cross-model
    comparison (divided by hidden size d).
    """
    if isinstance(hidden_icl, list):
        hidden_icl = torch.cat(hidden_icl, dim=0)
    if isinstance(hidden_tv, list):
        hidden_tv = torch.cat(hidden_tv, dim=0)

    if hidden_icl.numel() == 0 or hidden_tv.numel() == 0:
        return {"L_mse": 0.0, "L_mse_per_dim": 0.0}

    assert hidden_icl.shape == hidden_tv.shape, (
        f"Shape mismatch: {tuple(hidden_icl.shape)} vs {tuple(hidden_tv.shape)}"
    )

    sq_diff = (hidden_icl.float() - hidden_tv.float()).pow(2)
    l_mse = sq_diff.sum(dim=-1).mean().item()
    l_mse_per_dim = sq_diff.mean().item()
    return {"L_mse": l_mse, "L_mse_per_dim": l_mse_per_dim}


def plot_d_NTP(
    datasets_map: Dict[str, Dict[str, List[float]]],
    save_path: str,
) -> None:
    """Plot and save log-scale d_NTP bar chart."""
    if not datasets_map:
        print("No data to plot.")
        return

    dataset_order = list(datasets_map.keys())
    methods = sorted({m for ds in datasets_map.values() for m in ds.keys()}) or ["Ours"]
    legend_labels = {}
    xtick_labels = dataset_order
    colors = {"Ours": "#4C78A8", "LTV": "#4C78A8"}

    x = np.arange(len(dataset_order))
    width = 0.5
    plt.figure(figsize=(20, 3.5))

    for i, m in enumerate(methods):
        vals = []
        for ds in dataset_order:
            arr = datasets_map.get(ds, {}).get(m, [])

            clean_arr = []
            for v in arr:
                try:
                    clean_arr.append(float(v))
                except Exception:
                    continue

            vals.append(np.mean(clean_arr) if len(clean_arr) > 0 else np.nan)

        plt.bar(
            x + (i - 0) * width,
            vals,
            width=width,
            color=colors.get(m, "#4C78A8"),
            alpha=0.9,
            edgecolor="black",
            linewidth=0.5,
            label=legend_labels.get(m, m),
        )

    plt.yscale("log")
    plt.title("Comparison of d_NTP", fontsize=18)
    plt.xticks(x, xtick_labels, fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlim(x[0] - 0.6, x[-1] + 0.6)
    plt.legend(fontsize=16, loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()
    return
