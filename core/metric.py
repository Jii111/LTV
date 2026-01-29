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

def plot_d_NTP(
    datasets_map: Dict[str, Dict[str, List[float]]],
    dataset_order: List[str],
    save_path: str,
) -> None:
    """Plot and save log-scale d_NTP bar chart."""
    methods = ["Ours"]
    legend_labels = {"Ours": "LTV(Ours)"}
    xtick_labels = ["AGNews", "DBPedia", "HateSpeech18", "MR", "SST-2", "SST-5", "Subj", "TREC"]
    colors = {"Ours": "#4C78A8"}

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
            color=colors[m],
            alpha=0.9,
            edgecolor="black",
            linewidth=0.5,
            label=legend_labels[m],
        )

    plt.yscale("log")
    plt.xticks(x, xtick_labels, fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlim(x[0] - 0.6, x[-1] + 0.6)
    plt.tight_layout(rect=[0, 0.15, 1, 1])

    plt.legend(
        ncol=5,
        fontsize=18,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.8),
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()
    print("Saved bar plot in ", save_path)
