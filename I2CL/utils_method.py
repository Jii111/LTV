"""
Utility functions for M2 and M3 methods

Provides:
1. Ridge regression for task vector computation
2. Label position extraction
3. Demo position computation
4. Feature normalization utilities
"""

from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F


def ridge_regression(
        targets: torch.Tensor,
        features: torch.Tensor,
        lambda_reg: float = 0.01,
        device: str = 'cpu'
) -> torch.Tensor:
    """
    Solve ridge regression: B = Y @ X.T @ inv(X @ X.T + λI)

    Uses torch.linalg.solve for numerical stability instead of explicit inverse.

    Args:
        targets: (d, n) - target matrix (e.g., Δ_ℓ for all samples)
        features: (m, n) - feature matrix (e.g., φ_ℓ for all samples)
        lambda_reg: ridge regularization coefficient
        device: computation device

    Returns:
        B: (d, m) - regression coefficient matrix

    Example:
        >>> targets = torch.randn(768, 100)  # 100 samples, 768-dim targets
        >>> features = torch.randn(512, 100)  # 100 samples, 512-dim features
        >>> B = ridge_regression(targets, features, lambda_reg=0.01)
        >>> B.shape
        torch.Size([768, 512])
    """
    # Store original dtype
    original_dtype = targets.dtype

    # Convert to float32 for linalg operations (BFloat16 not supported)
    targets = targets.to(device=device, dtype=torch.float32)
    features = features.to(device=device, dtype=torch.float32)

    d, n1 = targets.shape
    m, n2 = features.shape

    assert n1 == n2, f"Sample size mismatch: targets {n1} vs features {n2}"

    # Compute X @ X.T + λI (m, m)
    gram_matrix = features @ features.T
    ridge_term = lambda_reg * torch.eye(m, device=device, dtype=torch.float32)
    gram_matrix_reg = gram_matrix + ridge_term

    # Compute Y @ X.T (d, m)
    target_feature = targets @ features.T

    # Solve (X @ X.T + λI) @ B.T = (Y @ X.T).T for B.T
    # More numerically stable than computing inverse
    try:
        B_T = torch.linalg.solve(gram_matrix_reg, target_feature.T)
        B = B_T.T
    except RuntimeError as e:
        # Fallback to lstsq if solve fails
        print(f"Warning: linalg.solve failed ({e}). Using lstsq.")
        B_T = torch.linalg.lstsq(gram_matrix_reg, target_feature.T).solution
        B = B_T.T

    # Convert back to original dtype
    B = B.to(dtype=original_dtype)

    return B


def extract_label_position_hidden(
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
) -> torch.Tensor:
    """
    Extract hidden states at label token positions (last non-padding token).

    Args:
        hidden_states: (batch, seq_len, hidden_dim)
        attention_mask: (batch, seq_len) - 1 for real tokens, 0 for padding

    Returns:
        label_hiddens: (batch, hidden_dim) - hidden states at label positions
    """
    batch_size = hidden_states.size(0)

    # Find last non-padding position for each sample
    # attention_mask.sum(dim=1) gives total length
    # Subtract 1 to get 0-indexed position
    label_positions = attention_mask.sum(dim=1) - 1

    # Extract hidden states at these positions
    label_hiddens = hidden_states[torch.arange(batch_size), label_positions]

    return label_hiddens


def compute_demo_end_positions(
        demo_text: str,
        queries: List[str],
        tokenizer
) -> torch.Tensor:
    """
    Compute the end position of demo tokens for each [demo, query] concatenation.

    Args:
        demo_text: demonstration string
        queries: list of query strings
        tokenizer: HuggingFace tokenizer

    Returns:
        demo_end_positions: (batch,) tensor with demo end indices
    """
    # Tokenize demo once
    demo_tokens = tokenizer(demo_text, return_tensors='pt', add_special_tokens=False)
    demo_length = demo_tokens['input_ids'].shape[1]

    # For each query, demo comes first, so demo_end = demo_length
    # (assuming left padding is handled separately)
    batch_size = len(queries)
    demo_end_positions = torch.full((batch_size,), demo_length, dtype=torch.long)

    return demo_end_positions


def normalize_features(
        features: torch.Tensor,
        method: str = 'l2'
) -> torch.Tensor:
    """
    Normalize feature vectors.

    Args:
        features: (n, m) or (m,) tensor
        method: 'l2' or 'standardize'

    Returns:
        normalized_features: same shape as input
    """
    if method == 'l2':
        # L2 normalization along feature dimension
        if features.dim() == 1:
            norm = torch.norm(features, p=2)
            return features / (norm + 1e-8)
        else:
            norm = torch.norm(features, p=2, dim=1, keepdim=True)
            return features / (norm + 1e-8)

    elif method == 'standardize':
        # Standardization: (x - mean) / std
        if features.dim() == 1:
            mean = features.mean()
            std = features.std()
            return (features - mean) / (std + 1e-8)
        else:
            mean = features.mean(dim=0, keepdim=True)
            std = features.std(dim=0, keepdim=True)
            return (features - mean) / (std + 1e-8)

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def compute_reconstruction_error(
        targets: torch.Tensor,
        predictions: torch.Tensor,
        metric: str = 'mse'
) -> float:
    """
    Compute reconstruction error between targets and predictions.

    Args:
        targets: (n, d) or (d,)
        predictions: (n, d) or (d,)
        metric: 'mse' or 'cosine'

    Returns:
        error: scalar error value
    """
    if metric == 'mse':
        # Mean squared error
        error = F.mse_loss(predictions, targets, reduction='mean').item()

    elif metric == 'cosine':
        # Average cosine similarity (1 - similarity as error)
        if targets.dim() == 1:
            cos_sim = F.cosine_similarity(predictions.unsqueeze(0), targets.unsqueeze(0)).item()
        else:
            cos_sim = F.cosine_similarity(predictions, targets, dim=1).mean().item()
        error = 1.0 - cos_sim

    else:
        raise ValueError(f"Unknown metric: {metric}")

    return error


def batch_iterator(
        data: List,
        batch_size: int,
        shuffle: bool = False
):
    """
    Create batches from a list of data.

    Args:
        data: list of items
        batch_size: batch size
        shuffle: whether to shuffle before batching

    Yields:
        batches: list of items in each batch
    """
    indices = list(range(len(data)))

    if shuffle:
        import random
        random.shuffle(indices)

    for start_idx in range(0, len(data), batch_size):
        end_idx = min(start_idx + batch_size, len(data))
        batch_indices = indices[start_idx:end_idx]
        yield [data[i] for i in batch_indices]


def save_task_vectors(
        task_vectors: Dict[int, torch.Tensor],
        save_path: str
):
    """
    Save task vectors to disk.

    Args:
        task_vectors: {layer_idx: vector or matrix}
        save_path: path to save file (.pt)
    """
    # Convert to CPU and save
    cpu_vectors = {k: v.cpu() for k, v in task_vectors.items()}
    torch.save(cpu_vectors, save_path)
    print(f"Task vectors saved to {save_path}")


def load_task_vectors(
        load_path: str,
        device: str = 'cpu'
) -> Dict[int, torch.Tensor]:
    """
    Load task vectors from disk.

    Args:
        load_path: path to saved file (.pt)
        device: device to load to

    Returns:
        task_vectors: {layer_idx: vector or matrix}
    """
    task_vectors = torch.load(load_path, map_location=device)
    print(f"Task vectors loaded from {load_path}")
    return task_vectors


def analyze_task_vectors(
        task_vectors: Dict[int, torch.Tensor],
        method: str = 'M2'
) -> Dict[str, any]:
    """
    Analyze task vector statistics.

    Args:
        task_vectors: {layer_idx: vector or matrix}
        method: 'M2' or 'M3'

    Returns:
        stats: dictionary with statistics
    """
    stats = {
        'num_layers': len(task_vectors),
        'layer_indices': list(task_vectors.keys()),
    }

    if method == 'M2':
        # For M2, task vectors are (d,) vectors
        norms = [torch.norm(v, p=2).item() for v in task_vectors.values()]
        stats['vector_norms'] = norms
        stats['mean_norm'] = np.mean(norms)
        stats['std_norm'] = np.std(norms)

    elif method == 'M3':
        # For M3, task vectors are (d, m) matrices
        shapes = [v.shape for v in task_vectors.values()]
        frobenius_norms = [torch.norm(v.float(), p='fro').item() for v in task_vectors.values()]
        spectral_norms = [torch.linalg.matrix_norm(v.float(), ord=2).item() for v in task_vectors.values()]

        stats['matrix_shapes'] = shapes
        stats['frobenius_norms'] = frobenius_norms
        stats['spectral_norms'] = spectral_norms
        stats['mean_frobenius_norm'] = np.mean(frobenius_norms)
        stats['mean_spectral_norm'] = np.mean(spectral_norms)

    return stats


def visualize_task_vector_norms(
        task_vectors: Dict[int, torch.Tensor],
        save_path: str,
        method: str = 'M2'
):
    """
    Visualize task vector norms across layers.

    Args:
        task_vectors: {layer_idx: vector or matrix}
        save_path: path to save figure
        method: 'M2' or 'M3'
    """
    import matplotlib.pyplot as plt

    layer_indices = sorted(task_vectors.keys())

    if method == 'M2':
        norms = [torch.norm(task_vectors[i].float(), p=2).item() for i in layer_indices]
        ylabel = 'L2 Norm'
        title = 'M2 Task Vector Norms Across Layers'
    else:
        norms = [torch.norm(task_vectors[i].float(), p='fro').item() for i in layer_indices]
        ylabel = 'Frobenius Norm'
        title = 'M3 Task Vector (Matrix) Norms Across Layers'

    plt.figure(figsize=(10, 5))
    plt.plot(layer_indices, norms, marker='o', linewidth=2, markersize=6)
    plt.xlabel('Layer Index', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Visualization saved to {save_path}")
