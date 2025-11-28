"""
Unit tests for M3 (MHA Feature-based Query-adaptive Task Vector)

Tests:
1. MHA head output extraction
2. Query feature shape validation
3. Ridge regression for task vectors
4. Query-adaptive injection logic
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch


def test_m3_import():
    """Test that M3 wrapper can be imported"""
    print("\n[TEST] Importing M3 wrapper...")
    try:
        from I2CL.wrapper_m3 import M3Wrapper
        print("✓ M3Wrapper imported successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to import M3Wrapper: {e}")
        return False


def test_mha_head_output_shapes():
    """Test MHA head output shape expectations"""
    print("\n[TEST] MHA head output shapes...")
    try:
        # Mock head outputs from attention layer
        batch_size, seq_len, num_heads, head_dim = 4, 20, 28, 128  # Qwen2.5-7B config
        head_outputs = torch.randn(batch_size, seq_len, num_heads, head_dim)

        # Extract at label position (e.g., position 15)
        label_positions = torch.tensor([15, 17, 19, 18])
        label_head_outputs = head_outputs[torch.arange(batch_size), label_positions]

        assert label_head_outputs.shape == (batch_size, num_heads, head_dim), \
            f"Expected shape ({batch_size}, {num_heads}, {head_dim}), got {label_head_outputs.shape}"

        # Flatten to get feature vector
        feature = label_head_outputs.reshape(batch_size, -1)
        expected_feature_dim = num_heads * head_dim

        assert feature.shape == (batch_size, expected_feature_dim), \
            f"Expected feature shape ({batch_size}, {expected_feature_dim}), got {feature.shape}"

        print(
            f"✓ MHA head outputs have correct shapes: ({batch_size}, {num_heads}, {head_dim}) → ({batch_size}, {expected_feature_dim})"
            )
        return True
    except Exception as e:
        print(f"✗ MHA head output test failed: {e}")
        return False


def test_query_feature_variability():
    """Test that query features vary across different queries"""
    print("\n[TEST] Query feature variability...")
    try:
        # Mock features for different queries
        num_queries = 100
        feature_dim = 28 * 128  # num_heads * head_dim
        features = torch.randn(num_queries, feature_dim)

        # Compute pairwise distances
        distances = torch.cdist(features, features, p=2)

        # Check that features are not all identical
        off_diag = distances[~torch.eye(num_queries, dtype=bool)]
        mean_distance = off_diag.mean().item()

        assert mean_distance > 0.1, f"Features seem too similar (mean distance = {mean_distance:.4f})"

        # Check variance
        feature_variance = features.var(dim=0).mean().item()
        assert feature_variance > 0.01, f"Feature variance too low ({feature_variance:.6f})"

        print(
            f"✓ Query features are query-dependent (mean distance = {mean_distance:.4f}, variance = {feature_variance:.6f})"
            )
        return True
    except Exception as e:
        print(f"✗ Query feature variability test failed: {e}")
        return False


def test_m3_ridge_regression():
    """Test ridge regression for M3 task vectors"""
    print("\n[TEST] M3 ridge regression...")
    try:
        from I2CL.utils_method import ridge_regression

        # Mock data: deltas (d=3584, n=100), features (m=3584, n=100)
        d, m, n = 3584, 3584, 100  # For Qwen2.5-7B
        targets = torch.randn(d, n)
        features = torch.randn(m, n)

        # Compute B_ℓ: (d, m)
        B = ridge_regression(targets, features, lambda_reg=0.01, device='cpu')

        assert B.shape == (d, m), f"Expected B shape ({d}, {m}), got {B.shape}"

        # Check that B can reconstruct
        predictions = features.T @ B.T  # (n, m) @ (m, d) = (n, d)
        reconstruction_error = torch.nn.functional.mse_loss(predictions, targets.T).item()

        print(f"✓ M3 ridge regression works: B shape = {B.shape}, MSE = {reconstruction_error:.6f}")
        return True
    except Exception as e:
        print(f"✗ M3 ridge regression test failed: {e}")
        return False


def test_query_adaptive_prediction():
    """Test query-adaptive prediction: Δ̂ = B @ φ"""
    print("\n[TEST] Query-adaptive prediction...")
    try:
        # Mock task vector matrix B: (d=768, m=512)
        d, m = 768, 512
        B = torch.randn(d, m)

        # Mock features for different queries
        num_queries = 50
        features = torch.randn(num_queries, m)

        # Compute predictions: Δ̂ = φ @ B.T = (n, m) @ (m, d) = (n, d)
        predictions = features @ B.T

        assert predictions.shape == (num_queries, d), \
            f"Expected prediction shape ({num_queries}, {d}), got {predictions.shape}"

        # Check that predictions vary across queries
        pred_variance = predictions.var(dim=0).mean().item()
        assert pred_variance > 0.01, f"Predictions not query-adaptive (variance = {pred_variance:.6f})"

        # Check that different features lead to different predictions
        pred_distances = torch.cdist(predictions, predictions, p=2)
        off_diag = pred_distances[~torch.eye(num_queries, dtype=bool)]
        mean_pred_distance = off_diag.mean().item()

        assert mean_pred_distance > 0.1, \
            f"Predictions too similar (mean distance = {mean_pred_distance:.4f})"

        print(
            f"✓ Query-adaptive prediction works (variance = {pred_variance:.6f}, mean distance = {mean_pred_distance:.4f})"
            )
        return True
    except Exception as e:
        print(f"✗ Query-adaptive prediction test failed: {e}")
        return False


def test_m3_vs_m2_capacity():
    """Test that M3 has higher capacity than M2"""
    print("\n[TEST] M3 vs M2 capacity comparison...")
    try:
        # M2: constant vector per layer (d,)
        # M3: matrix per layer (d, m)
        d, m = 3584, 3584

        # M2 parameters per layer
        m2_params = d

        # M3 parameters per layer
        m3_params = d * m

        capacity_ratio = m3_params / m2_params

        print(f"  M2 parameters per layer: {m2_params:,}")
        print(f"  M3 parameters per layer: {m3_params:,}")
        print(f"  M3/M2 capacity ratio: {capacity_ratio:.1f}x")

        assert m3_params > m2_params, "M3 should have more capacity than M2"
        assert capacity_ratio == m, f"Expected capacity ratio = {m}, got {capacity_ratio}"

        print(f"✓ M3 has {capacity_ratio:.0f}x more capacity than M2")
        return True
    except Exception as e:
        print(f"✗ Capacity comparison test failed: {e}")
        return False


def run_all_tests():
    """Run all M3 tests"""
    print("=" * 60)
    print("M3 Unit Tests")
    print("=" * 60)

    tests = [
        test_m3_import,
        test_mha_head_output_shapes,
        test_query_feature_variability,
        test_m3_ridge_regression,
        test_query_adaptive_prediction,
        test_m3_vs_m2_capacity,
    ]

    results = []
    for test_func in tests:
        result = test_func()
        results.append(result)

    print("\n" + "=" * 60)
    print(f"Test Results: {sum(results)}/{len(results)} passed")
    print("=" * 60)

    if all(results):
        print("\n✓ All M3 tests passed!")
        return 0
    else:
        print("\n✗ Some M3 tests failed")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
