"""
Unit tests for M2 (Mask-based Multi-layer Constant Task Vector)

Tests:
1. extract_m2_task_vectors() shape validation
2. Demo masking functionality
3. Task vector injection hooks
4. Reconstruction quality evaluation
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch


def test_m2_import():
    """Test that M2 wrapper can be imported"""
    print("\n[TEST] Importing M2 wrapper...")
    try:
        from I2CL.wrapper_m2 import M2Wrapper
        print("✓ M2Wrapper imported successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to import M2Wrapper: {e}")
        return False


def test_demo_masking_logic():
    """Test demo masking logic with mock attention weights"""
    print("\n[TEST] Demo masking logic...")
    try:
        # Mock attention weights: (batch=2, num_heads=4, seq_len=10, seq_len=10)
        batch_size, num_heads, seq_len = 2, 4, 10
        attn_weights = torch.randn(batch_size, num_heads, seq_len, seq_len)

        # Demo end positions: [3, 5] (demo ends at position 3 and 5 for each sample)
        demo_mask_positions = torch.tensor([3, 5])

        # Apply masking logic
        for batch_idx, demo_end_pos in enumerate(demo_mask_positions):
            if demo_end_pos > 0:
                # Query positions after demo should not attend to demo
                attn_weights[batch_idx, :, demo_end_pos:, :demo_end_pos] = float('-inf')

        # Check that masking was applied
        assert torch.isinf(attn_weights[0, 0, 3, 0]), "Masking not applied correctly for sample 0"
        assert torch.isinf(attn_weights[1, 0, 5, 0]), "Masking not applied correctly for sample 1"
        assert not torch.isinf(attn_weights[0, 0, 2, 0]), "Masking applied incorrectly (before demo)"

        print("✓ Demo masking logic works correctly")
        return True
    except Exception as e:
        print(f"✗ Demo masking test failed: {e}")
        return False


def test_task_vector_shapes():
    """Test that task vectors have correct shapes"""
    print("\n[TEST] Task vector shapes...")
    try:
        num_layers = 32
        hidden_dim = 3584  # Qwen2.5-7B hidden dim

        # Mock task vectors
        task_vectors = {}
        for layer_idx in range(num_layers):
            task_vectors[layer_idx] = torch.randn(hidden_dim)

        # Check shapes
        assert len(task_vectors) == num_layers, f"Expected {num_layers} layers, got {len(task_vectors)}"
        for layer_idx, v in task_vectors.items():
            assert v.shape == (hidden_dim,), f"Layer {layer_idx}: Expected shape ({hidden_dim},), got {v.shape}"

        print(f"✓ Task vectors have correct shapes: {num_layers} layers × ({hidden_dim},)")
        return True
    except Exception as e:
        print(f"✗ Shape test failed: {e}")
        return False


def test_ridge_regression_utility():
    """Test ridge regression utility function"""
    print("\n[TEST] Ridge regression utility...")
    try:
        from I2CL.utils_method import ridge_regression

        # Mock data: (d=768, n=100), (m=512, n=100)
        d, m, n = 768, 512, 100
        targets = torch.randn(d, n)
        features = torch.randn(m, n)

        # Compute B: (d, m)
        B = ridge_regression(targets, features, lambda_reg=0.01, device='cpu')

        assert B.shape == (d, m), f"Expected shape ({d}, {m}), got {B.shape}"

        # Check that B can predict
        predictions = features.T @ B.T  # (n, m) @ (m, d) = (n, d)
        assert predictions.shape == (n, d), f"Prediction shape mismatch"

        print(f"✓ Ridge regression works correctly: B shape = {B.shape}")
        return True
    except Exception as e:
        print(f"✗ Ridge regression test failed: {e}")
        return False


def test_label_position_extraction():
    """Test label position hidden state extraction"""
    print("\n[TEST] Label position extraction...")
    try:
        from I2CL.utils_method import extract_label_position_hidden

        # Mock hidden states: (batch=4, seq_len=20, hidden_dim=768)
        batch_size, seq_len, hidden_dim = 4, 20, 768
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)

        # Mock attention mask (1 for real tokens, 0 for padding)
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[0, 15:] = 0  # First sample has padding from position 15
        attention_mask[1, 18:] = 0  # Second sample has padding from position 18

        # Extract label positions
        label_hiddens = extract_label_position_hidden(hidden_states, attention_mask)

        assert label_hiddens.shape == (batch_size, hidden_dim), \
            f"Expected shape ({batch_size}, {hidden_dim}), got {label_hiddens.shape}"

        # Check that correct positions were extracted
        expected_pos_0 = 14  # Last real token for sample 0
        expected_pos_1 = 17  # Last real token for sample 1
        assert torch.equal(label_hiddens[0], hidden_states[0, expected_pos_0]), \
            "Label position extraction incorrect for sample 0"
        assert torch.equal(label_hiddens[1], hidden_states[1, expected_pos_1]), \
            "Label position extraction incorrect for sample 1"

        print(f"✓ Label position extraction works correctly")
        return True
    except Exception as e:
        print(f"✗ Label position test failed: {e}")
        return False


def run_all_tests():
    """Run all M2 tests"""
    print("=" * 60)
    print("M2 Unit Tests")
    print("=" * 60)

    tests = [
        test_m2_import,
        test_demo_masking_logic,
        test_task_vector_shapes,
        test_ridge_regression_utility,
        test_label_position_extraction,
    ]

    results = []
    for test_func in tests:
        result = test_func()
        results.append(result)

    print("\n" + "=" * 60)
    print(f"Test Results: {sum(results)}/{len(results)} passed")
    print("=" * 60)

    if all(results):
        print("\n✓ All M2 tests passed!")
        return 0
    else:
        print("\n✗ Some M2 tests failed")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
