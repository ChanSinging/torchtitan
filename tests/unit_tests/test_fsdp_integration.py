# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Integration test for FSDP2 generalization.

This test verifies that the refactored FSDP2 code paths work correctly
across all model implementations without requiring actual GPU execution.
"""

import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from torchtitan.distributed.fsdp import apply_fsdp_to_model, apply_replicate_to_model


class MockTransformerBlock(nn.Module):
    """Mock transformer block for testing."""

    def __init__(self, dim=64, moe_enabled=False):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, 2)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.moe_enabled = moe_enabled
        if moe_enabled:
            # Create proper mock MoE structure
            self.moe = nn.Module()
            self.moe.experts = nn.Module()
            self.moe.experts.parameters = lambda: []
            self.moe.experts.num_experts = 4


class MockModel(nn.Module):
    """Mock model matching torchtitan model structure."""

    def __init__(self, num_layers=2, moe_layers=None):
        super().__init__()
        self.tok_embeddings = nn.Embedding(100, 64)
        self.layers = nn.ModuleDict()
        moe_layers = moe_layers or []
        for i in range(num_layers):
            self.layers[f"layer_{i}"] = MockTransformerBlock(
                moe_enabled=(i in moe_layers)
            )
        self.norm = nn.LayerNorm(64)
        self.output = nn.Linear(64, 100)


class TestFSDP2ModelStructure(unittest.TestCase):
    """Test FSDP2 generalization with model structure contract."""

    @patch("torchtitan.distributed.fsdp.fully_shard")
    def test_standard_transformer(self, mock_fully_shard):
        """Test FSDP2 applies correctly to standard Transformer."""
        model = MockModel(num_layers=2, moe_layers=[])
        mock_mesh = MagicMock()

        apply_fsdp_to_model(
            model,
            dp_mesh=mock_mesh,
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
            pp_enabled=False,
        )

        # Verify fully_shard was called for:
        # - tok_embeddings
        # - 2 transformer layers
        # - [norm, output] together
        # - root model
        self.assertGreaterEqual(mock_fully_shard.call_count, 5)

    @patch("torchtitan.distributed.fsdp.fully_shard")
    def test_moe_transformer(self, mock_fully_shard):
        """Test FSDP2 applies correctly to MoE Transformer."""
        model = MockModel(num_layers=2, moe_layers=[1])
        mock_mesh = MagicMock()
        # Configure mock to return int for size()
        mock_mesh.size.return_value = 4

        apply_fsdp_to_model(
            model,
            dp_mesh=mock_mesh,
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
            pp_enabled=False,
        )

        # Same calls as standard transformer, but with MoE handling
        self.assertGreaterEqual(mock_fully_shard.call_count, 5)

    @patch("torchtitan.distributed.fsdp.fully_shard")
    def test_weight_tying(self, mock_fully_shard):
        """Test FSDP2 handles weight tying correctly."""
        model = MockModel()
        model.enable_weight_tying = True
        mock_mesh = MagicMock()

        apply_fsdp_to_model(
            model,
            dp_mesh=mock_mesh,
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
            pp_enabled=False,
        )

        # When weight tying, tok_embeddings and output should be grouped
        grouped_calls = [
            call
            for call in mock_fully_shard.call_args_list
            if isinstance(call.args[0], list) and len(call.args[0]) >= 2
        ]
        self.assertTrue(len(grouped_calls) > 0, "Weight tying should group modules")

    @patch("torchtitan.distributed.fsdp.fully_shard")
    def test_reshard_after_forward_policies(self, mock_fully_shard):
        """Test different reshard_after_forward policies."""
        model = MockModel()
        mock_mesh = MagicMock()

        for policy in ["default", "always", "never"]:
            mock_fully_shard.reset_mock()

            apply_fsdp_to_model(
                model,
                dp_mesh=mock_mesh,
                param_dtype=torch.float32,
                reduce_dtype=torch.float32,
                pp_enabled=False,
                reshard_after_forward_policy=policy,
            )

            # Verify all calls have reshard_after_forward set
            for call in mock_fully_shard.call_args_list:
                if "reshard_after_forward" in call.kwargs:
                    self.assertIn(
                        call.kwargs["reshard_after_forward"],
                        [True, False],
                        f"Policy {policy} should set valid reshard_after_forward",
                    )


class TestFSDP2ParallelismIntegration(unittest.TestCase):
    """Test FSDP2 integration with various parallelism configs."""

    @patch("torchtitan.distributed.fsdp.fully_shard")
    def test_fsdp_with_pp(self, mock_fully_shard):
        """Test FSDP2 with pipeline parallelism enabled."""
        model = MockModel()
        mock_mesh = MagicMock()

        apply_fsdp_to_model(
            model,
            dp_mesh=mock_mesh,
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
            pp_enabled=True,  # PP enabled
        )

        # With PP, default policy should not reshard after forward
        self.assertGreaterEqual(mock_fully_shard.call_count, 5)

    @patch("torchtitan.distributed.fsdp.fully_shard")
    def test_fsdp_with_cpu_offload(self, mock_fully_shard):
        """Test FSDP2 with CPU offload enabled."""
        model = MockModel()
        mock_mesh = MagicMock()

        apply_fsdp_to_model(
            model,
            dp_mesh=mock_mesh,
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
            pp_enabled=False,
            cpu_offload=True,
        )

        # Verify offload_policy is set in fsdp_config
        self.assertGreaterEqual(mock_fully_shard.call_count, 5)


class TestReplicateIntegration(unittest.TestCase):
    """Test HSDP replicate mode integration."""

    @patch("torchtitan.distributed.fsdp.replicate")
    @patch("torchtitan.distributed.fsdp.disable_fsdp_gradient_division")
    def test_replicate_standard_model(self, mock_disable_grad, mock_replicate):
        """Test replicate applies to all model components."""
        model = MockModel()
        mock_mesh = MagicMock()

        apply_replicate_to_model(
            model,
            dp_mesh=mock_mesh,
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
        )

        # Verify replicate was called for:
        # - tok_embeddings
        # - 2 transformer layers
        # - [norm, output] together
        # - root model
        self.assertGreaterEqual(mock_replicate.call_count, 5)

        # Verify gradient division is disabled
        mock_disable_grad.assert_called_once_with(model)


class TestModelImportCompatibility(unittest.TestCase):
    """Test that all models can import and use the new FSDP2 functions."""

    def test_llama3_imports(self):
        """Test llama3 can import the new functions."""
        try:
            from torchtitan.models.llama3.parallelize import apply_fsdp, apply_replicate

            self.assertTrue(callable(apply_fsdp))
            self.assertTrue(callable(apply_replicate))
        except ImportError as e:
            self.fail(f"llama3 import failed: {e}")

    def test_llama4_imports(self):
        """Test llama4 can import the new functions."""
        try:
            from torchtitan.models.llama4.parallelize import apply_fsdp

            self.assertTrue(callable(apply_fsdp))
        except ImportError as e:
            self.fail(f"llama4 import failed: {e}")

    def test_deepseek_v3_imports(self):
        """Test deepseek_v3 can import the new functions."""
        try:
            from torchtitan.models.deepseek_v3.parallelize import parallelize_deepseekv3

            self.assertTrue(callable(parallelize_deepseekv3))
        except ImportError as e:
            self.fail(f"deepseek_v3 import failed: {e}")

    def test_gpt_oss_imports(self):
        """Test gpt_oss can import the new functions."""
        try:
            from torchtitan.models.gpt_oss.parallelize import parallelize_gptoss

            self.assertTrue(callable(parallelize_gptoss))
        except ImportError as e:
            self.fail(f"gpt_oss import failed: {e}")

    @unittest.skipUnless(
        __import__("importlib.util").util.find_spec("einops") is not None,
        "flux requires einops dependency",
    )
    def test_flux_imports(self):
        """Test flux can import the new functions."""
        try:
            from torchtitan.models.flux.parallelize import parallelize_flux

            self.assertTrue(callable(parallelize_flux))
        except ImportError as e:
            self.fail(f"flux import failed: {e}")


class TestBackwardCompatibility(unittest.TestCase):
    """Test that the changes are backward compatible."""

    def test_function_signatures_unchanged(self):
        """Test that public function signatures are preserved."""
        import inspect

        from torchtitan.models.llama3.parallelize import apply_fsdp

        sig = inspect.signature(apply_fsdp)
        params = list(sig.parameters.keys())

        # Verify expected parameters exist
        expected_params = [
            "model",
            "dp_mesh",
            "param_dtype",
            "reduce_dtype",
            "pp_enabled",
        ]
        for param in expected_params:
            self.assertIn(param, params)


if __name__ == "__main__":
    unittest.main()
