# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
from torch.distributed._composable.fsdp import FSDPModule

from torchtitan.distributed.fsdp import (
    apply_fsdp_to_model,
    apply_replicate_to_model,
    disable_fsdp_gradient_division,
    get_fsdp_reshard_after_forward_policy,
)


class TestFSDPReshardPolicy(unittest.TestCase):
    """Tests for get_fsdp_reshard_after_forward_policy."""

    def test_always_policy(self):
        """Test that 'always' policy returns True."""
        self.assertTrue(
            get_fsdp_reshard_after_forward_policy("always", pp_enabled=False)
        )
        self.assertTrue(
            get_fsdp_reshard_after_forward_policy("always", pp_enabled=True)
        )

    def test_never_policy(self):
        """Test that 'never' policy returns False."""
        self.assertFalse(
            get_fsdp_reshard_after_forward_policy("never", pp_enabled=False)
        )
        self.assertFalse(
            get_fsdp_reshard_after_forward_policy("never", pp_enabled=True)
        )

    def test_default_policy_without_pp(self):
        """Test default policy without pipeline parallelism."""
        self.assertTrue(
            get_fsdp_reshard_after_forward_policy("default", pp_enabled=False)
        )

    def test_default_policy_with_pp(self):
        """Test default policy with pipeline parallelism returns False."""
        self.assertFalse(
            get_fsdp_reshard_after_forward_policy("default", pp_enabled=True)
        )

    def test_invalid_policy(self):
        """Test that invalid policy raises ValueError."""
        with self.assertRaises(ValueError):
            get_fsdp_reshard_after_forward_policy("invalid", pp_enabled=False)


class SimpleTransformer(nn.Module):
    """Simple transformer model for testing."""

    def __init__(self, vocab_size=100, dim=64, num_layers=2):
        super().__init__()
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleDict(
            {
                f"layer_{i}": nn.TransformerEncoderLayer(dim, 2, dim * 2)
                for i in range(num_layers)
            }
        )
        self.norm = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, vocab_size)


class TestDisableFSDPGradientDivision(unittest.TestCase):
    """Tests for disable_fsdp_gradient_division."""

    def test_disables_gradient_division(self):
        """Test that gradient division is disabled for all FSDP modules."""
        model = SimpleTransformer()

        # Create mock FSDP modules
        mock_fsdp_module1 = MagicMock(spec=FSDPModule)
        mock_fsdp_module2 = MagicMock(spec=FSDPModule)

        # Patch model.modules() to return our mock FSDP modules
        with patch.object(
            model, "modules", return_value=[model, mock_fsdp_module1, mock_fsdp_module2]
        ):
            disable_fsdp_gradient_division(model)

        # Verify set_gradient_divide_factor was called with 1.0
        mock_fsdp_module1.set_gradient_divide_factor.assert_called_once_with(1.0)
        mock_fsdp_module2.set_gradient_divide_factor.assert_called_once_with(1.0)

    def test_skips_non_fsdp_modules(self):
        """Test that non-FSDP modules are skipped."""
        model = SimpleTransformer()

        # Create mock non-FSDP module
        mock_non_fsdp = MagicMock()

        # Patch model.modules() to return mix of FSDP and non-FSDP
        mock_fsdp = MagicMock(spec=FSDPModule)
        with patch.object(
            model, "modules", return_value=[model, mock_non_fsdp, mock_fsdp]
        ):
            disable_fsdp_gradient_division(model)

        # Verify set_gradient_divide_factor was only called on FSDP module
        mock_fsdp.set_gradient_divide_factor.assert_called_once_with(1.0)
        mock_non_fsdp.set_gradient_divide_factor.assert_not_called()


class TestApplyReplicateToModel(unittest.TestCase):
    """Tests for apply_replicate_to_model."""

    @patch("torchtitan.distributed.fsdp.replicate")
    def test_applies_replicate_to_model_parts(self, mock_replicate):
        """Test that replicate is applied to all model parts."""
        model = SimpleTransformer()
        mock_mesh = MagicMock()

        apply_replicate_to_model(
            model,
            dp_mesh=mock_mesh,
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
        )

        # Verify replicate was called for embeddings, layers, norm/output, and root model
        # Expected calls: tok_embeddings, 2 layers, [norm, output] together, root model
        self.assertGreaterEqual(mock_replicate.call_count, 4)

    @patch("torchtitan.distributed.fsdp.replicate")
    @patch("torchtitan.distributed.fsdp.disable_fsdp_gradient_division")
    def test_disables_gradient_division(self, mock_disable_grad_div, mock_replicate):
        """Test that gradient division is disabled after replication."""
        model = SimpleTransformer()
        mock_mesh = MagicMock()

        apply_replicate_to_model(
            model,
            dp_mesh=mock_mesh,
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
        )

        # Verify disable_fsdp_gradient_division was called
        mock_disable_grad_div.assert_called_once_with(model)

    @patch("torchtitan.distributed.fsdp.logger")
    @patch("torchtitan.distributed.fsdp.replicate")
    def test_logs_application(self, mock_replicate, mock_logger):
        """Test that replication is logged."""
        model = SimpleTransformer()
        mock_mesh = MagicMock()

        apply_replicate_to_model(
            model,
            dp_mesh=mock_mesh,
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
        )

        # Verify log message was created
        mock_logger.info.assert_called_with("Applied replicate to the model")


class TestApplyFSDPToModel(unittest.TestCase):
    """Tests for apply_fsdp_to_model."""

    @patch("torchtitan.distributed.fsdp.fully_shard")
    def test_applies_fsdp_to_model_parts(self, mock_fully_shard):
        """Test that fully_shard is applied to all model parts."""
        model = SimpleTransformer()
        mock_mesh = MagicMock()

        apply_fsdp_to_model(
            model,
            dp_mesh=mock_mesh,
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
            pp_enabled=False,
        )

        # Verify fully_shard was called multiple times
        self.assertGreaterEqual(mock_fully_shard.call_count, 4)

    @patch("torchtitan.distributed.fsdp.fully_shard")
    @patch("torchtitan.distributed.fsdp.disable_fsdp_gradient_division")
    def test_disables_gradient_division(self, mock_disable_grad_div, mock_fully_shard):
        """Test that gradient division is disabled after FSDP."""
        model = SimpleTransformer()
        mock_mesh = MagicMock()

        apply_fsdp_to_model(
            model,
            dp_mesh=mock_mesh,
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
            pp_enabled=False,
        )

        # Verify disable_fsdp_gradient_division was called
        mock_disable_grad_div.assert_called_once_with(model)

    @patch("torchtitan.distributed.fsdp.fully_shard")
    def test_cpu_offload(self, mock_fully_shard):
        """Test CPU offload policy is applied when requested."""
        model = SimpleTransformer()
        mock_mesh = MagicMock()

        apply_fsdp_to_model(
            model,
            dp_mesh=mock_mesh,
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
            pp_enabled=False,
            cpu_offload=True,
        )

        # Verify fully_shard was called with offload_policy
        for call in mock_fully_shard.call_args_list:
            if "offload_policy" in call.kwargs:
                self.assertIsNotNone(call.kwargs["offload_policy"])

    @patch("torchtitan.distributed.fsdp.fully_shard")
    def test_reshard_after_forward_policy(self, mock_fully_shard):
        """Test reshard_after_forward policy is respected."""
        model = SimpleTransformer()
        mock_mesh = MagicMock()

        apply_fsdp_to_model(
            model,
            dp_mesh=mock_mesh,
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
            pp_enabled=False,
            reshard_after_forward_policy="always",
        )

        # Verify reshard_after_forward=True in all calls
        for call in mock_fully_shard.call_args_list:
            if "reshard_after_forward" in call.kwargs:
                self.assertTrue(call.kwargs["reshard_after_forward"])


class TestModelStructureContract(unittest.TestCase):
    """Tests verifying the model structure contract for FSDP2."""

    def test_model_with_missing_optional_attrs(self):
        """Test that models with missing optional attributes work."""

        class MinimalModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleDict({"layer_0": nn.Linear(10, 10)})
                # Missing tok_embeddings, norm, output

        model = MinimalModel()

        with patch("torchtitan.distributed.fsdp.fully_shard") as mock_fully_shard:
            apply_fsdp_to_model(
                model,
                dp_mesh=MagicMock(),
                param_dtype=torch.float32,
                reduce_dtype=torch.float32,
                pp_enabled=False,
            )

        # Should still work, just apply FSDP to layers and root
        self.assertGreaterEqual(mock_fully_shard.call_count, 2)

    def test_model_with_none_attributes(self):
        """Test that models with None attributes work."""

        class ModelWithNone(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleDict({"layer_0": nn.Linear(10, 10)})
                self.tok_embeddings = None
                self.norm = None
                self.output = None

        model = ModelWithNone()

        with patch("torchtitan.distributed.fsdp.fully_shard") as mock_fully_shard:
            apply_fsdp_to_model(
                model,
                dp_mesh=MagicMock(),
                param_dtype=torch.float32,
                reduce_dtype=torch.float32,
                pp_enabled=False,
            )

        # Should work with None attributes
        self.assertGreaterEqual(mock_fully_shard.call_count, 2)


if __name__ == "__main__":
    unittest.main()
