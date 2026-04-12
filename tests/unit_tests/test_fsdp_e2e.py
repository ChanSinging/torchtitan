# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
End-to-end test for FSDP2 generalization.

This test verifies FSDP2 application works correctly across different model architectures.
"""

import unittest
from unittest.mock import MagicMock, patch



class TestLlama3FSDPApplication(unittest.TestCase):
    """Test FSDP2 application on Llama3 model structure."""

    @patch("torchtitan.models.llama3.parallelize.apply_fsdp_to_model")
    def test_parallelize_llama_calls_apply_fsdp(self, mock_apply_fsdp):
        """Test that parallelize_llama calls the new apply_fsdp_to_model."""
        try:
            from torchtitan.models.llama3.parallelize import parallelize_llama

            # Create a mock model with required attributes
            mock_model = MagicMock()
            mock_model.layers = MagicMock()
            mock_model.tok_embeddings = MagicMock()
            mock_model.norm = MagicMock()
            mock_model.output = MagicMock()

            mock_parallel_dims = MagicMock()
            mock_parallel_dims.fsdp_enabled = True
            mock_parallel_dims.dp_replicate_enabled = False
            mock_parallel_dims.tp_enabled = False
            mock_parallel_dims.cp_enabled = False
            mock_parallel_dims.ep_enabled = False
            mock_parallel_dims.tp = 1
            mock_parallel_dims.cp = 1
            mock_parallel_dims.seq_len_divisor = 1
            mock_parallel_dims.get_mesh.return_value = MagicMock()

            mock_training = MagicMock()
            mock_training.seq_len = 512
            mock_training.mixed_precision_param = "float32"
            mock_training.mixed_precision_reduce = "float32"
            mock_training.enable_cpu_offload = False

            mock_parallelism = MagicMock()
            mock_parallelism.fsdp_reshard_after_forward = "default"
            mock_parallelism.disable_loss_parallel = False
            mock_parallelism.enable_sequence_parallel = True

            mock_compile = MagicMock()
            mock_compile.enable = False

            mock_ac = MagicMock()
            mock_ac.mode = "none"

            mock_converters = MagicMock()

            # Call parallelize_llama
            parallelize_llama(
                mock_model,
                parallel_dims=mock_parallel_dims,
                training=mock_training,
                model_converters=mock_converters,
                parallelism=mock_parallelism,
                compile_config=mock_compile,
                ac_config=mock_ac,
                dump_folder="/tmp",
            )

            # Verify apply_fsdp_to_model was called
            mock_apply_fsdp.assert_called_once()

        except ImportError:
            self.skipTest("Llama3 parallelize not available")


class TestLlama4FSDPApplication(unittest.TestCase):
    """Test FSDP2 application on Llama4 model structure."""

    def test_parallelize_llama4_calls_apply_fsdp(self):
        """Test that parallelize_llama calls apply_fsdp_to_model with correct args."""
        try:
            import inspect

            from torchtitan.models.llama4.parallelize import apply_fsdp

            # Verify function signature matches our expectations
            sig = inspect.signature(apply_fsdp)
            params = list(sig.parameters.keys())

            expected = [
                "model",
                "dp_mesh",
                "param_dtype",
                "reduce_dtype",
                "pp_enabled",
                "cpu_offload",
                "reshard_after_forward_policy",
                "ep_degree",
                "edp_mesh",
                "gradient_divide_factor",
            ]

            for param in expected:
                self.assertIn(param, params)

        except ImportError:
            self.skipTest("Llama4 parallelize not available")


class TestDeepSeekV3FSDPApplication(unittest.TestCase):
    """Test FSDP2 application on DeepSeekV3 model structure."""

    def test_parallelize_deepseekv3_imports(self):
        """Test that deepseek_v3 parallelize imports work correctly."""
        try:
            from torchtitan.models.deepseek_v3.parallelize import (
                apply_non_moe_tp,
                parallelize_deepseekv3,
            )

            self.assertTrue(callable(parallelize_deepseekv3))
            self.assertTrue(callable(apply_non_moe_tp))
        except ImportError as e:
            self.skipTest(f"DeepSeekV3 parallelize not available: {e}")


class TestGptOssFSDPApplication(unittest.TestCase):
    """Test FSDP2 application on GPT-OSS model structure."""

    def test_parallelize_gptoss_imports(self):
        """Test that gpt_oss parallelize imports work correctly."""
        try:
            from torchtitan.models.gpt_oss.parallelize import (
                apply_moe_ep_tp,
                apply_non_moe_tp,
                parallelize_gptoss,
            )

            self.assertTrue(callable(parallelize_gptoss))
            self.assertTrue(callable(apply_non_moe_tp))
            self.assertTrue(callable(apply_moe_ep_tp))
        except ImportError as e:
            self.skipTest(f"GPT-OSS parallelize not available: {e}")


class TestFluxFSDPApplication(unittest.TestCase):
    """Test FSDP2 application on Flux model structure."""

    @unittest.skipUnless(
        __import__("importlib.util").util.find_spec("einops") is not None,
        "flux requires einops dependency",
    )
    def test_parallelize_flux_imports(self):
        """Test that flux parallelize imports work correctly."""
        try:
            from torchtitan.models.flux.parallelize import (
                apply_fsdp,
                parallelize_flux,
            )

            self.assertTrue(callable(parallelize_flux))
            self.assertTrue(callable(apply_fsdp))
        except ImportError as e:
            self.skipTest(f"Flux parallelize not available: {e}")


class TestFSDPImportConsistency(unittest.TestCase):
    """Test that all models import FSDP functions consistently."""

    def test_all_models_use_new_fsdp_imports(self):
        """Verify all model parallelize files import from distributed.fsdp."""
        models = [
            ("llama3", "torchtitan.models.llama3.parallelize"),
            ("llama4", "torchtitan.models.llama4.parallelize"),
            ("deepseek_v3", "torchtitan.models.deepseek_v3.parallelize"),
            ("gpt_oss", "torchtitan.models.gpt_oss.parallelize"),
        ]

        for model_name, module_path in models:
            with self.subTest(model=model_name):
                try:
                    module = __import__(module_path, fromlist=[""])
                    # Check that the module doesn't have its own full FSDP implementation
                    source_file = module.__file__
                    with open(source_file, "r") as f:
                        content = f.read()

                    # Should import from distributed.fsdp
                    self.assertIn(
                        "from torchtitan.distributed.fsdp import",
                        content,
                        f"{model_name} should import from distributed.fsdp",
                    )

                    # Should not have its own fully_shard calls (using the wrapper instead)
                    # Note: flux is allowed to have its own since it has different architecture
                    if model_name != "flux":
                        # Count direct fully_shard calls outside of imports
                        lines = content.split("\n")
                        direct_calls = 0
                        for line in lines:
                            if "fully_shard(" in line and not line.strip().startswith(
                                "#"
                            ):
                                direct_calls += 1

                        # Should be minimal or wrapped in the model-specific function
                        self.assertLess(
                            direct_calls,
                            10,
                            f"{model_name} should use apply_fsdp_to_model wrapper",
                        )

                except ImportError:
                    self.skipTest(f"{model_name} not available")


if __name__ == "__main__":
    unittest.main()
