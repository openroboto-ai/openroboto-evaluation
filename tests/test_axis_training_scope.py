"""Action-expert training must leave visual/language parameters untouched."""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "libero_eval"))
from axis_training_scope import action_expert_parameter


class ParameterScopeTests(unittest.TestCase):
    def test_action_expert_and_both_state_input_projection_variants_are_selected(self):
        for path in [
            ("PaliGemma", "llm", "layers", "attn", "q_einsum_1", "w"),
            ("PaliGemma", "llm", "layers", "mlp_1", "gating_einsum"),
            ("PaliGemma", "llm", "final_norm_1", "scale"),
            ("action_in_proj", "kernel"),
            ("action_out_proj", "bias"),
            ("time_mlp_in", "kernel"),
            ("time_mlp_out", "kernel"),
            ("state_proj", "kernel"),
            ("action_time_mlp_in", "kernel"),
            ("action_time_mlp_out", "kernel"),
        ]:
            with self.subTest(path=path):
                self.assertTrue(action_expert_parameter(path, None))

    def test_vision_language_embeddings_and_similar_names_remain_frozen(self):
        for path in [
            (),
            ("PaliGemma", "img", "Transformer", "encoderblock", "LayerNorm_1", "bias"),
            ("PaliGemma", "llm", "embedder", "input_embedding"),
            ("PaliGemma", "llm", "layers", "attn", "q_einsum", "w"),
            ("PaliGemma", "llm", "layers", "mlp", "gating_einsum"),
            ("PaliGemma", "llm", "layers", "attn", "q_einsum_10", "w"),
            ("other_1", "weight"),
            ("action_in_proj_backup", "kernel"),
        ]:
            with self.subTest(path=path):
                self.assertFalse(action_expert_parameter(path, None))


if __name__ == "__main__":
    unittest.main()
