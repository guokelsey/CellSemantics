"""Synthetic CPU packaging checks; no Qwen loading or biological scoring."""
import ast
import importlib
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import torch
from models import prompt_config
from models.special_tokens import CELL_PLACEHOLDER
from models_v3.uce_projector import UCEProjectorV3
from scripts.step1_train_projector import parse_rich_label, parse_target, fuzzy_match

torch.set_num_threads(2)


class ReleaseTests(unittest.TestCase):
    def model(self):
        torch.manual_seed(7)
        return UCEProjectorV3(uce_dim=4, hidden=8, num_tokens=4,
                              output_dim=16, num_mixer_layers=1,
                              num_heads=2, dropout=0.0,
                              n_cell_types=3, n_organs=2)

    def test_all_python_syntax(self):
        for path in ROOT.rglob("*.py"):
            with self.subTest(path=str(path.relative_to(ROOT))):
                ast.parse(path.read_text(encoding="utf-8"))

    def test_restored_local_import_symbols(self):
        local = {"models.prompt_config", "scripts.step1_train_projector"}
        tree = ast.parse((ROOT / "code/scripts_v3/train_v3.py").read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in local:
                module = importlib.import_module(node.module)
                for alias in node.names:
                    self.assertTrue(hasattr(module, alias.name), alias.name)

    def test_cell_literal_matches_wrapper(self):
        tree = ast.parse((ROOT / "code/models_v3/sagav3_llm.py").read_text())
        values = [ast.literal_eval(node.value) for node in tree.body
                  if isinstance(node, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "CELL_PLACEHOLDER"
                          for t in node.targets)]
        self.assertEqual(values, [CELL_PLACEHOLDER])

    def test_prompt_has_selected_slot_count(self):
        text = prompt_config.make_freeform_prompt(32, rng=random.Random(42))
        self.assertEqual(text.count(CELL_PLACEHOLDER), 32)
        self.assertIn("assistant", text)

    def test_legacy_helpers_preserved(self):
        self.assertEqual(parse_target("T cell | lung"), ("T cell", "lung"))
        self.assertEqual(parse_rich_label("This is a T cell from the lung of donor X"),
                         ("T cell", "lung"))
        self.assertTrue(fuzzy_match("T-cell", "Tcell"))
        self.assertFalse(fuzzy_match("", "cell"))

    def test_tiny_forward_shapes_and_finiteness(self):
        tokens, aux = self.model().forward_with_aux_all(torch.randn(2, 4))
        self.assertEqual(tuple(tokens.shape), (2, 4, 16))
        self.assertEqual(tuple(aux["pooled_features"].shape), (2, 16))
        self.assertTrue(torch.isfinite(tokens).all())

    def test_raw_aux_does_not_train_token_path(self):
        model = self.model()
        _, aux = model.forward_with_aux_all(torch.randn(2, 4))
        sum(logits.square().mean() for logits in aux["raw"]).backward()
        self.assertIsNone(model.expand[0].weight.grad)
        self.assertIsNotNone(model.aux_ct_head.weight.grad)

    def test_pooled_aux_trains_token_path(self):
        model = self.model()
        _, aux = model.forward_with_aux_all(torch.randn(2, 4))
        sum(logits.square().mean() for logits in aux["pooled"]).backward()
        grad = model.expand[0].weight.grad
        self.assertIsNotNone(grad)
        self.assertGreater(float(grad.abs().sum()), 0)

    def test_toy_frozen_decoder_preserves_input_gradient(self):
        model = self.model()
        decoder = torch.nn.Linear(16, 3)
        decoder.requires_grad_(False)
        decoder(model(torch.randn(2, 4)).mean(1)).square().mean().backward()
        self.assertIsNone(decoder.weight.grad)
        self.assertGreater(float(model.to_tokens.weight.grad.abs().sum()), 0)


if __name__ == "__main__":
    unittest.main()
