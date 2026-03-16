"""
Tests for the affine_ln feature in GPT models.

Verifies that:
- affine_ln=False (default) produces no learnable LN parameters (backward-compatible)
- affine_ln=True produces learnable weight and bias per RMSNorm instance
- Weights are initialized to 1.0 and biases to 0.0
- The optimizer setup correctly routes LN params to a dedicated AdamW group
- Parameter counts are consistent (no double-counting)

Run: python -m pytest tests/test_affine_ln.py -v
"""

import torch
import pytest
from nanochat.gpt import GPT, GPTConfig, RMSNorm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_model(affine_ln=False, n_layer=2, n_head=4, n_embd=128):
    """Build a small GPT model on CPU."""
    config = GPTConfig(
        n_layer=n_layer, n_head=n_head, n_kv_head=n_head,
        n_embd=n_embd, vocab_size=256, sequence_len=64,
        affine_ln=affine_ln,
    )
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device="cpu")
    model.init_weights()
    return model


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAffineLn:
    """Tests for the affine_ln flag in GPT models."""

    def test_default_no_ln_params(self):
        """affine_ln=False (default) must produce zero learnable LN parameters."""
        model = make_model(affine_ln=False)
        ln_params = model._collect_ln_params()
        assert len(ln_params) == 0, "Expected no LN params when affine_ln=False"

    def test_affine_ln_has_params(self):
        """affine_ln=True must produce learnable LN parameters."""
        model = make_model(affine_ln=True)
        ln_params = model._collect_ln_params()
        assert len(ln_params) > 0, "Expected LN params when affine_ln=True"

    def test_weight_init_ones(self):
        """Affine LN weights must be initialized to 1.0."""
        model = make_model(affine_ln=True)
        for module in model.modules():
            if isinstance(module, RMSNorm) and module.weight is not None:
                assert torch.allclose(module.weight, torch.ones_like(module.weight)), \
                    f"LN weight not initialized to 1: {module.weight}"

    def test_bias_init_zeros(self):
        """Affine LN biases must be initialized to 0.0."""
        model = make_model(affine_ln=True)
        for module in model.modules():
            if isinstance(module, RMSNorm) and module.bias is not None:
                assert torch.allclose(module.bias, torch.zeros_like(module.bias)), \
                    f"LN bias not initialized to 0: {module.bias}"

    def test_param_count_consistent(self):
        """num_scaling_params total must match sum(p.numel() for all parameters)."""
        for affine_ln in (False, True):
            model = make_model(affine_ln=affine_ln)
            counts = model.num_scaling_params()
            assert counts["total"] == sum(p.numel() for p in model.parameters()), \
                f"Param count mismatch with affine_ln={affine_ln}"

    def test_ln_count_increases_with_affine(self):
        """'ln' entry in num_scaling_params must be 0 without affine, >0 with affine."""
        model_no_affine = make_model(affine_ln=False)
        model_affine = make_model(affine_ln=True)
        assert model_no_affine.num_scaling_params()["ln"] == 0
        assert model_affine.num_scaling_params()["ln"] > 0

    def test_forward_pass_both_modes(self):
        """Forward pass must not raise for either affine_ln setting."""
        idx = torch.randint(0, 256, (1, 10))
        for affine_ln in (False, True):
            model = make_model(affine_ln=affine_ln)
            loss = model(idx, idx)
            assert not torch.isnan(loss), f"NaN loss with affine_ln={affine_ln}"

    def test_optimizer_no_ln_group_when_not_affine(self):
        """When affine_ln=False, no extra LN param group should appear in the optimizer."""
        model = make_model(affine_ln=False)
        optimizer = model.setup_optimizer(ln_lr=3e-4)
        ln_groups = [g for g in optimizer.param_groups
                     if g.get("kind") == "adamw" and abs(g["lr"] - 3e-4) < 1e-10]
        assert len(ln_groups) == 0, "Did not expect an LN group when affine_ln=False"

    def test_optimizer_ln_group_when_affine(self):
        """When affine_ln=True, exactly one AdamW LN param group with lr=ln_lr must exist."""
        model = make_model(affine_ln=True)
        ln_lr = 3e-4
        optimizer = model.setup_optimizer(ln_lr=ln_lr)
        ln_groups = [g for g in optimizer.param_groups
                     if g.get("kind") == "adamw" and abs(g["lr"] - ln_lr) < 1e-10]
        assert len(ln_groups) == 1, f"Expected 1 LN optimizer group, got {len(ln_groups)}"

    def test_optimizer_ln_params_not_in_muon_group(self):
        """LN params must never appear in Muon (matrix) groups."""
        model = make_model(affine_ln=True)
        optimizer = model.setup_optimizer(ln_lr=3e-4)
        ln_param_ids = {id(p) for p in model._collect_ln_params()}
        for group in optimizer.param_groups:
            if group.get("kind") == "muon":
                for p in group["params"]:
                    assert id(p) not in ln_param_ids, \
                        "LN param found in Muon group – should only be in AdamW"

    def test_no_double_counting_in_optimizer(self):
        """Every model parameter must appear in exactly one optimizer param group."""
        model = make_model(affine_ln=True)
        optimizer = model.setup_optimizer(ln_lr=3e-4)
        seen_ids = {}
        for i, group in enumerate(optimizer.param_groups):
            for p in group["params"]:
                assert id(p) not in seen_ids, \
                    f"Parameter {p.shape} appears in groups {seen_ids[id(p)]} and {i}"
                seen_ids[id(p)] = i
        # Every model param must appear somewhere
        model_param_ids = {id(p) for p in model.parameters()}
        assert model_param_ids == set(seen_ids.keys()), \
            "Not all model parameters are covered by the optimizer"
