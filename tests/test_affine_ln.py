"""
Tests for the affine_ln feature in GPT models.

Verifies that:
- affine_ln=False (default) produces no learnable LN parameters (backward-compatible)
- affine_ln=True produces learnable weight and bias per RMSNorm instance
- Weights are initialized to 1.0 and biases to 0.0
- The number of RMSNorm instances per block matches the norm_pos semantics:
    pre / reordered / post / pre_post: 2 (one for attn, one for MLP)
    _post:                             1 (only MLP; no norm for attention)
    peri / sandwich:                   4 (independent pre+post for each sublayer)
- q/k/v norms in attention are independent modules, created only for enabled flags
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

def make_model(affine_ln=False, norm_pos="pre", n_layer=2, n_head=4, n_embd=128,
               w_norm=True, k_norm=True, v_norm=False):
    """Build a small GPT model on CPU."""
    config = GPTConfig(
        n_layer=n_layer, n_head=n_head, n_kv_head=n_head,
        n_embd=n_embd, vocab_size=256, sequence_len=64,
        affine_ln=affine_ln, norm_pos=norm_pos,
        w_norm=w_norm, k_norm=k_norm, v_norm=v_norm,
    )
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device="cpu")
    model.init_weights()
    return model


def count_block_rms_norms(block):
    """Count RMSNorm instances that are direct or indirect children of a Block,
    but NOT inside the attention sub-module (attn is counted separately)."""
    # Collect all RMSNorm modules in the block
    all_in_block = {id(m) for m in block.modules() if isinstance(m, RMSNorm)}
    # Exclude those inside attn
    in_attn = {id(m) for m in block.attn.modules() if isinstance(m, RMSNorm)}
    return len(all_in_block - in_attn)


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

    # ------------------------------------------------------------------
    # norm_pos-specific RMSNorm instance counts
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("norm_pos", ["pre", "reordered", "post", "pre_post"])
    def test_two_block_norms_for_standard_positions(self, norm_pos):
        """pre / reordered / post / pre_post must give exactly 2 block-level norms."""
        model = make_model(norm_pos=norm_pos)
        for block in model.transformer.h:
            n = count_block_rms_norms(block)
            assert n == 2, f"Expected 2 block norms for norm_pos={norm_pos!r}, got {n}"

    def test_one_block_norm_for_post_only(self):
        """_post must create only 1 block-level norm (no norm for attention)."""
        model = make_model(norm_pos="_post")
        for block in model.transformer.h:
            n = count_block_rms_norms(block)
            assert n == 1, f"Expected 1 block norm for norm_pos='_post', got {n}"

    @pytest.mark.parametrize("norm_pos", ["peri", "sandwich"])
    def test_four_block_norms_for_peri_sandwich(self, norm_pos):
        """peri / sandwich must create 4 independent block-level norms."""
        model = make_model(norm_pos=norm_pos)
        for block in model.transformer.h:
            n = count_block_rms_norms(block)
            assert n == 4, f"Expected 4 block norms for norm_pos={norm_pos!r}, got {n}"

    @pytest.mark.parametrize("norm_pos", ["peri", "sandwich"])
    def test_peri_sandwich_norms_are_independent(self, norm_pos):
        """peri / sandwich block norms must be 4 distinct Python objects."""
        model = make_model(norm_pos=norm_pos, affine_ln=True)
        for block in model.transformer.h:
            norms = [m for m in block.modules()
                     if isinstance(m, RMSNorm) and m not in
                     [m2 for m2 in block.attn.modules() if isinstance(m2, RMSNorm)]]
            ids = [id(m) for m in norms]
            assert len(set(ids)) == len(ids), \
                "peri/sandwich block norms must be distinct module instances"

    def test_post_only_no_attn_norm_in_forward(self):
        """_post forward pass must not apply any norm to the attention input."""
        model = make_model(norm_pos="_post", affine_ln=True)
        block = model.transformer.h[0]
        # norm1 must not exist on _post blocks
        assert not hasattr(block, "norm1"), \
            "_post block must not have norm1 (attention receives no normalization)"

    # ------------------------------------------------------------------
    # Attention QK/V norms
    # ------------------------------------------------------------------

    def test_attn_qk_norms_are_independent(self):
        """norm_q and norm_k must be independent module instances."""
        model = make_model(affine_ln=True, w_norm=True, k_norm=True)
        for block in model.transformer.h:
            assert block.attn.norm_q is not block.attn.norm_k, \
                "norm_q and norm_k must be independent"

    def test_attn_v_norm_absent_when_disabled(self):
        """norm_v must be None when v_norm=False."""
        model = make_model(affine_ln=True, v_norm=False)
        for block in model.transformer.h:
            assert block.attn.norm_v is None

    def test_attn_v_norm_present_when_enabled(self):
        """norm_v must be a RMSNorm when v_norm=True."""
        model = make_model(affine_ln=True, v_norm=True)
        for block in model.transformer.h:
            assert isinstance(block.attn.norm_v, RMSNorm)

    # ------------------------------------------------------------------
    # Forward pass across all norm_pos variants
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("norm_pos", ["pre", "reordered", "post", "pre_post", "_post", "peri", "sandwich"])
    def test_forward_pass_all_norm_pos(self, norm_pos):
        """Forward pass must succeed and produce a finite loss for every norm_pos."""
        idx = torch.randint(0, 256, (1, 10))
        for affine_ln in (False, True):
            model = make_model(norm_pos=norm_pos, affine_ln=affine_ln)
            loss = model(idx, idx)
            assert not torch.isnan(loss), \
                f"NaN loss with norm_pos={norm_pos!r}, affine_ln={affine_ln}, loss={loss}"

    # ------------------------------------------------------------------
    # Optimizer correctness
    # ------------------------------------------------------------------

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

    @pytest.mark.parametrize("norm_pos", ["pre", "_post", "peri"])
    def test_no_double_counting_in_optimizer(self, norm_pos):
        """Every model parameter must appear in exactly one optimizer param group."""
        model = make_model(affine_ln=True, norm_pos=norm_pos)
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

    @pytest.mark.parametrize("norm_pos", ["pre", "_post", "peri"])
    def test_param_count_consistent_per_norm_pos(self, norm_pos):
        """num_scaling_params total must match actual parameter count for each norm_pos."""
        for affine_ln in (False, True):
            model = make_model(affine_ln=affine_ln, norm_pos=norm_pos)
            counts = model.num_scaling_params()
            actual = sum(p.numel() for p in model.parameters())
            assert counts["total"] == actual, \
                f"Param count mismatch: norm_pos={norm_pos!r}, affine_ln={affine_ln}"

