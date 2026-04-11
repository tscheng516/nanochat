"""
Read resid_lambdas, x0_lambdas, smear_lambda, and backout_lambda from nanochat pretrained checkpoints.

When using --pt, automatically looks for the companion meta_NNNNNN.json in the same
directory (step extracted from the filename).  If found, reads the ``reinit`` flag from
``model_config`` and shows the init values next to the trained values.

Supports two formats:
  1. Native nanochat .pt checkpoint (karpathy/nanochat-d32, sdobson/nanochat)
  2. HuggingFace safetensors (nanochat-students/nanochat-d20, karpathy/nanochat-d32 via HF)

Usage:
    # Option A – HuggingFace safetensors (recommended, no GPU needed)
    python scripts/read_lambdas.py --hf nanochat-students/nanochat-d20

    # Option B – native .pt checkpoint
    python scripts/read_lambdas.py --pt ~/.cache/nanochat/base_checkpoints/d24_18108/model_005568.pt

    # Option C – Karpathy's official d32 model via HF
    python scripts/read_lambdas.py --hf karpathy/nanochat-d32

Command:
python read_lambdas.py   --pt ~/.cache/nanochat/base_checkpoints/d16_lambdas/model_041600.pt    
"""

import argparse
import json
import os
import re
import sys

import torch
import numpy as np


# ─────────────────────────────────────────────
# Helper: classify lambda keys from a state dict
# ─────────────────────────────────────────────

def _classify_lambda_keys(state_dict):
    """Return (resid_keys, x0_keys, smear_keys, backout_keys) from state_dict."""
    resid_keys = sorted([k for k in state_dict if "resid_lambda" in k])
    x0_keys    = sorted([k for k in state_dict if "x0_lambda"    in k])

    if not resid_keys:
        # Try flat tensor names used by nanochat's own save format
        resid_keys = sorted([k for k in state_dict if "resid_lambdas" in k])
        x0_keys    = sorted([k for k in state_dict if "x0_lambdas"    in k])

    smear_keys   = sorted([k for k in state_dict if "smear_lambda" in k])
    # Support both "backout_lambda" (nn.Parameter attr name) and "backout.lambda"
    # (dot-separated key that appears in some checkpoint formats)
    backout_keys = sorted([k for k in state_dict if "backout_lambda" in k or "backout.lambda" in k])

    return resid_keys, x0_keys, smear_keys, backout_keys


def _print_key_counts(resid_keys, x0_keys, smear_keys, backout_keys):
    print(f"Found {len(resid_keys)} resid_lambda keys, {len(x0_keys)} x0_lambda keys, "
          f"{len(smear_keys)} smear_lambda keys, {len(backout_keys)} backout_lambda keys")


# ─────────────────────────────────────────────
# Helper: load meta JSON and compute init values
# ─────────────────────────────────────────────

def _extract_step_from_pt(pt_path: str):
    """Extract the integer step from a filename like model_041600.pt."""
    basename = os.path.basename(pt_path)
    m = re.search(r"(\d+)\.pt$", basename)
    return int(m.group(1)) if m else None


def _load_meta_json(pt_path: str):
    """Try to load meta_NNNNNN.json from the same directory as the .pt file.

    Returns the parsed dict, or None if the file does not exist.
    """
    step = _extract_step_from_pt(pt_path)
    if step is None:
        return None
    meta_path = os.path.join(os.path.dirname(pt_path), f"meta_{step:06d}.json")
    if not os.path.isfile(meta_path):
        return None
    print(f"  Found companion meta file: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _compute_init_lambdas(n_layer: int, reinit: bool, relambdas: bool):
    """Reproduce the init values from GPT.init_weights for display purposes.

    Returns (resid_init, x0_init, smear_init, backout_init) where the per-layer
    lists have length *n_layer* and the scalar lists have length 1.

    Either *reinit* or *relambdas* triggers lambda reinitialization for
    resid/x0.  They differ only in smear/backout defaults:
      - relambdas (old): smear_init=0.0, backout_init=0.2
      - reinit   (new): smear_init=0.2, backout_init=0.0
    """
    use_lambda_init = reinit or relambdas
    resid_init = []
    x0_init = []
    for i in range(n_layer):
        r = i / max(n_layer - 1, 1)
        resid_init.append(0.5 + 0.5 * r if use_lambda_init else 1.15 - 0.10 * r)
        x0_init.append(3.0 * (0.1 / 3.0) ** r if use_lambda_init else 0.20 - 0.15 * r)
    if reinit:
        smear_init = [0.2]
        backout_init = [0.0]
    elif relambdas:
        smear_init = [0.0]
        backout_init = [0.2]
    else:
        smear_init = [0.0]
        backout_init = [0.2]
    return resid_init, x0_init, smear_init, backout_init


# ─────────────────────────────────────────────
# 1. Load from native .pt checkpoint
# ─────────────────────────────────────────────

def load_from_pt(path: str):
    print(f"\n[native .pt] Loading from: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=True)

    # The checkpoint is either the raw state_dict or a dict containing one
    state_dict = ckpt if not isinstance(ckpt, dict) or "model" not in ckpt else ckpt["model"]
    if hasattr(state_dict, "state_dict"):          # nn.Module was saved
        state_dict = state_dict.state_dict()

    resid_keys, x0_keys, smear_keys, backout_keys = _classify_lambda_keys(state_dict)
    _print_key_counts(resid_keys, x0_keys, smear_keys, backout_keys)
    return state_dict, resid_keys, x0_keys, smear_keys, backout_keys


# ─────────────────────────────────────────────
# 2. Load from HuggingFace safetensors
# ─────────────────────────────────────────────

def load_from_hf(repo_id: str):
    print(f"\n[HuggingFace] Loading from repo: {repo_id}")
    try:
        from safetensors import safe_open
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        sys.exit("Please install: pip install safetensors huggingface_hub")

    # Find all .safetensors shards
    all_files = list(list_repo_files(repo_id))
    shard_files = sorted([f for f in all_files if f.endswith(".safetensors")])
    if not shard_files:
        sys.exit(f"No .safetensors files found in {repo_id}. "
                 "Try the --pt path option instead.")

    print(f"  Shards found: {shard_files}")

    state_dict = {}
    for shard in shard_files:
        local = hf_hub_download(repo_id=repo_id, filename=shard)
        with safe_open(local, framework="pt", device="cpu") as f:
            for key in f.keys():
                if "lambda" in key.lower():
                    state_dict[key] = f.get_tensor(key)

    resid_keys, x0_keys, smear_keys, backout_keys = _classify_lambda_keys(state_dict)
    _print_key_counts(resid_keys, x0_keys, smear_keys, backout_keys)
    return state_dict, resid_keys, x0_keys, smear_keys, backout_keys


# ─────────────────────────────────────────────
# 3. Pretty-print and analyse the lambdas
# ─────────────────────────────────────────────

def _to_float_list(state_dict, keys):
    """
    Extract a flat list of Python floats from a collection of tensor keys.

    Handles three storage layouts nanochat uses in practice:
      - One scalar tensor per layer  (shape [] or [1])
      - One 1-D tensor per layer     (shape [1] or [d] — squeeze to scalar)
      - A single packed vector       (shape [L]) stored under one key
    """
    vals = []
    for k in keys:
        t = state_dict[k].float().cpu()
        flat = t.reshape(-1).tolist()   # always gives a plain list of floats
        vals.extend(flat)
    return vals


def analyse(state_dict, resid_keys, x0_keys, smear_keys, backout_keys,
            init_lambdas=None):
    """Pretty-print trained lambda values and optionally show init values side-by-side.

    *init_lambdas*, when provided, is a tuple
    ``(resid_init, x0_init, smear_init, backout_init)`` as returned by
    ``_compute_init_lambdas``.
    """
    if not resid_keys and not x0_keys and not smear_keys and not backout_keys:
        print("\n[WARNING] No lambda keys found. Printing ALL keys for inspection:")
        for k in sorted(state_dict.keys()):
            print(f"  {k:60s}  shape={tuple(state_dict[k].shape)}")
        return

    has_init = init_lambdas is not None
    if has_init:
        resid_init, x0_init, smear_init, backout_init = init_lambdas

    # Print raw shapes so the user can see what was loaded
    print("\nRaw tensor shapes:")
    for k in resid_keys + x0_keys + smear_keys + backout_keys:
        print(f"  {k:60s}  shape={tuple(state_dict[k].shape)}")

    resid_vals = _to_float_list(state_dict, resid_keys)
    x0_vals    = _to_float_list(state_dict, x0_keys)
    smear_vals   = _to_float_list(state_dict, smear_keys)
    backout_vals = _to_float_list(state_dict, backout_keys)

    # ── per-layer table ──────────────────────────────────────────────────
    if has_init:
        print("\n" + "─"*94)
        print(f"{'Layer':>6}  {'resid_lambda':>14} {'(init)':>10}  {'x0_lambda':>12} {'(init)':>10}")
        print("─"*94)
    else:
        print("\n" + "─"*60)
        print(f"{'Layer':>6}  {'resid_lambda':>14}  {'x0_lambda':>12}")
        print("─"*60)
    n = max(len(resid_vals), len(x0_vals))
    for i in range(n):
        rl = f"{resid_vals[i]:.6f}" if i < len(resid_vals) else "  —"
        xl = f"{x0_vals[i]:.6f}"   if i < len(x0_vals)    else "  —"
        if has_init:
            ri = f"{resid_init[i]:.6f}" if i < len(resid_init) else "  —"
            xi = f"{x0_init[i]:.6f}"    if i < len(x0_init)    else "  —"
            print(f"{i:>6}  {rl:>14} {ri:>10}  {xl:>12} {xi:>10}")
        else:
            print(f"{i:>6}  {rl:>14}  {xl:>12}")
    print("─"*(94 if has_init else 60))

    # ── summary statistics ───────────────────────────────────────────────
    if resid_vals:
        rv = np.array(resid_vals, dtype=float)
        print(f"\nresid_lambdas ")
        print(f"  mean={rv.mean():.4f}  std={rv.std():.4f}  "
              f"min={rv.min():.4f}  max={rv.max():.4f}")
        print(f"  layers with lambda < 1.0 : {(rv < 1.0).sum()}/{len(rv)}")
        print(f"  layers with lambda > 1.0 : {(rv > 1.0).sum()}/{len(rv)}")

    if x0_vals:
        xv = np.array(x0_vals, dtype=float)
        print(f"\nx0_lambdas ")
        print(f"  mean={xv.mean():.4f}  std={xv.std():.4f}  "
              f"min={xv.min():.4f}  max={xv.max():.4f}")

    if smear_vals:
        sv = np.array(smear_vals, dtype=float)
        label = f"\nsmear_lambda "
        if has_init:
            label += f" (init={'  '.join(f'{v:.6f}' for v in smear_init)})"
        print(label)
        print(f"  value={'  '.join(f'{v:.6f}' for v in sv)}")

    if backout_vals:
        bv = np.array(backout_vals, dtype=float)
        label = f"\nbackout_lambda "
        if has_init:
            label += f" (init={'  '.join(f'{v:.6f}' for v in backout_init)})"
        print(label)
        print(f"  value={'  '.join(f'{v:.6f}' for v in bv)}")

    # ── theory check ─────────────────────────────────────────────────────
    # if resid_vals and x0_vals:
    #     rv = np.array(resid_vals, dtype=float)
    #     xv = np.array(x0_vals,   dtype=float)
    #     # Fixed-point variance (sigma_F^2 ~ 1 at init, V0 ~ 1)
    #     # V* = (mu^2 * V0) / (1 - lambda) + sigma_F^2 / (1 - lambda^2)
    #     lam   = rv.mean()
    #     mu    = xv.mean()
    #     sigma_F2 = 1.0   # approximate
    #     V0       = 1.0
    #     if lam < 1.0:
    #         Vstar = (mu**2 * V0) / (1 - lam) + sigma_F2 / (1 - lam**2)
    #         grad_gain = mu / (1 - lam)
    #         print(f"\n── Theory (mean-field, sigma_F=1, V0=1) ──────────────")
    #         print(f"  lambda_mean = {lam:.4f},  mu_mean = {mu:.4f}")
    #         print(f"  Fixed-point variance  V*  = {Vstar:.4f}")
    #         print(f"  E2E gradient gain  mu/(1-lambda) = {grad_gain:.4f}")
    #     else:
    #         print(f"\n[NOTE] mean lambda >= 1 ({lam:.4f}); "
    #               "fixed-point analysis requires lambda < 1.")


# ─────────────────────────────────────────────
# 4. Entry point
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Read nanochat lambda scalars")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--hf", metavar="REPO_ID",
                       help="HuggingFace repo id, e.g. nanochat-students/nanochat-d20")
    group.add_argument("--pt", metavar="PATH",
                       help="Path to a native nanochat .pt checkpoint file")
    args = parser.parse_args()

    init_lambdas = None

    if args.hf:
        state_dict, resid_keys, x0_keys, smear_keys, backout_keys = load_from_hf(args.hf)
    else:
        state_dict, resid_keys, x0_keys, smear_keys, backout_keys = load_from_pt(args.pt)
        # Try to load the companion meta JSON to determine reinit and n_layer
        meta = _load_meta_json(args.pt)
        if meta is not None:
            model_config = meta.get("model_config", {})
            reinit = model_config.get("reinit", False)
            relambdas = model_config.get("relambdas", False)
            n_layer = model_config.get("n_layer")
            print(f"  Training config: reinit={reinit}, relambdas={relambdas}, n_layer={n_layer}")
            if n_layer is not None:
                init_lambdas = _compute_init_lambdas(n_layer, reinit, relambdas)

    analyse(state_dict, resid_keys, x0_keys, smear_keys, backout_keys,
            init_lambdas=init_lambdas)


if __name__ == "__main__":
    main()