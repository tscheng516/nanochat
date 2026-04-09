"""
Read resid_lambdas and x0_lambdas from nanochat pretrained checkpoints.

Supports two formats:
  1. Native nanochat .pt checkpoint (karpathy/nanochat-d32, sdobson/nanochat)
  2. HuggingFace safetensors (nanochat-students/nanochat-d20, karpathy/nanochat-d32 via HF)

Usage:
    # Option A – HuggingFace safetensors (recommended, no GPU needed)
    python read_lambdas.py --hf nanochat-students/nanochat-d20

    # Option B – native .pt checkpoint
    python read_lambdas.py --pt ~/.cache/nanochat/chatsft_checkpoints/d32/model_000650.pt

    # Option C – Karpathy's official d32 model via HF
    python read_lambdas.py --hf karpathy/nanochat-d32

Command:
python read_lambdas.py   --pt ~/.cache/nanochat/base_checkpoints/d16_lambdas/model_041600.pt    
"""

import argparse
import sys

import torch
import numpy as np


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

    resid_keys = sorted([k for k in state_dict if "resid_lambda" in k])
    x0_keys    = sorted([k for k in state_dict if "x0_lambda"    in k])

    if not resid_keys:
        # Try flat tensor names used by nanochat's own save format
        resid_keys = sorted([k for k in state_dict if "resid_lambdas" in k])
        x0_keys    = sorted([k for k in state_dict if "x0_lambdas"    in k])

    print(f"Found {len(resid_keys)} resid_lambda keys, {len(x0_keys)} x0_lambda keys")
    return state_dict, resid_keys, x0_keys


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

    resid_keys = sorted([k for k in state_dict if "resid_lambda" in k])
    x0_keys    = sorted([k for k in state_dict if "x0_lambda"    in k])
    print(f"Found {len(resid_keys)} resid_lambda keys, {len(x0_keys)} x0_lambda keys")
    return state_dict, resid_keys, x0_keys


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


def analyse(state_dict, resid_keys, x0_keys):
    if not resid_keys and not x0_keys:
        print("\n[WARNING] No lambda keys found. Printing ALL keys for inspection:")
        for k in sorted(state_dict.keys()):
            print(f"  {k:60s}  shape={tuple(state_dict[k].shape)}")
        return

    # Print raw shapes so the user can see what was loaded
    print("\nRaw tensor shapes:")
    for k in resid_keys + x0_keys:
        print(f"  {k:60s}  shape={tuple(state_dict[k].shape)}")

    resid_vals = _to_float_list(state_dict, resid_keys)
    x0_vals    = _to_float_list(state_dict, x0_keys)

    # ── per-layer table ──────────────────────────────────────────────────
    print("\n" + "─"*60)
    print(f"{'Layer':>6}  {'resid_lambda':>14}  {'x0_lambda':>12}")
    print("─"*60)
    n = max(len(resid_vals), len(x0_vals))
    for i in range(n):
        rl = f"{resid_vals[i]:.6f}" if i < len(resid_vals) else "  —"
        xl = f"{x0_vals[i]:.6f}"   if i < len(x0_vals)    else "  —"
        print(f"{i:>6}  {rl:>14}  {xl:>12}")
    print("─"*60)

    # ── summary statistics ───────────────────────────────────────────────
    if resid_vals:
        rv = np.array(resid_vals, dtype=float)
        print(f"\nresid_lambdas  (init=1.0)")
        print(f"  mean={rv.mean():.4f}  std={rv.std():.4f}  "
              f"min={rv.min():.4f}  max={rv.max():.4f}")
        print(f"  layers with lambda < 1.0 : {(rv < 1.0).sum()}/{len(rv)}")
        print(f"  layers with lambda > 1.0 : {(rv > 1.0).sum()}/{len(rv)}")

    if x0_vals:
        xv = np.array(x0_vals, dtype=float)
        print(f"\nx0_lambdas     (init=0.1)")
        print(f"  mean={xv.mean():.4f}  std={xv.std():.4f}  "
              f"min={xv.min():.4f}  max={xv.max():.4f}")

    # ── theory check ─────────────────────────────────────────────────────
    if resid_vals and x0_vals:
        rv = np.array(resid_vals, dtype=float)
        xv = np.array(x0_vals,   dtype=float)
        # Fixed-point variance (sigma_F^2 ~ 1 at init, V0 ~ 1)
        # V* = (mu^2 * V0) / (1 - lambda) + sigma_F^2 / (1 - lambda^2)
        lam   = rv.mean()
        mu    = xv.mean()
        sigma_F2 = 1.0   # approximate
        V0       = 1.0
        if lam < 1.0:
            Vstar = (mu**2 * V0) / (1 - lam) + sigma_F2 / (1 - lam**2)
            grad_gain = mu / (1 - lam)
            print(f"\n── Theory (mean-field, sigma_F=1, V0=1) ──────────────")
            print(f"  lambda_mean = {lam:.4f},  mu_mean = {mu:.4f}")
            print(f"  Fixed-point variance  V*  = {Vstar:.4f}")
            print(f"  E2E gradient gain  mu/(1-lambda) = {grad_gain:.4f}")
        else:
            print(f"\n[NOTE] mean lambda >= 1 ({lam:.4f}); "
                  "fixed-point analysis requires lambda < 1.")


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

    if args.hf:
        state_dict, resid_keys, x0_keys = load_from_hf(args.hf)
    else:
        state_dict, resid_keys, x0_keys = load_from_pt(args.pt)

    analyse(state_dict, resid_keys, x0_keys)


if __name__ == "__main__":
    main()