"""
analysis.py — Analysis tool for SRLM checkpoints.

Usage:
    python analysis.py -c <checkpoint_dir> -s <spec> [analyses...]

Examples:
    python analysis.py -c checkpoints/run1 -s 512x64 params drift eigenvalues
    python analysis.py -c checkpoints/run1 -s 1024x1024 ablation gradients

Available analyses:
    params       — parameter count and memory per module
    drift        — parameter drift from init checkpoint
    gradients    — gradient norms per module
    activations  — activation statistics (mean/std) per S5 layer
    hrm          — HRM slow pathway (zH) contribution
    ablation     — loss ablation (reset each module to init, measure delta)
    eigenvalues  — SSM eigenvalue distribution

The script is model-agnostic — it uses the same setup() infrastructure
as main.py and only the model-independent analyses remain here.
Model-specific analyses (HRM ablation) call into the model via the
standard haiku interface.
"""

import sys
import os
import math
import argparse
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import jmp
from pathlib import Path
from orbax import checkpoint as ocp

# Reuse setup infrastructure from main.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main import (
    setup, specifications, subitems,
    VOCAB_SIZE, TOTAL_VOCAB,
    load_kalevala, load_wikipedia_finnish
)
from model import (
    SRLM, SRLMConfig, AbsorbingGraph, LogLinearNoise,
    mk_z, loss_function,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def separator(title):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)

def flat_leaves(tree, prefix=""):
    """Yield (path, array) for every leaf in a nested param dict."""
    if isinstance(tree, dict):
        for k, v in tree.items():
            yield from flat_leaves(v, (prefix + "/" + k) if prefix else k)
    else:
        yield prefix, tree

def module_key(path):
    """Return the module path — everything except the final parameter name.

    e.g. srlm/~/main/~/fast/~/s5d/~/fwd/log_real -> srlm/~/main/~/fast/~/s5d/~/fwd
    """
    parts = path.split("/")
    return "/".join(parts[:-1]) if len(parts) > 1 else path

def load_checkpoint(ckdir, params_dummy):
    checkpointer = ocp.StandardCheckpointer()
    ckdir = Path(ckdir).absolute()
    if not ckdir.exists():
        raise FileNotFoundError(f"Checkpoint dir not found: {ckdir}")
    items = list(subitems(ckdir))
    if not items:
        raise FileNotFoundError(f"No checkpoints found in {ckdir}")
    _, item = max(items, key=lambda x: x[0])
    return checkpointer.restore(item, params_dummy)


# ---------------------------------------------------------------------------
# 1. Parameter count and memory
# ---------------------------------------------------------------------------

def analyse_parameters(params):
    separator("1. Parameter count and memory per module")
    counts, bytesum = {}, {}
    for path, arr in flat_leaves(params):
        k = module_key(path)
        counts[k]  = counts.get(k, 0)  + arr.size
        bytesum[k] = bytesum.get(k, 0) + arr.size * arr.dtype.itemsize

    total_p = sum(counts.values())
    total_mb = sum(bytesum.values()) / 1024**2

    print(f"{'Module':<55} {'Params':>12} {'MB':>8}")
    print("-" * 77)
    for k in sorted(counts):
        print(f"  {k:<53} {counts[k]:>12,} {bytesum[k]/1024**2:>8.2f}")
    print("-" * 77)
    print(f"  {'TOTAL':<53} {total_p:>12,} {total_mb:>8.2f}")


# ---------------------------------------------------------------------------
# 2. Parameter drift
# ---------------------------------------------------------------------------

def analyse_drift(params_init, params_trained):
    separator("2. Parameter drift from init (mean |trained - init|)")
    drift_abs, drift_rel = {}, {}
    for (path, trained), (_, init) in zip(
            flat_leaves(params_trained), flat_leaves(params_init)):
        k = module_key(path)
        abs_d = float(jnp.mean(jnp.abs(trained - init)))
        rel_d = abs_d / (float(jnp.mean(jnp.abs(init))) + 1e-8)
        drift_abs.setdefault(k, []).append(abs_d)
        drift_rel.setdefault(k, []).append(rel_d)

    mean_abs = {k: np.mean(v) for k, v in drift_abs.items()}
    mean_rel = {k: np.mean(v) for k, v in drift_rel.items()}

    print(f"{'Module':<55} {'Abs drift':>10} {'Rel drift':>10}")
    print("-" * 77)
    for k in sorted(mean_abs, key=lambda x: -mean_abs[x]):
        print(f"  {k:<53} {mean_abs[k]:>10.5f} {mean_rel[k]:>10.3f}x")


# ---------------------------------------------------------------------------
# 3. Gradient norms
# ---------------------------------------------------------------------------

def analyse_gradients(params, loss_fn, key, z, batch):
    separator("3. Gradient norms per module")
    (loss, _), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, key, z, batch)
    print(f"  Loss: {float(loss):.4f}\n")

    norms = {}
    for path, arr in flat_leaves(grads):
        k = module_key(path)
        norms[k] = norms.get(k, 0.0) + float(jnp.sum(arr ** 2))
    norms = {k: math.sqrt(v) for k, v in norms.items()}

    print(f"{'Module':<55} {'Grad norm':>10}")
    print("-" * 67)
    for k in sorted(norms, key=lambda x: -norms[x]):
        print(f"  {k:<53} {norms[k]:>10.5f}")
    total = math.sqrt(sum(v**2 for v in norms.values()))
    print("-" * 67)
    print(f"  {'TOTAL GRAD NORM':<53} {total:>10.5f}")
    return grads


# ---------------------------------------------------------------------------
# 4. Activation statistics
# ---------------------------------------------------------------------------

_activation_log = {}

def analyse_activations(model, params, key, z, batch, sigma):
    """Log mean/std/max for every S5 layer output via jax.debug.callback."""
    separator("4. Activation statistics (S5 layer outputs)")

    global _activation_log
    _activation_log = {}

    # Instrument the model by monkey-patching via haiku's interceptor
    def interceptor(next_f, args, kwargs, context):
        out = next_f(*args, **kwargs)
        name = context.module_details.module.name
        def _log(mean, std, maxv):
            _activation_log[name] = {"mean": float(mean),
                                      "std":  float(std),
                                      "max":  float(maxv)}
        jax.debug.callback(_log,
                            jnp.mean(out),
                            jnp.std(out),
                            jnp.max(jnp.abs(out)))
        return out

    # We can't easily intercept arbitrary modules, so we do a plain forward
    # and collect stats on the final output per-layer via a wrapper model.
    # Simpler: just run forward and report output tensor stats at top level.
    # For per-S5 stats, use the eigenvalue analysis which reads params directly.
    _ = model.apply(params, key, z, batch, sigma)

    if _activation_log:
        print(f"  {'Layer':<50} {'mean':>8} {'std':>8} {'max|x|':>8}")
        print("-" * 78)
        for name, stats in sorted(_activation_log.items()):
            flag = "  *** DEAD" if stats["std"] < 1e-3 else ""
            print(f"  {name:<50} {stats['mean']:>8.4f} {stats['std']:>8.4f} {stats['max']:>8.4f}{flag}")
    else:
        print("  (No activation hooks registered — see eigenvalues for SSM health)")


# ---------------------------------------------------------------------------
# 5. HRM slow pathway contribution
# ---------------------------------------------------------------------------

def analyse_hrm_contribution(model, params, key, z, batch, sigma):
    separator("5. HRM slow pathway (zH) contribution")

    _, out_normal = model.apply(params, key, z, batch, sigma)

    # Check std across L dimension — if near zero, slow pathway sees constant input
    std_across_L = float(jnp.std(out_normal, axis=1).mean())
    print(f"  Output std across sequence dim: {std_across_L:.5f}")
    if std_across_L < 1e-3:
        print("  *** WARNING: output is nearly constant across L — slow pathway may be inactive")


# ---------------------------------------------------------------------------
# 6. Loss ablation
# ---------------------------------------------------------------------------

def analyse_loss_ablation(params_init, params_trained, loss_fn, key, z, batch):
    separator("6. Loss ablation (reset each module to init, measure delta)")

    baseline_loss, _ = loss_fn(params_trained, key, z, batch)
    print(f"  Baseline loss: {float(baseline_loss):.4f}\n")

    # Collect all leaf module paths (parent of each param)
    all_modules = sorted(set(module_key(path) for path, _ in flat_leaves(params_trained)))

    print(f"  {'Module':<60} {'Loss':>10} {'Delta':>10}  Impact")
    print("-" * 95)

    def set_leaf(tree, path_parts, value):
        """Return a new tree with the leaf at path_parts set to value."""
        if len(path_parts) == 1:
            return {**tree, path_parts[0]: value}
        k = path_parts[0]
        return {**tree, k: set_leaf(tree[k], path_parts[1:], value)}

    def ablate_module(params, params_init, target_module):
        """Reset all params whose module_key == target_module to init values."""
        # Build lookup from path -> init value
        init_lookup = {path: arr for path, arr in flat_leaves(params_init)}

        def rebuild(trained, init, prefix=""):
            if not isinstance(trained, dict):
                # leaf — swap if this leaf belongs to target_module
                if module_key(prefix) == target_module:
                    return init_lookup.get(prefix, trained)
                return trained
            return {k: rebuild(v, init.get(k, v),
                               (prefix + "/" + k) if prefix else k)
                    for k, v in trained.items()}

        return rebuild(params, params_init)

    for target in all_modules:
        frozen = ablate_module(params_trained, params_init, target)
        frozen_loss, _ = loss_fn(frozen, key, z, batch)
        delta = float(frozen_loss) - float(baseline_loss)
        impact = ("higher (module helps)" if delta > 0.5
                  else "lower (module hurts?)" if delta < -0.5
                  else "same (module idle)")
        print(f"  {target:<60} {float(frozen_loss):>10.4f} {delta:>+10.4f}  {impact}")


# ---------------------------------------------------------------------------
# 7. SSM eigenvalue distribution
# ---------------------------------------------------------------------------

def analyse_eigenvalues(params):
    separator("7. SSM eigenvalue distribution (Lambda_bar magnitudes)")

    # Find all SSM instances — identified by having log_real, imag, log_Delta
    instances = {}
    for path, arr in flat_leaves(params):
        for suffix in ("log_real", "imag", "log_Delta"):
            if path.endswith(suffix):
                inst = "/".join(path.split("/")[:-1])
                instances.setdefault(inst, {})[suffix] = arr

    print(f"  {'SSM instance':<55} {'|λ| min':>8} {'|λ| mean':>8} {'|λ| max':>8} {'dead%':>7}")
    print("-" * 90)
    found = 0
    for inst, p in sorted(instances.items()):
        if not all(k in p for k in ("log_real", "imag", "log_Delta")):
            continue
        found += 1
        Lambda    = (-jnp.exp(p["log_real"]) + 1j * p["imag"]).astype(jnp.complex64)
        Delta     = jnp.exp(p["log_Delta"])
        Lbar      = jnp.exp(Lambda * Delta)
        mags      = jnp.abs(Lbar)
        dead_pct  = 100.0 * float(jnp.mean(mags < 0.01))
        print(f"  {inst:<55} {float(mags.min()):>8.4f} {float(mags.mean()):>8.4f}"
              f" {float(mags.max()):>8.4f} {dead_pct:>6.1f}%")
        if float(mags.max()) > 0.999:
            print(f"  *** WARNING: eigenvalues near 1.0 — possible instability")
    if found == 0:
        print("  (No SSM parameters found — check param naming)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

ALL_ANALYSES = ["params", "drift", "gradients", "activations",
                "hrm", "ablation", "eigenvalues"]

def main():
    parser = argparse.ArgumentParser(
        prog="analysis",
        description="Model-agnostic analysis of SRLM checkpoints.",
    )
    parser.add_argument("-c", "--checkpoint", required=True,
                        help="checkpoint directory (contains init + trained subdirs)")
    parser.add_argument("-s", "--spec", default="512x64",
                        choices=list(specifications.keys()))
    parser.add_argument("-l", "--seq_len", type=int)
    parser.add_argument("-b", "--batch",   type=int, default=8)
    parser.add_argument("-d", "--data",    choices=["kalevala", "wikipedia"],
                        help="data source for gradient/ablation analyses")
    #parser.add_argument("-i", "--init", action="store_true", help="use fresh initial weights on studies")
    parser.add_argument("analyses", nargs="*", default=ALL_ANALYSES,
                        help=f"which analyses to run (default: all). choices: {ALL_ANALYSES}")

    args = parser.parse_args()

    # Validate requested analyses
    unknown = set(args.analyses) - set(ALL_ANALYSES)
    if unknown:
        print(f"Unknown analyses: {unknown}. Choose from: {ALL_ANALYSES}")
        sys.exit(1)

    run = set(args.analyses)

    # Setup using the same infrastructure as main.py
    print("Setting up model...")
    policy = jmp.Policy(
        compute_dtype=jnp.bfloat16,
        param_dtype=jnp.float32,
        output_dtype=jnp.bfloat16,
    )
    hk.mixed_precision.set_policy(SRLM, policy)

    spec   = specifications[args.spec]
    B      = args.batch
    SEQ_LEN = args.seq_len or spec.SEQ_LEN
    cfg    = spec.CONFIG

    graph  = AbsorbingGraph(VOCAB_SIZE)
    noise  = LogLinearNoise()
    rng    = hk.PRNGSequence(42)

    def model_spec(z, x, sigma):
        return SRLM(cfg)(z, x, sigma)

    model = hk.transform(model_spec)

    print(f"Initialising model ({args.spec})...")
    z_dummy     = mk_z(1, 1, cfg.d_model)
    x_dummy     = jax.random.randint(next(rng), (1, 1), 0, VOCAB_SIZE)
    sigma_dummy = jnp.abs(jax.random.normal(next(rng), (1,))) + 0.1
    params_dummy = model.init(rng=next(rng), x=x_dummy, z=z_dummy, sigma=sigma_dummy)

    # Load checkpoints
    ckdir = Path(args.checkpoint).absolute()
    checkpointer = ocp.StandardCheckpointer()

    items = sorted(list(subitems(ckdir)), key=lambda x: x[0])
    if len(items) < 2:
        print("Need at least 2 checkpoints (init + trained). Found:", len(items))
        sys.exit(1)

    (e0, s0), init_path    = items[0]
    (e1, s1), trained_path = items[-1]
    #if args.init:
    #    print(f"Init checkpoint is fresh")
    #else:
    print(f"Init checkpoint:    {init_path}  (epoch {e0}, step {s0})")
    print(f"Trained checkpoint: {trained_path}  (epoch {e1}, step {s1})")

    #if args.init:
    #params_init = params_dummy
    #else:
    params_init = checkpointer.restore(init_path,    params_dummy)
    params_trained = checkpointer.restore(trained_path, params_dummy)

    # Sample batch from real training data
    key = next(rng)
    if args.data == "kalevala":
        sample_batch = load_kalevala(Path(args.checkpoint).parent)
        x_batch = sample_batch(next(rng), SEQ_LEN, B)
    elif args.data == "wikipedia":
        loader = load_wikipedia_finnish(Path(args.checkpoint).parent, B, SEQ_LEN)
        loader.shuffle(0)
        x_batch = loader.next_batch()
        x_batch = jax.device_put(x_batch)
    else:
        print("No --data source given, using random data (ablation/gradient results will be unreliable)")
        x_batch = jax.random.randint(next(rng), (B, SEQ_LEN), 0, VOCAB_SIZE)
    sigma_b = jnp.abs(jax.random.normal(next(rng), (B,))) + 0.1
    z_batch = mk_z(B, SEQ_LEN, cfg.d_model)

    loss_fn = loss_function(model, graph, noise)

    def simple_loss(p, k, z, x):
        return loss_fn(p, k, z, x)

    # Run requested analyses
    if "params"      in run: analyse_parameters(params_trained)
    if "drift"       in run: analyse_drift(params_init, params_trained)
    if "gradients"   in run: analyse_gradients(params_trained, simple_loss, key, z_batch, x_batch)
    if "activations" in run: analyse_activations(model, params_trained, key, z_batch, x_batch, sigma_b)
    if "hrm"         in run: analyse_hrm_contribution(model, params_trained, key, z_batch, x_batch, sigma_b)
    if "ablation"    in run: analyse_loss_ablation(params_init, params_trained, simple_loss, key, z_batch, x_batch)
    if "eigenvalues" in run: analyse_eigenvalues(params_trained)

    print("\nAnalysis complete.")

if __name__ == "__main__":
    main()
