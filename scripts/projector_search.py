#!/usr/bin/env python3
"""Experiment 1: Random computational-basis projector scan.

For 2000 random projectors (rank 382 subsets of the 512 computational basis states),
compute the frequency-class epsilon at L=15 with 5 Haar-random seeds.
Record gap ratio = eps(combinatorial peak) / eps(Born frequency).
Save top-50 and bottom-50 projectors.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path
from scipy.linalg import expm

from recohere.analysis import CoherenceResult
from recohere.ising_direct import (
    build_ising_hamiltonian,
    frequency_class_analysis,
)

OUT = Path("results/projector_search")

plt.rcParams.update({
    "font.size": 13, "axes.grid": True, "grid.alpha": 0.25,
    "savefig.dpi": 200, "savefig.bbox": "tight", "font.family": "serif",
})

# Physics parameters (same as existing code)
M = 9
D = 2**M  # 512
L = 15
DT = 1.2
J = 1.0
HX = 0.9045
HZ = 0.0

# Experiment parameters
D1 = 382          # rank of projector (matches p1 ≈ 0.746)
P1 = D1 / D
N_PROJECTORS = 2000
N_SEEDS = 5


def make_initial_state(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    psi0 = rng.standard_normal(D) + 1j * rng.standard_normal(D)
    return (psi0 / np.linalg.norm(psi0)).astype(np.complex128)


def evaluate_projector(args: tuple) -> dict:
    """Evaluate a single projector across multiple seeds.

    Returns dict with gap_ratio, eps_born, eps_comb, mask indices.
    """
    proj_id, mask_indices, U = args
    mask_1 = np.zeros(D, dtype=bool)
    mask_1[mask_indices] = True

    born_n1 = round(L * P1)  # 11
    comb_n1 = L // 2         # 7

    eps_born_list = []
    eps_comb_list = []

    for seed in range(N_SEEDS):
        psi0 = make_initial_state(seed)
        result = frequency_class_analysis(psi0, U, mask_1, L, P1)
        eb = result.epsilon[born_n1]
        ec = result.epsilon[comb_n1]
        if not np.isnan(eb):
            eps_born_list.append(eb)
        if not np.isnan(ec):
            eps_comb_list.append(ec)

    eps_born = np.mean(eps_born_list) if eps_born_list else np.nan
    eps_comb = np.mean(eps_comb_list) if eps_comb_list else np.nan
    gap_ratio = eps_comb / eps_born if eps_born > 0.001 else np.nan

    return {
        "id": proj_id,
        "mask_indices": mask_indices,
        "eps_born": eps_born,
        "eps_comb": eps_comb,
        "gap_ratio": gap_ratio,
    }


def run_experiment_1():
    print(f"Experiment 1: Random projector scan")
    print(f"  D={D}, L={L}, d1={D1} (p1={P1:.3f}), {N_PROJECTORS} projectors, {N_SEEDS} seeds each")

    # Build unitary once
    print("  Building Hamiltonian and unitary...")
    H = build_ising_hamiltonian(M, J, HX, HZ)
    U = expm(-1j * H * DT)

    # Estimate runtime: ~0.15s per (projector, seed) evaluation
    est_total = N_PROJECTORS * N_SEEDS * 0.15
    print(f"  Estimated runtime: {est_total/60:.0f} min (sequential)")
    print(f"  Using {cpu_count()} cores for parallelization")

    # Generate random projectors
    rng = np.random.default_rng(12345)
    all_indices = np.arange(D)
    projector_masks = []
    for i in range(N_PROJECTORS):
        subset = rng.choice(all_indices, size=D1, replace=False)
        subset.sort()
        projector_masks.append(subset)

    # Run evaluations (sequential — U is large, multiprocessing copies it)
    print("  Running evaluations...")
    t0 = time.time()
    results = []
    for i, mask_idx in enumerate(projector_masks):
        r = evaluate_projector((i, mask_idx, U))
        results.append(r)
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (N_PROJECTORS - i - 1) / rate
            print(f"    {i+1}/{N_PROJECTORS} done ({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")

    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Collect results
    gap_ratios = np.array([r["gap_ratio"] for r in results])
    eps_borns = np.array([r["eps_born"] for r in results])
    eps_combs = np.array([r["eps_comb"] for r in results])

    # Sort by gap ratio (higher = better Born filtering)
    valid = ~np.isnan(gap_ratios)
    sorted_idx = np.argsort(gap_ratios[valid])[::-1]
    valid_results = [r for r, v in zip(results, valid) if v]
    sorted_results = [valid_results[i] for i in sorted_idx]

    top50 = sorted_results[:50]
    bottom50 = sorted_results[-50:]

    # Save data
    np.savez(
        OUT / "experiment1_results.npz",
        gap_ratios=gap_ratios,
        eps_borns=eps_borns,
        eps_combs=eps_combs,
        projector_masks=np.array(projector_masks, dtype=object),
        top50_ids=np.array([r["id"] for r in top50]),
        bottom50_ids=np.array([r["id"] for r in bottom50]),
        top50_masks=np.array([r["mask_indices"] for r in top50]),
        bottom50_masks=np.array([r["mask_indices"] for r in bottom50]),
    )

    # Also compute and save the Hamming-weight reference
    from recohere.ising_direct import _build_mask
    hamming_mask = _build_mask(M, "hamming", 4)
    hamming_result = evaluate_projector((-1, np.where(hamming_mask)[0], U))
    print(f"\n  Hamming reference: gap={hamming_result['gap_ratio']:.2f}, "
          f"eps_born={hamming_result['eps_born']:.3f}, eps_comb={hamming_result['eps_comb']:.3f}")

    # Print summary
    valid_gaps = gap_ratios[valid]
    print(f"\n  Results ({np.sum(valid)}/{N_PROJECTORS} valid):")
    print(f"    Gap ratio: mean={np.mean(valid_gaps):.2f}, "
          f"median={np.median(valid_gaps):.2f}, "
          f"std={np.std(valid_gaps):.2f}")
    print(f"    Range: [{np.min(valid_gaps):.2f}, {np.max(valid_gaps):.2f}]")
    print(f"    eps_born: mean={np.nanmean(eps_borns):.3f}")
    print(f"    eps_comb: mean={np.nanmean(eps_combs):.3f}")

    # Plot histogram
    plot_histogram(valid_gaps, hamming_result["gap_ratio"])
    plot_scatter(eps_borns, eps_combs, hamming_result)

    return results, hamming_result


def plot_histogram(gap_ratios, hamming_gap):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(gap_ratios, bins=60, color="steelblue", alpha=0.75, edgecolor="white")
    ax.axvline(hamming_gap, color="crimson", ls="--", lw=2.5,
               label=f"Hamming-weight projector (gap={hamming_gap:.2f})")
    ax.axvline(1.0, color="gray", ls=":", lw=1.5, alpha=0.5, label="No filtering (gap=1)")
    ax.set_xlabel("Gap ratio: ε(comb peak) / ε(Born frequency)", fontsize=14)
    ax.set_ylabel("Count", fontsize=14)
    ax.set_title(
        f"Experiment 1: Random projector scan ({len(gap_ratios)} projectors)\n"
        f"D={D}, L={L}, d₁={D1} (p₁={P1:.3f}), {N_SEEDS} seeds each",
        fontsize=14, fontweight="bold")
    ax.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "exp1_gap_histogram.png")
    fig.savefig(OUT / "exp1_gap_histogram.pdf")
    plt.close(fig)
    print("  -> exp1_gap_histogram.png")


def plot_scatter(eps_borns, eps_combs, hamming_ref):
    fig, ax = plt.subplots(figsize=(10, 10))
    valid = ~(np.isnan(eps_borns) | np.isnan(eps_combs))
    ax.scatter(eps_borns[valid], eps_combs[valid], alpha=0.3, s=15, color="steelblue",
               label="Random projectors")
    ax.scatter(hamming_ref["eps_born"], hamming_ref["eps_comb"],
               color="crimson", s=150, marker="*", zorder=10,
               label="Hamming-weight projector")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="ε_comb = ε_born (no gap)")
    ax.set_xlabel("ε at Born frequency", fontsize=14)
    ax.set_ylabel("ε at combinatorial peak", fontsize=14)
    ax.set_title("Born vs combinatorial-peak recoherence", fontsize=14, fontweight="bold")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "exp1_scatter.png")
    fig.savefig(OUT / "exp1_scatter.pdf")
    plt.close(fig)
    print("  -> exp1_scatter.png")


if __name__ == "__main__":
    results, hamming_ref = run_experiment_1()
