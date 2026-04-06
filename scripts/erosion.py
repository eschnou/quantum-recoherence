#!/usr/bin/env python3
"""Erosion analysis: probing the mechanism behind Born-rule filtering.

Three measurements:
  A. Overlap vs last-divergence age (shared suffix length)
  B. Overlap vs suffix content (how many 1-projectors in the shared suffix)
  C. Ising vs Haar-random vs product control side by side
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from pathlib import Path

from recohere.branches import (
    BranchResult,
    simulate_branches,
    simulate_branches_haar,
    simulate_branches_product,
)
from recohere.ising_direct import IsingDirectParams

OUT = Path("results")
OUT.mkdir(exist_ok=True)
plt.rcParams.update({
    "font.size": 13, "axes.grid": True, "grid.alpha": 0.25,
    "savefig.dpi": 200, "savefig.bbox": "tight", "font.family": "serif",
})

M, THR = 9, 4
D = 2**M
DT, SEED = 1.2, 42


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def shared_suffix_length(h1: tuple[int, ...], h2: tuple[int, ...]) -> int:
    """Number of matching steps from the end of two histories."""
    k = 0
    L = len(h1)
    while k < L and h1[-(k + 1)] == h2[-(k + 1)]:
        k += 1
    return k


def suffix_n1(h: tuple[int, ...], k: int) -> int:
    """Count of 1-outcomes in the last k steps of history h."""
    if k == 0:
        return 0
    return sum(h[-k:])


def pairwise_overlap_squared(r: BranchResult) -> np.ndarray:
    """Compute |<psi_i|psi_j>|^2 / (w_i * w_j) for all pairs, using the Gram matrix.

    Returns (N, N) array of normalized overlap squared. Only upper triangle is meaningful.
    """
    if r.gram_normalized is not None:
        return np.abs(r.gram_normalized) ** 2
    # Fall back to direct computation
    from recohere.branches import _normalize_states
    states_norm = _normalize_states(r.branch_states, r.weights)
    G = states_norm @ states_norm.conj().T
    return np.abs(G) ** 2


# ---------------------------------------------------------------------------
# A. Overlap vs last-divergence age (shared suffix length)
# ---------------------------------------------------------------------------

def measure_overlap_vs_suffix_length(r: BranchResult) -> dict[int, list[float]]:
    """Group all branch pairs by shared suffix length, collect overlaps."""
    overlap_sq = pairwise_overlap_squared(r)
    N = len(r.histories)
    L = len(r.histories[0])

    groups: dict[int, list[float]] = defaultdict(list)
    for i in range(N):
        for j in range(i + 1, N):
            k = shared_suffix_length(r.histories[i], r.histories[j])
            groups[k].append(overlap_sq[i, j])

    return groups


def plot_overlap_vs_suffix_length(results_by_dynamics: dict[str, BranchResult], L: int):
    """Plot A: mean |overlap|^2 vs shared suffix length, one curve per dynamics."""
    fig, ax = plt.subplots(figsize=(10, 7))

    colors = {"Ising": "royalblue", "Haar": "firebrick", "Product": "seagreen"}

    for name, r in results_by_dynamics.items():
        groups = measure_overlap_vs_suffix_length(r)
        ks = sorted(k for k in groups if k > 0)
        means = [np.mean(groups[k]) for k in ks]
        stds = [np.std(groups[k]) / max(np.sqrt(len(groups[k])), 1) for k in ks]

        color = colors.get(name, "gray")
        ax.errorbar(ks, means, yerr=stds, fmt="o-", color=color, lw=2, ms=7,
                    capsize=4, label=name)

    ax.set_xlabel("Shared suffix length k (identical trailing projections)", fontsize=14)
    ax.set_ylabel("Mean |overlap|$^2$ (normalized)", fontsize=14)
    ax.set_title(
        f"A. Overlap vs shared suffix length — L={L}, {M} qubits (D={D})\n"
        f"Each shared projection step erodes the memory of past divergence",
        fontsize=14, fontweight="bold")
    ax.set_yscale("log")
    ax.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "erosion_suffix_length.png")
    fig.savefig(OUT / "erosion_suffix_length.pdf")
    plt.close(fig)
    print("  -> erosion_suffix_length.png")


# ---------------------------------------------------------------------------
# B. Overlap vs suffix content (n1 in the shared suffix)
# ---------------------------------------------------------------------------

def measure_overlap_vs_suffix_content(r: BranchResult) -> dict[int, dict[int, list[float]]]:
    """Group pairs by (suffix_length, n1_in_suffix), collect overlaps.

    Returns {k: {n1_suffix: [overlap_sq, ...]}}.
    For n1_suffix we use the average of n1 in the two branches' suffixes
    (they share the suffix, so it's the same).
    """
    overlap_sq = pairwise_overlap_squared(r)
    N = len(r.histories)
    L = len(r.histories[0])

    groups: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for i in range(N):
        for j in range(i + 1, N):
            k = shared_suffix_length(r.histories[i], r.histories[j])
            if k == 0:
                continue  # no shared suffix → no content to analyze
            # The shared suffix is identical for both branches
            n1_s = suffix_n1(r.histories[i], k)
            groups[k][n1_s].append(overlap_sq[i, j])

    return groups


def plot_overlap_vs_suffix_content(r: BranchResult, L: int, name: str = "Ising"):
    """Plot B: for each suffix length k, show overlap vs fraction of 1s in suffix."""
    content_groups = measure_overlap_vs_suffix_content(r)

    # Pick suffix lengths with enough data
    ks_to_plot = sorted(k for k in content_groups if len(content_groups[k]) >= 2)
    if not ks_to_plot:
        print("  -> (skipping suffix content plot: not enough data)")
        return

    # Limit to at most 6 panels
    if len(ks_to_plot) > 6:
        ks_to_plot = ks_to_plot[:6]

    n_panels = len(ks_to_plot)
    cols = min(n_panels, 3)
    rows = (n_panels + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)

    for idx, k in enumerate(ks_to_plot):
        ax = axes[idx // cols][idx % cols]
        by_n1 = content_groups[k]
        n1_vals = sorted(by_n1.keys())
        means = [np.mean(by_n1[n]) for n in n1_vals]
        stds = [np.std(by_n1[n]) / max(np.sqrt(len(by_n1[n])), 1) for n in n1_vals]
        counts = [len(by_n1[n]) for n in n1_vals]

        fracs = [n / k for n in n1_vals]
        ax.errorbar(fracs, means, yerr=stds, fmt="s-", color="royalblue", lw=2, ms=7, capsize=4)

        for f, m, c in zip(fracs, means, counts):
            ax.annotate(f"n={c}", (f, m), fontsize=7,
                        textcoords="offset points", xytext=(5, 5), color="gray")

        ax.set_xlabel("Fraction of 1-projectors in suffix", fontsize=12)
        ax.set_ylabel("Mean |overlap|$^2$", fontsize=12)
        ax.set_title(f"Suffix length k={k}", fontsize=13, fontweight="bold")
        if max(means) > 10 * min(means) and min(means) > 0:
            ax.set_yscale("log")

    # Hide unused
    for idx in range(n_panels, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle(
        f"B. Overlap vs suffix content ({name}) — L={L}, {M} qubits (D={D})\n"
        f"Does the decay rate depend on how many 1-projectors are in the shared suffix?",
        fontsize=14, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(OUT / "erosion_suffix_content.png")
    fig.savefig(OUT / "erosion_suffix_content.pdf")
    plt.close(fig)
    print("  -> erosion_suffix_content.png")


# ---------------------------------------------------------------------------
# C. Three-dynamics comparison (epsilon vs frequency, side by side)
# ---------------------------------------------------------------------------

def plot_dynamics_comparison(results_by_dynamics: dict[str, BranchResult], L: int):
    """Plot C: epsilon vs frequency scatter, one panel per dynamics type."""
    n_dyn = len(results_by_dynamics)
    fig, axes = plt.subplots(1, n_dyn, figsize=(8 * n_dyn, 7))
    if n_dyn == 1:
        axes = [axes]

    EPS_THRESHOLD = 0.3

    for ax, (name, r) in zip(axes, results_by_dynamics.items()):
        freq = r.n1 / L
        log_w = np.log10(np.maximum(r.weights, 1e-50))

        sc = ax.scatter(freq, r.epsilon, c=log_w, s=10, alpha=0.6,
                        cmap="viridis", vmin=log_w.min(), vmax=0, rasterized=True)
        ax.axhline(EPS_THRESHOLD, color="gray", ls="--", lw=1, alpha=0.5)
        ax.axvline(r.p1, color="crimson", ls="--", lw=2, alpha=0.7, label=f"Born p₁={r.p1:.3f}")
        ax.axvline(0.5, color="seagreen", ls=":", lw=2, alpha=0.7, label="Combinatorial 0.5")

        n_alive = int(np.sum(r.epsilon < EPS_THRESHOLD))
        ax.set_title(f"{name}\n{len(r.histories)} branches, {n_alive} decoherent", fontsize=14)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.03, 1.05)
        ax.set_xlabel("Frequency n₁/L", fontsize=13)
        ax.set_ylabel("ε (max overlap)", fontsize=13)
        ax.legend(fontsize=10, loc="upper left")

    fig.suptitle(
        f"C. Dynamics comparison — L={L}, {M} qubits (D={D})\n"
        f"Same projectors, same initial state seed. Color = log₁₀ weight.",
        fontsize=15, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(OUT / "erosion_dynamics_comparison.png")
    fig.savefig(OUT / "erosion_dynamics_comparison.pdf")
    plt.close(fig)
    print("  -> erosion_dynamics_comparison.png")


# ---------------------------------------------------------------------------
# D. Overlap decay fit — extract erosion rate per dynamics
# ---------------------------------------------------------------------------

def plot_erosion_rate_fit(results_by_dynamics: dict[str, BranchResult], L: int):
    """Fit exponential growth of overlap with suffix length, extract erosion rate."""
    fig, ax = plt.subplots(figsize=(10, 7))

    colors = {"Ising": "royalblue", "Haar": "firebrick", "Product": "seagreen"}

    for name, r in results_by_dynamics.items():
        groups = measure_overlap_vs_suffix_length(r)
        ks = sorted(k for k in groups if k > 0 and len(groups[k]) >= 5)
        if len(ks) < 2:
            continue

        means = np.array([np.mean(groups[k]) for k in ks])
        ks_arr = np.array(ks, dtype=float)

        # Log-linear fit: log(overlap) = gamma * k + const
        valid = means > 0
        if valid.sum() < 2:
            continue
        log_means = np.log(means[valid])
        ks_fit = ks_arr[valid]
        coeffs = np.polyfit(ks_fit, log_means, 1)
        gamma = coeffs[0]  # positive = overlap grows with k = erosion works
        fit_line = np.exp(np.polyval(coeffs, ks_fit))

        color = colors.get(name, "gray")
        ax.semilogy(ks_arr, means, "o", color=color, ms=8, label=f"{name} (γ={gamma:.2f})")
        ax.semilogy(ks_fit, fit_line, "--", color=color, lw=2, alpha=0.7)

    ax.set_xlabel("Shared suffix length k", fontsize=14)
    ax.set_ylabel("Mean |overlap|$^2$", fontsize=14)
    ax.set_title(
        f"Erosion rate — L={L}, {M} qubits (D={D})\n"
        f"Fit: ⟨|overlap|²⟩ ~ exp(γk). Larger γ = faster erosion of distinguishability.",
        fontsize=14, fontweight="bold")
    ax.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "erosion_rate_fit.png")
    fig.savefig(OUT / "erosion_rate_fit.pdf")
    plt.close(fig)
    print("  -> erosion_rate_fit.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    L = 10  # 1024 branches: tractable pairwise analysis (~500K pairs)

    params = IsingDirectParams(m=M, L=L, hamming_threshold=THR, dt=DT, seed=SEED)
    p1 = params.p1
    print(f"Erosion analysis: m={M} (D={D}), L={L}, p1={p1:.3f}\n")

    # Run all three dynamics
    print("Simulating Ising branches ...", flush=True)
    r_ising = simulate_branches(params, gram=True)
    print(f"  {len(r_ising.histories)} branches, "
          f"{int(np.sum(r_ising.epsilon < 0.3))} decoherent")

    print("Simulating Haar-random branches ...", flush=True)
    r_haar = simulate_branches_haar(params, gram=True)
    print(f"  {len(r_haar.histories)} branches, "
          f"{int(np.sum(r_haar.epsilon < 0.3))} decoherent")

    print("Simulating Product branches ...", flush=True)
    r_product = simulate_branches_product(params, gram=True)
    print(f"  {len(r_product.histories)} branches, "
          f"{int(np.sum(r_product.epsilon < 0.3))} decoherent")

    dynamics = {"Ising": r_ising, "Haar": r_haar, "Product": r_product}

    print("\nComputing pairwise overlap statistics ...", flush=True)

    # A. Overlap vs suffix length
    print("\nPlot A: Overlap vs divergence age")
    plot_overlap_vs_suffix_length(dynamics, L)

    # B. Overlap vs suffix content (Ising only — most interesting)
    print("Plot B: Overlap vs suffix content")
    plot_overlap_vs_suffix_content(r_ising, L, name="Ising")

    # C. Dynamics comparison
    print("Plot C: Dynamics comparison")
    plot_dynamics_comparison(dynamics, L)

    # D. Erosion rate fit
    print("Plot D: Erosion rate fit")
    plot_erosion_rate_fit(dynamics, L)

    print(f"\nAll erosion plots saved to {OUT}/")


if __name__ == "__main__":
    main()
