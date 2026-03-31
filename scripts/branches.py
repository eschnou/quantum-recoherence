#!/usr/bin/env python3
"""Branch-level analysis: watching worlds die as Hilbert space fills up."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from math import comb
from pathlib import Path
from scipy.stats import binom

from recohere.branches import BranchResult, simulate_branches
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
EPS_THRESHOLD = 0.3
L_VALUES = [4, 6, 8, 9, 10, 12, 13]


def run_all() -> dict[int, BranchResult]:
    results = {}
    p1 = IsingDirectParams(m=M, L=1, hamming_threshold=THR).p1
    print(f"Branch analysis: m={M} (D={D}), p1={p1:.3f}, seed={SEED}")
    for L in L_VALUES:
        print(f"  L={L}: {2**L} branches ...", end=" ", flush=True)
        p = IsingDirectParams(m=M, L=L, hamming_threshold=THR,
                              dt=DT, seed=SEED)
        r = simulate_branches(p)
        n_alive = np.sum(r.epsilon < EPS_THRESHOLD)
        print(f"{len(r.histories)} surviving, {n_alive} decoherent (eps<{EPS_THRESHOLD})")
        results[L] = r
    return results


def plot_death_of_worlds(results: dict[int, BranchResult]):
    """Scatter: epsilon vs frequency, one panel per L."""
    fig, axes = plt.subplots(2, 4, figsize=(24, 11))

    for ax, L in zip(axes.flatten(), L_VALUES):
        r = results[L]
        freq = r.n1 / L
        log_w = np.log10(np.maximum(r.weights, 1e-50))

        sc = ax.scatter(freq, r.epsilon, c=log_w, s=8, alpha=0.6,
                        cmap="viridis", vmin=log_w.min(), vmax=0, rasterized=True)
        ax.axhline(EPS_THRESHOLD, color="gray", ls="--", lw=1, alpha=0.5)
        ax.axvline(r.p1, color="crimson", ls="--", lw=2, alpha=0.7)
        ax.axvline(0.5, color="seagreen", ls=":", lw=2, alpha=0.7)

        n_alive = np.sum(r.epsilon < EPS_THRESHOLD)
        ax.set_title(f"L={L}   ({2**L} branches, {n_alive} decoherent)", fontsize=13)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.03, 1.05)
        ax.set_xlabel("frequency n₁/L")
        ax.set_ylabel("ε (max overlap)")

    # Hide unused panels
    for i in range(len(L_VALUES), len(axes.flatten())):
        axes.flatten()[i].set_visible(False)

    fig.suptitle(
        f"Branch decoherence — {M} qubits (D={D}), p₁={results[L_VALUES[0]].p1:.3f}\n"
        f"Red = Born frequency, green = combinatorial peak, color = log₁₀ weight",
        fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "branch_death.png")
    fig.savefig(OUT / "branch_death.pdf")
    plt.close(fig)
    print("  -> branch_death.png")


def plot_branch_census(results: dict[int, BranchResult]):
    """Count of decoherent branches vs L."""
    L_all = sorted(results.keys())
    total = [2**L for L in L_all]
    alive = [int(np.sum(results[L].epsilon < EPS_THRESHOLD)) for L in L_all]

    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.semilogy(L_all, total, "o-", color="firebrick", lw=2.5, ms=10,
                label="Total branches (2ᴸ)", zorder=5)
    ax.semilogy(L_all, alive, "s-", color="royalblue", lw=2.5, ms=10,
                label=f"Decoherent branches (ε < {EPS_THRESHOLD})", zorder=5)
    ax.axhline(D, color="black", ls="--", lw=2, alpha=0.5,
               label=f"Hilbert space dimension D = {D}")

    ax.set_xlabel("History length L", fontsize=14)
    ax.set_ylabel("Number of branches", fontsize=14)
    ax.set_title(
        f"Branch census — {M} qubits (D={D})\n"
        f"Decoherent branches saturate at ~D while total grows exponentially",
        fontsize=14, fontweight="bold")
    ax.legend(fontsize=12)
    ax.set_xticks(L_all)
    fig.tight_layout()
    fig.savefig(OUT / "branch_census.png")
    fig.savefig(OUT / "branch_census.pdf")
    plt.close(fig)
    print("  -> branch_census.png")


def plot_survivor_histogram(results: dict[int, BranchResult]):
    """Frequency distribution of surviving branches."""
    L = max(L_VALUES)
    r = results[L]
    alive = r.epsilon < EPS_THRESHOLD

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

    # Left: unweighted count of survivors
    n1_alive = r.n1[alive]
    bins = np.arange(-0.5, L + 1.5, 1)
    ax = axes[0]
    ax.hist(n1_alive, bins=bins, color="royalblue", alpha=0.7, edgecolor="white",
            label="Surviving branches")

    # Born binomial overlay (scaled to match total count)
    n1_range = np.arange(0, L + 1)
    born_pmf = binom.pmf(n1_range, L, r.p1)
    born_scaled = born_pmf * len(n1_alive)
    ax.plot(n1_range, born_scaled, "o-", color="crimson", lw=2, ms=6,
            label=f"Born: Binom(L={L}, p₁={r.p1:.3f})")

    # Combinatorial overlay
    comb_pmf = np.array([comb(L, k) for k in n1_range]) / 2**L
    comb_scaled = comb_pmf * len(n1_alive)
    ax.plot(n1_range, comb_scaled, "s--", color="seagreen", lw=2, ms=6,
            label="Combinatorial: C(L,n₁)/2ᴸ")

    ax.set_xlabel("n₁ (number of '1' outcomes)", fontsize=13)
    ax.set_ylabel("Count of surviving branches", fontsize=13)
    ax.set_title(f"Survivors at L={L} (unweighted)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)

    # Right: weight-weighted histogram
    ax = axes[1]
    ax.hist(r.n1[alive], bins=bins, weights=r.weights[alive],
            color="royalblue", alpha=0.7, edgecolor="white",
            label="Surviving branches (weighted)")

    total_w = r.weights[alive].sum()
    ax.plot(n1_range, born_pmf * total_w, "o-", color="crimson", lw=2, ms=6,
            label=f"Born: Binom(L={L}, p₁={r.p1:.3f})")
    ax.plot(n1_range, comb_pmf * total_w, "s--", color="seagreen", lw=2, ms=6,
            label="Combinatorial: C(L,n₁)/2ᴸ")

    ax.set_xlabel("n₁ (number of '1' outcomes)", fontsize=13)
    ax.set_ylabel("Total weight of survivors", fontsize=13)
    ax.set_title(f"Survivors at L={L} (weight-weighted)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)

    fig.suptitle(
        f"Which worlds survive? — {M} qubits (D={D}), ε < {EPS_THRESHOLD}\n"
        f"{int(alive.sum())} branches survive out of {2**L}",
        fontsize=15, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(OUT / "branch_survivors.png")
    fig.savefig(OUT / "branch_survivors.pdf")
    plt.close(fig)
    print("  -> branch_survivors.png")


def plot_gram_heatmap(results: dict[int, BranchResult]):
    """Gram matrix heatmap sorted by n1."""
    L = 10  # good balance: 1024 branches, visible structure
    r = results[L]

    # Sort by n1, then by history for determinism
    order = np.lexsort([list(range(len(r.histories))), r.n1])
    G_sorted = np.abs(r.gram_normalized[np.ix_(order, order)])

    # Find n1 block boundaries for annotation
    n1_sorted = r.n1[order]
    boundaries = []
    for k in range(L + 1):
        idx = np.where(n1_sorted == k)[0]
        if len(idx) > 0:
            boundaries.append((k, idx[0], idx[-1]))

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(G_sorted, cmap="magma", vmin=0, vmax=1,
                   interpolation="none", aspect="equal")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label="|G(x,x')| (normalized)")

    # Mark n1 boundaries
    for k, start, end in boundaries:
        if start > 0:
            ax.axhline(start - 0.5, color="cyan", lw=0.5, alpha=0.5)
            ax.axvline(start - 0.5, color="cyan", lw=0.5, alpha=0.5)

    # Label a few n1 blocks
    for k, start, end in boundaries:
        mid = (start + end) / 2
        if end - start > 5:  # only label blocks large enough to see
            ax.text(-0.02 * len(order), mid, f"n₁={k}", fontsize=8,
                    ha="right", va="center", color="cyan",
                    transform=ax.transData)

    ax.set_title(
        f"Branch Gram matrix |G(x,x')| — L={L}, {len(r.histories)} branches\n"
        f"Sorted by n₁. Block-diagonal = decoherence between frequency classes.",
        fontsize=13, fontweight="bold")
    ax.set_xlabel("Branch index (sorted by n₁)")
    ax.set_ylabel("Branch index (sorted by n₁)")
    fig.tight_layout()
    fig.savefig(OUT / "branch_gram.png")
    fig.savefig(OUT / "branch_gram.pdf")
    plt.close(fig)
    print("  -> branch_gram.png")


if __name__ == "__main__":
    print("Branch-level analysis: watching worlds die\n")
    results = run_all()
    print("\nGenerating plots...")
    plot_death_of_worlds(results)
    plot_branch_census(results)
    plot_survivor_histogram(results)
    plot_gram_heatmap(results)
    print(f"\nAll plots saved to {OUT}/")
