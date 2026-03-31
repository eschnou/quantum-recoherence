#!/usr/bin/env python3
"""Robustness checks: threshold sensitivity and dt variation."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.stats import binom

from recohere.branches import simulate_branches
from recohere.ising_direct import IsingDirectParams, simulate_and_analyze

OUT = Path("results")
OUT.mkdir(exist_ok=True)
plt.rcParams.update({
    "font.size": 13, "axes.grid": True, "grid.alpha": 0.25,
    "savefig.dpi": 200, "savefig.bbox": "tight", "font.family": "serif",
})

M, THR = 9, 4
D = 2**M
SEED = 42
TROTTER = 1


def threshold_sensitivity():
    """Vary epsilon threshold, show survivor count and frequency distribution."""
    L = 12
    DT = 1.2
    thresholds = np.arange(0.05, 0.65, 0.05)

    print(f"Threshold sensitivity: L={L}, dt={DT}")
    p = IsingDirectParams(m=M, L=L, hamming_threshold=THR,
                          trotter_steps=TROTTER, dt=DT, seed=SEED)
    r = simulate_branches(p)
    print(f"  {len(r.histories)} branches computed")

    counts = []
    for thr in thresholds:
        n_alive = int(np.sum(r.epsilon < thr))
        counts.append(n_alive)
        print(f"  eps < {thr:.2f}: {n_alive} decoherent")

    # Plot: left = count vs threshold, right = frequency distributions at 3 thresholds
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

    ax = axes[0]
    ax.plot(thresholds, counts, "o-", color="royalblue", lw=2.5, ms=8)
    ax.axhline(D, color="black", ls="--", lw=1.5, alpha=0.5, label=f"D = {D}")
    ax.set_xlabel("Decoherence threshold εmax", fontsize=14)
    ax.set_ylabel("Number of decoherent branches", fontsize=14)
    ax.set_title(f"Survivor count vs threshold (L={L})", fontsize=14, fontweight="bold")
    ax.legend(fontsize=12)

    ax = axes[1]
    showcase = [0.1, 0.3, 0.5]
    colors = ["navy", "royalblue", "cornflowerblue"]
    for thr, color in zip(showcase, colors):
        alive = r.epsilon < thr
        n1_alive = r.n1[alive]
        bins = np.arange(-0.5, L + 1.5, 1)
        ax.hist(n1_alive, bins=bins, alpha=0.4, color=color, edgecolor=color,
                label=f"ε < {thr} ({int(alive.sum())} branches)")

    # Born overlay
    n1_range = np.arange(0, L + 1)
    born_pmf = binom.pmf(n1_range, L, r.p1)
    # Scale to middle threshold count
    mid_count = int(np.sum(r.epsilon < 0.3))
    ax.plot(n1_range, born_pmf * mid_count, "o-", color="crimson", lw=2, ms=5,
            label=f"Born: Binom(L={L}, p₁={r.p1:.3f})")
    ax.axvline(r.p1 * L, color="crimson", ls="--", lw=1.5, alpha=0.5)

    ax.set_xlabel("n₁ (number of '1' outcomes)", fontsize=14)
    ax.set_ylabel("Count", fontsize=14)
    ax.set_title("Frequency distribution at three thresholds", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)

    fig.suptitle(
        f"Threshold robustness — {M} qubits (D={D}), L={L}",
        fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "robustness_threshold.png")
    fig.savefig(OUT / "robustness_threshold.pdf")
    plt.close(fig)
    print("  -> robustness_threshold.png")


def dt_sensitivity():
    """Vary dt, show epsilon profile at L=15."""
    L = 15
    dt_values = [0.4, 0.6, 0.8, 1.0, 1.2]
    N_SEEDS = 5

    print(f"\nDt sensitivity: L={L}, {N_SEEDS} seeds each")

    fig, ax = plt.subplots(figsize=(12, 6.5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(dt_values)))

    p1 = IsingDirectParams(m=M, L=1, hamming_threshold=THR).p1

    for dt, color in zip(dt_values, colors):
        eps_all = []
        for seed in range(N_SEEDS):
            p = IsingDirectParams(m=M, L=L, hamming_threshold=THR,
                                  trotter_steps=TROTTER, dt=dt, seed=seed)
            r = simulate_and_analyze(p)
            eps_all.append(r.epsilon)

        eps_arr = np.array(eps_all)
        eps_mean = np.nanmean(eps_arr, axis=0)
        eps_std = np.nanstd(eps_arr, axis=0)
        freq = np.arange(L + 1) / L

        born_n1 = round(L * p1)
        comb_n1 = L // 2
        gap = eps_mean[comb_n1] / eps_mean[born_n1] if eps_mean[born_n1] > 0.001 else 0

        ax.plot(freq, eps_mean, "o-", color=color, lw=2, ms=6,
                label=f"Δt = {dt} (gap = {gap:.1f}×)")
        ax.fill_between(freq, eps_mean - eps_std, eps_mean + eps_std,
                        color=color, alpha=0.1)
        print(f"  dt={dt}: gap = {gap:.1f}×")

    ax.axvline(p1, color="crimson", ls="--", lw=2, alpha=0.6, label=f"Born p₁={p1:.2f}")
    ax.axvline(0.5, color="seagreen", ls=":", lw=2, alpha=0.6, label="Combinatorial peak")
    ax.set_xlabel("Frequency n₁/L", fontsize=14)
    ax.set_ylabel("ε (recoherence parameter)", fontsize=14)
    ax.set_title(
        f"Time-step robustness — Ising circuit ({M} qubits, D={D}, L={L})\n"
        f"Averaged over {N_SEEDS} seeds per Δt",
        fontsize=14, fontweight="bold")
    ax.set_xlim(-0.03, 1.03)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "robustness_dt.png")
    fig.savefig(OUT / "robustness_dt.pdf")
    plt.close(fig)
    print("  -> robustness_dt.png")


if __name__ == "__main__":
    print("Robustness checks\n")
    threshold_sensitivity()
    dt_sensitivity()
    print(f"\nAll plots saved to {OUT}/")
