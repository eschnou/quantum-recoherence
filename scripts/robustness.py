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


def _get_branch_result():
    """Shared branch simulation for threshold analyses."""
    L = 12
    DT = 1.2
    p = IsingDirectParams(m=M, L=L, hamming_threshold=THR,
                          trotter_steps=TROTTER, dt=DT, seed=SEED)
    r = simulate_branches(p)
    print(f"  {len(r.histories)} branches computed")
    return r


def threshold_sensitivity(r=None):
    """Vary epsilon threshold, show survivor count and frequency distribution."""
    L = 12
    DT = 1.2
    thresholds = np.arange(0.05, 0.65, 0.05)

    print(f"Threshold sensitivity: L={L}, dt={DT}")
    if r is None:
        r = _get_branch_result()

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


def threshold_detail(r=None):
    """Enhanced threshold figure: epsilon profiles colored by survival status at multiple thresholds."""
    L = 12
    DT = 1.2
    thresholds = np.arange(0.05, 0.70, 0.05)

    print(f"\nThreshold detail: L={L}, dt={DT}")
    if r is None:
        p = IsingDirectParams(m=M, L=L, hamming_threshold=THR,
                              trotter_steps=TROTTER, dt=DT, seed=SEED)
        r = simulate_branches(p)

    # 3-panel figure: survivor frequency distributions at 5 thresholds + gap vs threshold
    showcase = [0.10, 0.20, 0.30, 0.40, 0.50]
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Panel 1: Survivor count vs threshold (continuous)
    ax = axes[0]
    counts = [int(np.sum(r.epsilon < thr)) for thr in thresholds]
    ax.plot(thresholds, counts, "o-", color="royalblue", lw=2.5, ms=8)
    ax.axhline(D, color="black", ls="--", lw=1.5, alpha=0.5, label=f"D = {D}")
    for thr in showcase:
        n_alive = int(np.sum(r.epsilon < thr))
        ax.plot(thr, n_alive, "o", color="crimson", ms=12, zorder=10)
        ax.annotate(f"{n_alive}", (thr, n_alive), textcoords="offset points",
                    xytext=(8, 5), fontsize=10, color="crimson")
    ax.set_xlabel("Decoherence threshold εmax", fontsize=14)
    ax.set_ylabel("Number of decoherent branches", fontsize=14)
    ax.set_title("Survivor count vs threshold", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)

    # Panel 2: Frequency distributions at multiple thresholds
    ax = axes[1]
    colors_sc = plt.cm.plasma(np.linspace(0.15, 0.85, len(showcase)))
    for thr, color in zip(showcase, colors_sc):
        alive = r.epsilon < thr
        n1_alive = r.n1[alive]
        bins = np.arange(-0.5, L + 1.5, 1)
        ax.hist(n1_alive, bins=bins, alpha=0.35, color=color, edgecolor=color,
                label=f"ε < {thr:.2f} ({int(alive.sum())})")

    n1_range = np.arange(0, L + 1)
    born_pmf = binom.pmf(n1_range, L, r.p1)
    mid_count = int(np.sum(r.epsilon < 0.3))
    ax.plot(n1_range, born_pmf * mid_count, "o-", color="crimson", lw=2, ms=5,
            label=f"Born: Binom(L={L}, p₁={r.p1:.3f})")
    ax.set_xlabel("n₁ (number of '1' outcomes)", fontsize=14)
    ax.set_ylabel("Count", fontsize=14)
    ax.set_title("Survivor frequency distribution", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)

    # Panel 3: Mean epsilon of survivors vs threshold
    ax = axes[2]
    mean_eps_survivors = []
    frac_born = []
    for thr in thresholds:
        alive = r.epsilon < thr
        if alive.sum() > 0:
            mean_eps_survivors.append(np.mean(r.epsilon[alive]))
            # Fraction of survivors near Born frequency
            born_n1 = round(L * r.p1)
            near_born = alive & (np.abs(r.n1 - born_n1) <= 2)
            frac_born.append(near_born.sum() / alive.sum())
        else:
            mean_eps_survivors.append(np.nan)
            frac_born.append(np.nan)

    ax2 = ax.twinx()
    ax.plot(thresholds, mean_eps_survivors, "o-", color="royalblue", lw=2, ms=7,
            label="Mean ε of survivors")
    ax2.plot(thresholds, frac_born, "s--", color="firebrick", lw=2, ms=7,
             label="Fraction near Born freq")
    ax.set_xlabel("Decoherence threshold εmax", fontsize=14)
    ax.set_ylabel("Mean ε of survivors", fontsize=14, color="royalblue")
    ax2.set_ylabel("Fraction near Born (±2)", fontsize=14, color="firebrick")
    ax.set_title("Survivor quality vs threshold", fontsize=13, fontweight="bold")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=10)

    fig.suptitle(
        f"Threshold sensitivity — {M} qubits (D={D}), L={L}\n"
        f"Born filtering is robust across thresholds: survivors cluster at Born frequency regardless of cutoff",
        fontsize=14, fontweight="bold", y=1.04)
    fig.tight_layout()
    fig.savefig(OUT / "threshold_detail.png")
    fig.savefig(OUT / "threshold_detail.pdf")
    plt.close(fig)
    print("  -> threshold_detail.png")


if __name__ == "__main__":
    print("Robustness checks\n")
    branch_result = _get_branch_result()
    threshold_sensitivity(branch_result)
    dt_sensitivity()
    threshold_detail(branch_result)
    print(f"\nAll plots saved to {OUT}/")
