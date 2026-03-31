#!/usr/bin/env python3
"""Hardening: Ising direct (no recorder gate) vs Strasberg, multi-seed."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from recohere.ising_direct import IsingDirectParams, simulate_and_analyze
from recohere.strasberg import StrasbergParams, run_strasberg

OUT = Path("results")
plt.rcParams.update({
    "font.size": 13, "axes.grid": True, "grid.alpha": 0.25,
    "savefig.dpi": 200, "savefig.bbox": "tight", "font.family": "serif",
})

M, THR, L = 9, 4, 15  # p1=0.746, D=512
DT = 1.2
N_SEEDS = 10


def run_all():
    p1 = IsingDirectParams(m=M, L=1, hamming_threshold=THR).p1
    D = 2**M

    # --- Ising direct ---
    print(f"Ising direct: m={M} ({D} dims), p1={p1:.3f}, L={L}, {N_SEEDS} seeds")
    ising_eps = []
    for seed in range(N_SEEDS):
        p = IsingDirectParams(m=M, L=L, hamming_threshold=THR, dt=DT, seed=seed)
        r = simulate_and_analyze(p)
        ising_eps.append(r.epsilon)
        born = round(r.born_frequency)
        mn = int(np.nanargmin(r.epsilon))
        print(f"  seed={seed}: min@{mn}({mn/L:.2f}) eps_b={r.epsilon[born]:.3f} eps_c={r.epsilon[L//2]:.3f}")

    # --- Strasberg at matching D ---
    print(f"\nStrasberg: D={D}, p1={p1:.3f}, L={L}, {N_SEEDS} seeds")
    stras_eps = []
    for seed in range(N_SEEDS):
        sp = StrasbergParams(D=D, d1_ratio=p1, L=L)
        r = run_strasberg(sp, seed=seed)
        stras_eps.append(r.epsilon)
        born = round(r.born_frequency)
        mn = int(np.nanargmin(r.epsilon))
        print(f"  seed={seed}: min@{mn}({mn/L:.2f}) eps_b={r.epsilon[born]:.3f} eps_c={r.epsilon[L//2]:.3f}")

    ising_eps = np.array(ising_eps)
    stras_eps = np.array(stras_eps)
    return ising_eps, stras_eps, p1


def plot_overlay(ising_eps, stras_eps, p1):
    freq = np.arange(L + 1) / L
    D = 2**M

    ie_mean, ie_std = np.nanmean(ising_eps, axis=0), np.nanstd(ising_eps, axis=0)
    se_mean, se_std = np.nanmean(stras_eps, axis=0), np.nanstd(stras_eps, axis=0)

    fig, ax = plt.subplots(figsize=(12, 6.5))

    ax.plot(freq, se_mean, "o-", color="royalblue", lw=2, ms=7,
            label=f"Strasberg random matrix (D={D}, avg {N_SEEDS} seeds)", zorder=5)
    ax.fill_between(freq, se_mean - se_std, se_mean + se_std, color="royalblue", alpha=0.12)

    ax.plot(freq, ie_mean, "s-", color="firebrick", lw=2, ms=7,
            label=f"Ising circuit ({M} qubits, D={D}, avg {N_SEEDS} seeds)", zorder=5)
    ax.fill_between(freq, ie_mean - ie_std, ie_mean + ie_std, color="firebrick", alpha=0.12)

    ax.axvline(p1, color="crimson", ls="--", lw=2, alpha=0.6, label=f"Born p₁={p1:.2f}")
    ax.axvline(0.5, color="seagreen", ls=":", lw=2, alpha=0.6, label="Combinatorial peak")

    ax.set_xlabel("Frequency n₁/L", fontsize=14)
    ax.set_ylabel("ε (recoherence parameter)", fontsize=14)
    ax.set_title(
        f"Born-rule filtering: Strasberg vs Ising circuit (D={D}, L={L}, p₁={p1:.2f})\n"
        f"Shaded = ±1σ over {N_SEEDS} initial states",
        fontsize=14, fontweight="bold")
    ax.set_xlim(-0.03, 1.03)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "overlay.png")
    fig.savefig(OUT / "overlay.pdf")
    plt.close(fig)
    print("  -> overlay.png")


def plot_robustness(ising_eps, p1):
    freq = np.arange(L + 1) / L
    born = round(L * p1)

    fig, ax = plt.subplots(figsize=(12, 6.5))
    for i in range(ising_eps.shape[0]):
        ax.plot(freq, ising_eps[i], "o-", alpha=0.25, color="firebrick", ms=4)

    mean = np.nanmean(ising_eps, axis=0)
    ax.plot(freq, mean, "s-", color="black", lw=2.5, ms=8, label=f"Mean ({N_SEEDS} seeds)", zorder=10)

    ax.axvline(p1, color="crimson", ls="--", lw=2, alpha=0.6, label=f"Born p₁={p1:.2f}")
    ax.axvline(0.5, color="seagreen", ls=":", lw=2, alpha=0.6, label="Combinatorial peak")

    ax.set_xlabel("Frequency n₁/L", fontsize=14)
    ax.set_ylabel("ε (recoherence parameter)", fontsize=14)
    ax.set_title(
        f"Robustness: {N_SEEDS} different initial states\n"
        f"Ising circuit, {M} qubits, D={2**M}, L={L}, p₁={p1:.2f}",
        fontsize=14, fontweight="bold")
    ax.set_xlim(-0.03, 1.03)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "robustness.png")
    fig.savefig(OUT / "robustness.pdf")
    plt.close(fig)
    print("  -> robustness.png")


def plot_emergence(p1):
    """Epsilon profile as L grows."""
    L_values = [2, 4, 6, 8, 10, 12, 15]
    eps_born_list, eps_comb_list = [], []

    fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharey=True)
    axes_flat = axes.flatten()

    for i, Lv in enumerate(L_values):
        ax = axes_flat[i]
        # Average 5 seeds per L
        eps_avg = np.zeros(Lv + 1)
        for seed in range(5):
            p = IsingDirectParams(m=M, L=Lv, hamming_threshold=THR,
                                  dt=DT, seed=seed)
            r = simulate_and_analyze(p)
            eps_avg += np.nan_to_num(r.epsilon)
        eps_avg /= 5

        freq = np.arange(Lv + 1) / Lv
        ax.bar(freq, eps_avg, width=0.7/Lv, color="steelblue", alpha=0.75, edgecolor="steelblue")
        ax.axvline(p1, color="crimson", ls="--", lw=2, alpha=0.7)
        ax.axvline(0.5, color="seagreen", ls=":", lw=2, alpha=0.7)
        ax.set_title(f"L = {Lv}", fontsize=15, fontweight="bold")
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlim(-0.05, 1.05)
        ax.set_xlabel("n₁/L", fontsize=12)
        if i % 4 == 0:
            ax.set_ylabel("ε(n₁)", fontsize=14)

        born_n1 = min(round(Lv * p1), Lv)
        comb_n1 = Lv // 2
        eps_born_list.append(eps_avg[born_n1])
        eps_comb_list.append(eps_avg[comb_n1])

    axes_flat[7].set_visible(False)
    fig.suptitle(
        f"Emergence of Born-rule filtering — Ising circuit ({M} qubits, p₁={p1:.2f})\n"
        f"Red = Born, green = combinatorial. Averaged over 5 seeds.",
        fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "emergence.png")
    fig.savefig(OUT / "emergence.pdf")
    plt.close(fig)
    print("  -> emergence.png")

    # Tracking plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(L_values, eps_born_list, "o-", color="crimson", lw=2.5, ms=10,
            label=f"ε at Born frequency (n₁/L ≈ {p1:.2f})", zorder=5)
    ax.plot(L_values, eps_comb_list, "s-", color="seagreen", lw=2.5, ms=10,
            label="ε at combinatorial peak (n₁/L = 0.5)", zorder=5)
    ax.set_xlabel("History length L", fontsize=14)
    ax.set_ylabel("ε (recoherence parameter)", fontsize=14)
    ax.set_title(f"Born-rule emergence — Ising circuit ({M} qubits, p₁={p1:.2f})",
                 fontsize=14, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "tracking.png")
    fig.savefig(OUT / "tracking.pdf")
    plt.close(fig)
    print("  -> tracking.png")


def print_summary(ising_eps, stras_eps, p1):
    born = round(L * p1)
    comb = L // 2
    ie, se = np.nanmean(ising_eps, axis=0), np.nanmean(stras_eps, axis=0)
    ie_s, se_s = np.nanstd(ising_eps, axis=0), np.nanstd(stras_eps, axis=0)

    print(f"\n{'='*60}")
    print(f"SUMMARY ({N_SEEDS} seeds, D={2**M}, L={L}, p₁={p1:.3f})")
    print(f"{'='*60}")
    print(f"{'':>22} {'Ising circuit':>18} {'Strasberg':>18}")
    print(f"{'ε at Born (n₁='+str(born)+')':>22} {ie[born]:>7.3f} ± {ie_s[born]:.3f} {se[born]:>7.3f} ± {se_s[born]:.3f}")
    print(f"{'ε at comb (n₁='+str(comb)+')':>22} {ie[comb]:>7.3f} ± {ie_s[comb]:.3f} {se[comb]:>7.3f} ± {se_s[comb]:.3f}")
    r_i = ie[comb] / ie[born] if ie[born] > 0.001 else 0
    r_s = se[comb] / se[born] if se[born] > 0.001 else 0
    print(f"{'Ratio comb/born':>22} {r_i:>7.1f}x           {r_s:>7.1f}x")
    print(f"{'='*60}")


if __name__ == "__main__":
    print("Hardening v2: Ising direct (no recorder gate)\n")
    ising_eps, stras_eps, p1 = run_all()
    print("\nGenerating plots...")
    plot_overlay(ising_eps, stras_eps, p1)
    plot_robustness(ising_eps, p1)
    plot_emergence(p1)
    print_summary(ising_eps, stras_eps, p1)
    print(f"\nAll plots saved to {OUT}/")
