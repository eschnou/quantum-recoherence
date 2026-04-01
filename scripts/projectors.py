#!/usr/bin/env python3
"""Projector study: Born filtering across different coarse-graining choices.

Tests Hamming >=3, >=4, >=5 and spatial-majority projectors to show
that Born filtering is not an artifact of the specific projector choice.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from recohere.ising_direct import IsingDirectParams, simulate_and_analyze

OUT = Path("results")
OUT.mkdir(exist_ok=True)
plt.rcParams.update({
    "font.size": 13, "axes.grid": True, "grid.alpha": 0.25,
    "savefig.dpi": 200, "savefig.bbox": "tight", "font.family": "serif",
})

M = 9
D = 2**M
L = 15
DT = 1.2
N_SEEDS = 10

# Projector configurations: (name, projector_type, hamming_threshold)
PROJECTORS = [
    ("Hamming ≥ 3", "hamming", 3),
    ("Hamming ≥ 4", "hamming", 4),
    ("Left-block ≥ 2/5", "left_heavy", 2),   # first 5 qubits, HW >= 2 → p1=0.8125
    ("Hamming ≥ 5", "hamming", 5),            # p1=0.5, control case
]


def run_projectors():
    results = {}

    for name, proj, thr in PROJECTORS:
        p = IsingDirectParams(m=M, L=L, hamming_threshold=thr, dt=DT, seed=0,
                              projector=proj)
        p1 = p.p1
        print(f"\n{name} (projector={proj}, thr={thr}): p1={p1:.3f}")

        eps_all = []
        for seed in range(N_SEEDS):
            p = IsingDirectParams(m=M, L=L, hamming_threshold=thr, dt=DT,
                                  seed=seed, projector=proj)
            r = simulate_and_analyze(p)
            eps_all.append(r.epsilon)

            born_n1 = min(round(L * p1), L)
            comb_n1 = L // 2
            if seed == 0:
                print(f"  seed={seed}: eps_born={r.epsilon[born_n1]:.3f}, "
                      f"eps_comb={r.epsilon[comb_n1]:.3f}")

        eps_arr = np.array(eps_all)
        eps_mean = np.nanmean(eps_arr, axis=0)
        born_n1 = min(round(L * p1), L)
        comb_n1 = L // 2
        gap = eps_mean[comb_n1] / eps_mean[born_n1] if eps_mean[born_n1] > 0.001 else 0

        results[name] = {
            "projector": proj, "threshold": thr, "p1": p1,
            "eps_mean": eps_mean, "eps_std": np.nanstd(eps_arr, axis=0),
            "gap": gap,
        }
        print(f"  Mean gap: {gap:.1f}x")

    return results


def plot_projectors(results):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5.5), sharey=True)
    if n == 1:
        axes = [axes]

    freq = np.arange(L + 1) / L
    colors = ["steelblue", "firebrick", "seagreen", "darkorange"]

    for i, (name, data) in enumerate(results.items()):
        ax = axes[i]
        ax.plot(freq, data["eps_mean"], "s-", color=colors[i % len(colors)],
                lw=2, ms=5)
        ax.fill_between(freq,
                         data["eps_mean"] - data["eps_std"],
                         data["eps_mean"] + data["eps_std"],
                         color=colors[i % len(colors)], alpha=0.15)
        ax.axvline(data["p1"], color="crimson", ls="--", lw=2, alpha=0.6)
        ax.axvline(0.5, color="seagreen", ls=":", lw=2, alpha=0.6)
        ax.set_title(f"{name}\np₁={data['p1']:.3f}, gap={data['gap']:.1f}×",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("n₁/L", fontsize=12)
        if i == 0:
            ax.set_ylabel("ε (recoherence parameter)", fontsize=14)
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.02, 1.05)

    fig.suptitle(
        f"Born filtering across projectors — {M} qubits (D={D}), L={L}\n"
        f"({N_SEEDS} seeds, Δt={DT}). Red dashed = Born p₁, green dotted = combinatorial peak.",
        fontsize=14, fontweight="bold", y=1.04)
    fig.tight_layout()
    fig.savefig(OUT / "projectors.png")
    fig.savefig(OUT / "projectors.pdf")
    plt.close(fig)
    print("\n  -> projectors.png")

    # Summary table
    print(f"\n{'='*65}")
    print(f"{'Projector':>20} {'p₁':>8} {'ε(Born)':>10} {'ε(comb)':>10} {'Gap':>8}")
    print(f"{'='*65}")
    for name, data in results.items():
        born_n1 = min(round(L * data["p1"]), L)
        comb_n1 = L // 2
        print(f"{name:>20} {data['p1']:>8.3f} "
              f"{data['eps_mean'][born_n1]:>10.3f} "
              f"{data['eps_mean'][comb_n1]:>10.3f} "
              f"{data['gap']:>7.1f}×")
    print(f"{'='*65}")


if __name__ == "__main__":
    print("Projector study: Born filtering across coarse-graining choices\n")
    results = run_projectors()
    print("\nGenerating plots...")
    plot_projectors(results)
    print(f"\nAll plots saved to {OUT}/")
