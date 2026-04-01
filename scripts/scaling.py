#!/usr/bin/env python3
"""Scaling study: Born-filtering gap vs system size (D).

Runs the frequency-class simulation at m=7,8,9,10 qubits to show
how the Born/combinatorial gap scales with Hilbert space dimension.
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

N_SEEDS = 10
DT = 1.2

# System sizes to test. For each m, pick hamming_threshold that gives
# p1 closest to 0.746 (the 9-qubit value), and L large enough to
# oversubscribe the Hilbert space.
CONFIGS = [
    # (m, hamming_threshold, L_values)
    (7,  3, [2, 4, 6, 8, 10, 12]),       # D=128,  p1=0.773
    (8,  4, [2, 4, 6, 8, 10, 12, 14]),   # D=256,  p1=0.637
    (8,  3, [2, 4, 6, 8, 10, 12, 14]),   # D=256,  p1=0.855
    (9,  4, [2, 4, 6, 8, 10, 12, 15]),   # D=512,  p1=0.746
    (10, 5, [2, 4, 6, 8, 10, 12, 15]),   # D=1024, p1=0.623
    (10, 4, [2, 4, 6, 8, 10, 12, 15]),   # D=1024, p1=0.828
]


def run_scaling():
    results = {}

    for m, thr, L_values in CONFIGS:
        p1 = IsingDirectParams(m=m, L=1, hamming_threshold=thr).p1
        D = 2**m
        print(f"\nm={m} (D={D}), threshold={thr}, p1={p1:.3f}")

        gaps = []
        max_L_eps = None

        for Lv in L_values:
            eps_born_list, eps_comb_list = [], []
            eps_all = []

            for seed in range(N_SEEDS):
                p = IsingDirectParams(m=m, L=Lv, hamming_threshold=thr,
                                      dt=DT, seed=seed)
                r = simulate_and_analyze(p)
                born_n1 = min(round(Lv * p1), Lv)
                comb_n1 = Lv // 2

                eps_born_list.append(r.epsilon[born_n1])
                eps_comb_list.append(r.epsilon[comb_n1])
                eps_all.append(r.epsilon)

            eb = np.nanmean(eps_born_list)
            ec = np.nanmean(eps_comb_list)
            gap = ec / eb if eb > 0.001 else 0
            gaps.append((Lv, eb, ec, gap))
            max_L_eps = np.array(eps_all)  # only keep the last (max L)
            print(f"  L={Lv:2d}: eps_born={eb:.3f}, eps_comb={ec:.3f}, gap={gap:.1f}x")

        results[(m, thr)] = {
            "D": D, "p1": p1, "gaps": gaps, "L_values": L_values,
            "max_L_eps": max_L_eps,
        }

    return results


def plot_scaling(results):
    # 1. Gap vs L for each system size
    fig, ax = plt.subplots(figsize=(12, 6.5))
    markers = ["o", "s", "D", "^", "v", "P"]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(results)))

    for i, ((m, thr), data) in enumerate(sorted(results.items())):
        L_vals = [g[0] for g in data["gaps"]]
        gap_vals = [g[3] for g in data["gaps"]]
        ax.plot(L_vals, gap_vals, f"{markers[i % len(markers)]}-",
                color=colors[i], lw=2, ms=8,
                label=f"m={m}, D={data['D']}, p₁={data['p1']:.3f} (thr={thr})")

    ax.axhline(1.0, color="gray", ls="--", lw=1.5, alpha=0.5, label="No filtering")
    ax.set_xlabel("History length L", fontsize=14)
    ax.set_ylabel("Gap: ε(comb) / ε(Born)", fontsize=14)
    ax.set_title("Born-filtering gap vs system size", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, None)
    fig.tight_layout()
    fig.savefig(OUT / "scaling_gap.png")
    fig.savefig(OUT / "scaling_gap.pdf")
    plt.close(fig)
    print("\n  -> scaling_gap.png")

    # 2. Epsilon profiles at max-L for each system size
    n_configs = len(results)
    cols = min(3, n_configs)
    rows = (n_configs + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5.5 * rows), sharey=True)
    if n_configs == 1:
        axes = np.array([axes])
    axes_flat = np.atleast_1d(axes.flatten())

    for i, ((m, thr), data) in enumerate(sorted(results.items())):
        ax = axes_flat[i]
        max_L = max(data["L_values"])
        eps_arr = data["max_L_eps"]
        freq = np.arange(max_L + 1) / max_L

        eps_mean = np.nanmean(eps_arr, axis=0)
        eps_std = np.nanstd(eps_arr, axis=0)

        ax.plot(freq, eps_mean, "s-", color="firebrick", lw=2, ms=5)
        ax.fill_between(freq, eps_mean - eps_std, eps_mean + eps_std,
                         color="firebrick", alpha=0.15)
        ax.axvline(data["p1"], color="crimson", ls="--", lw=2, alpha=0.6)
        ax.axvline(0.5, color="seagreen", ls=":", lw=2, alpha=0.6)
        ax.set_title(f"m={m}, D={data['D']}, p₁={data['p1']:.2f}, L={max_L}",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("n₁/L", fontsize=12)
        if i % cols == 0:
            ax.set_ylabel("ε", fontsize=14)
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.02, 1.05)

    # Hide unused subplots
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        f"Epsilon profiles at max L — scaling across system sizes\n"
        f"({N_SEEDS} seeds, Δt={DT})",
        fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "scaling_profiles.png")
    fig.savefig(OUT / "scaling_profiles.pdf")
    plt.close(fig)
    print("  -> scaling_profiles.png")

    # 3. Gap vs p1 asymmetry, with D as color (the key scaling plot)
    fig, axes = plt.subplots(1, 2, figsize=(18, 6.5))

    # Left: gap vs |p1 - 0.5| colored by D
    ax = axes[0]
    D_unique = sorted(set(data["D"] for data in results.values()))
    cmap = plt.cm.viridis
    d_colors = {d: cmap(i / max(1, len(D_unique) - 1)) for i, d in enumerate(D_unique)}

    for (m, thr), data in sorted(results.items()):
        asym = abs(data["p1"] - 0.5)
        max_gap = data["gaps"][-1][3]
        color = d_colors[data["D"]]
        ax.scatter(asym, max_gap, s=200, color=color, edgecolors="black",
                   linewidths=1.5, zorder=5)
        ax.annotate(f"m={m}, thr={thr}\np₁={data['p1']:.2f}",
                    (asym, max_gap), textcoords="offset points",
                    xytext=(12, -5), fontsize=9, color="dimgray")

    # Legend for D values
    for d in D_unique:
        ax.scatter([], [], s=150, color=d_colors[d], edgecolors="black",
                   linewidths=1.5, label=f"D={d}")
    ax.axhline(1.0, color="gray", ls="--", lw=1.5, alpha=0.5, label="No filtering")
    ax.set_xlabel("|p₁ - 0.5| (projector asymmetry)", fontsize=14)
    ax.set_ylabel("Gap: ε(comb) / ε(Born) at max L", fontsize=14)
    ax.set_title("Gap depends on p₁ asymmetry", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    ax.set_xlim(-0.02, 0.40)
    ax.set_ylim(0, None)

    # Right: gap vs L for matched-p1 pairs (high-p1 group only)
    ax = axes[1]
    for i, ((m, thr), data) in enumerate(sorted(results.items())):
        if data["p1"] < 0.6:  # skip near-symmetric cases
            continue
        L_vals = [g[0] for g in data["gaps"]]
        gap_vals = [g[3] for g in data["gaps"]]
        color = d_colors[data["D"]]
        ax.plot(L_vals, gap_vals, f"{markers[i % len(markers)]}-",
                color=color, lw=2, ms=8,
                label=f"D={data['D']}, p₁={data['p1']:.2f}")

    ax.axhline(1.0, color="gray", ls="--", lw=1.5, alpha=0.5, label="No filtering")
    ax.set_xlabel("History length L", fontsize=14)
    ax.set_ylabel("Gap: ε(comb) / ε(Born)", fontsize=14)
    ax.set_title("Gap growth with L (asymmetric projectors only, p₁ > 0.6)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, None)

    fig.suptitle(
        "Born-filtering gap: dependence on system size and projector asymmetry",
        fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "scaling_vs_D.png")
    fig.savefig(OUT / "scaling_vs_D.pdf")
    plt.close(fig)
    print("  -> scaling_vs_D.png")


if __name__ == "__main__":
    print("Scaling study: Born-filtering gap vs system size\n")
    results = run_scaling()
    print("\nGenerating plots...")
    plot_scaling(results)
    print(f"\nAll plots saved to {OUT}/")
