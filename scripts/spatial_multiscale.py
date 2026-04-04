#!/usr/bin/env python3
"""Spatial multi-scale Born filtering.

8 qubits split into two groups of 4. At each step, independent Hamming-weight
projectors on each group give a 4-outcome measurement (00, 01, 10, 11).

Three nested scales of proper orthogonal projectors:
  Scale 1a/1b: each group independently (p1 = 5/16 = 0.3125)
  Scale 2: both groups (p1 = 25/256 = 0.0977)

Born filtering operates at all levels simultaneously.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from recohere.spatial_multiscale import SpatialParams, simulate_spatial_multiscale

OUT = Path("results")
OUT.mkdir(exist_ok=True)
plt.rcParams.update({
    "font.size": 13, "axes.grid": True, "grid.alpha": 0.25,
    "savefig.dpi": 200, "savefig.bbox": "tight", "font.family": "serif",
})

THR = 2
DT = 1.2
N_SEEDS = 10
L = 20


def run_seeds(L: int, n_seeds: int) -> list:
    results = []
    for seed in range(n_seeds):
        p = SpatialParams(L=L, hamming_threshold=THR, dt=DT, seed=seed)
        results.append(simulate_spatial_multiscale(p))
    return results


def plot_overlay():
    """Epsilon at all three scales on the same axes, averaged over seeds."""
    results = run_seeds(L, N_SEEDS)
    p0 = results[0].params
    freq = np.arange(L + 1) / L

    # Collect epsilon arrays
    eps_1a = np.array([r.scale1_left.epsilon for r in results])
    eps_1b = np.array([r.scale1_right.epsilon for r in results])
    eps_s2 = np.array([r.scale2_either.epsilon for r in results])

    fig, ax = plt.subplots(figsize=(12, 6.5))

    for eps_arr, color, ms, label, born_p in [
        (eps_1a, "firebrick", "s", f"Scale 1a (left, p$_1$={p0.p1_left:.3f})", p0.p1_left),
        (eps_1b, "darkorange", "^", f"Scale 1b (right, p$_1$={p0.p1_right:.3f})", p0.p1_right),
        (eps_s2, "royalblue", "o", f"Scale 2 (either, p$_1$={p0.p1_either:.3f})", p0.p1_either),
    ]:
        mean = np.nanmean(eps_arr, axis=0)
        std = np.nanstd(eps_arr, axis=0)
        ax.plot(freq, mean, f"{ms}-", color=color, lw=2, ms=7, label=label, zorder=5)
        ax.fill_between(freq, mean - std, mean + std, color=color, alpha=0.12)
        ax.axvline(born_p, color=color, ls="--", lw=1.5, alpha=0.5)

    ax.axvline(0.5, color="seagreen", ls=":", lw=2, alpha=0.6, label="Combinatorial peak")
    ax.set_xlabel("Frequency n/L", fontsize=14)
    ax.set_ylabel(r"$\varepsilon$ (recoherence parameter)", fontsize=14)
    ax.set_title(
        f"Spatial multi-scale Born filtering — 8 qubits (D={p0.D}), L={L}\n"
        f"Three projector scales on the same Hilbert space. "
        f"Shaded = $\\pm 1\\sigma$ over {N_SEEDS} seeds.",
        fontsize=14, fontweight="bold")
    ax.set_xlim(-0.03, 1.03)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "spatial_overlay.png")
    fig.savefig(OUT / "spatial_overlay.pdf")
    plt.close(fig)
    print("  -> spatial_overlay.png")


def plot_emergence():
    """Epsilon profile panels as L grows, for scale 1a and scale 2."""
    L_values = [4, 6, 8, 10, 12, 15, 20]
    n_avg = 5

    fig, axes = plt.subplots(2, len(L_values), figsize=(4 * len(L_values), 9),
                              sharey=True)

    for col, Lv in enumerate(L_values):
        # Average over seeds
        eps_1a_avg = np.zeros(Lv + 1)
        eps_s2_avg = np.zeros(Lv + 1)
        for seed in range(n_avg):
            p = SpatialParams(L=Lv, hamming_threshold=THR, dt=DT, seed=seed)
            r = simulate_spatial_multiscale(p)
            eps_1a_avg += np.nan_to_num(r.scale1_left.epsilon)
            eps_s2_avg += np.nan_to_num(r.scale2_either.epsilon)
        eps_1a_avg /= n_avg
        eps_s2_avg /= n_avg

        freq = np.arange(Lv + 1) / Lv
        p0 = SpatialParams(L=Lv, hamming_threshold=THR, dt=DT, seed=0)

        # Top row: scale 1a
        ax = axes[0, col]
        ax.bar(freq, eps_1a_avg, width=0.7/Lv, color="firebrick", alpha=0.75)
        ax.axvline(p0.p1_left, color="crimson", ls="--", lw=2, alpha=0.7)
        ax.axvline(0.5, color="seagreen", ls=":", lw=2, alpha=0.7)
        ax.set_title(f"L = {Lv}", fontsize=13, fontweight="bold")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel("n/L", fontsize=11)
        if col == 0:
            ax.set_ylabel(r"Scale 1a: $\varepsilon$", fontsize=13)

        # Bottom row: scale 2
        ax = axes[1, col]
        ax.bar(freq, eps_s2_avg, width=0.7/Lv, color="royalblue", alpha=0.75)
        ax.axvline(p0.p1_either, color="navy", ls="--", lw=2, alpha=0.7)
        ax.axvline(0.5, color="seagreen", ls=":", lw=2, alpha=0.7)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel("n/L", fontsize=11)
        if col == 0:
            ax.set_ylabel(r"Scale 2: $\varepsilon$", fontsize=13)

    fig.suptitle(
        f"Emergence of Born filtering at two spatial scales — 8 qubits (D=256)\n"
        f"Top: scale 1a (left group, p$_1$={p0.p1_left:.3f}). "
        f"Bottom: scale 2 (either group, p$_1$={p0.p1_either:.3f}). "
        f"Averaged over {n_avg} seeds.",
        fontsize=14, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(OUT / "spatial_emergence.png")
    fig.savefig(OUT / "spatial_emergence.pdf")
    plt.close(fig)
    print("  -> spatial_emergence.png")


def plot_gap_growth():
    """Born-combinatorial gap as a function of L, for both scales."""
    L_values = [4, 6, 8, 10, 12, 15, 18, 20, 25]
    n_avg = 5

    gaps_1a, gaps_s2 = [], []
    eps_born_1a, eps_born_s2 = [], []

    for Lv in L_values:
        g1a_seeds, gs2_seeds = [], []
        eb1a_seeds, ebs2_seeds = [], []
        for seed in range(n_avg):
            p = SpatialParams(L=Lv, hamming_threshold=THR, dt=DT, seed=seed)
            r = simulate_spatial_multiscale(p)

            for cr, g_list, eb_list in [
                (r.scale1_left, g1a_seeds, eb1a_seeds),
                (r.scale2_either, gs2_seeds, ebs2_seeds),
            ]:
                born_n = round(cr.born_frequency)
                comb_n = Lv // 2
                v = ~np.isnan(cr.epsilon)
                eb = cr.epsilon[born_n] if v[born_n] else np.nan
                ec = cr.epsilon[comb_n] if v[comb_n] else np.nan
                g_list.append(ec / eb if eb > 0.001 else np.nan)
                eb_list.append(eb)

        gaps_1a.append(np.nanmean(g1a_seeds))
        gaps_s2.append(np.nanmean(gs2_seeds))
        eps_born_1a.append(np.nanmean(eb1a_seeds))
        eps_born_s2.append(np.nanmean(ebs2_seeds))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

    ax = axes[0]
    ax.plot(L_values, gaps_1a, "s-", color="firebrick", lw=2.5, ms=10,
            label=f"Scale 1a (p$_1$=0.312)")
    ax.plot(L_values, gaps_s2, "o-", color="royalblue", lw=2.5, ms=10,
            label=f"Scale 2 (p$_1$=0.902)")
    ax.axhline(1, color="gray", ls="--", lw=1, alpha=0.5)
    ax.set_xlabel("History length L", fontsize=14)
    ax.set_ylabel(r"Gap: $\varepsilon_\mathrm{comb} / \varepsilon_\mathrm{Born}$", fontsize=14)
    ax.set_title("Born-combinatorial gap vs L", fontsize=14, fontweight="bold")
    ax.legend(fontsize=12)

    ax = axes[1]
    ax.plot(L_values, eps_born_1a, "s-", color="firebrick", lw=2.5, ms=10,
            label=f"Scale 1a (p$_1$=0.312)")
    ax.plot(L_values, eps_born_s2, "o-", color="royalblue", lw=2.5, ms=10,
            label=f"Scale 2 (p$_1$=0.902)")
    ax.set_xlabel("History length L", fontsize=14)
    ax.set_ylabel(r"$\varepsilon_\mathrm{Born}$", fontsize=14)
    ax.set_title(r"$\varepsilon$ at Born frequency vs L", fontsize=14, fontweight="bold")
    ax.legend(fontsize=12)

    fig.suptitle(
        f"Multi-scale filtering strength — 8 qubits (D=256), avg {n_avg} seeds",
        fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "spatial_gap.png")
    fig.savefig(OUT / "spatial_gap.pdf")
    plt.close(fig)
    print("  -> spatial_gap.png")


def print_summary():
    p = SpatialParams(L=L, hamming_threshold=THR, dt=DT, seed=42)
    r = simulate_spatial_multiscale(p)

    print(f"\n{'='*70}")
    print(f"SPATIAL MULTI-SCALE SUMMARY — L={L}, D={p.D}")
    print(f"{'='*70}")
    print(f"Scale 1a (left 4 qubits,  HW>={THR}): p1 = {p.p1_left:.4f}")
    print(f"Scale 1b (right 4 qubits, HW>={THR}): p1 = {p.p1_right:.4f}")
    print(f"Scale 2  (either group,   HW>={THR}): p1 = {p.p1_either:.4f}")
    print(f"  rank(Pi_either) = {p.d1_either} out of D={p.D}")

    for name, cr in [("1a(left)", r.scale1_left), ("1b(right)", r.scale1_right),
                      ("2(either)", r.scale2_either)]:
        born_n = round(cr.born_frequency)
        comb_n = L // 2
        v = ~np.isnan(cr.epsilon)
        eb = cr.epsilon[born_n] if v[born_n] else float('nan')
        ec = cr.epsilon[comb_n] if v[comb_n] else float('nan')
        g = ec / eb if eb > 0.001 else float('nan')
        print(f"\n  {name}: eps_born={eb:.3f}, eps_comb={ec:.3f}, gap={g:.1f}x")

    print(f"{'='*70}")


if __name__ == "__main__":
    print("Spatial multi-scale Born filtering analysis\n")
    print("Generating plots...")
    plot_overlay()
    plot_emergence()
    plot_gap_growth()
    print_summary()
    print(f"\nAll plots saved to {OUT}/")
