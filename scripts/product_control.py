#!/usr/bin/env python3
"""Product-rotation control: is Born filtering dynamics-driven or purely kinematic?

Compares the Ising (scrambling) unitary against a product rotation
exp(-i theta X)^{otimes m} (non-scrambling) using identical projectors,
initial states, and p1. If filtering vanishes with the product unitary,
the scrambling dynamics are a necessary ingredient.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from recohere.ising_direct import IsingDirectParams, simulate_and_analyze, simulate_product

OUT = Path("results")
OUT.mkdir(exist_ok=True)
plt.rcParams.update({
    "font.size": 13, "axes.grid": True, "grid.alpha": 0.25,
    "savefig.dpi": 200, "savefig.bbox": "tight", "font.family": "serif",
})

M, THR, L = 9, 4, 15
DT = 1.2
N_SEEDS = 10


def run():
    p1 = IsingDirectParams(m=M, L=1, hamming_threshold=THR).p1
    D = 2**M

    print(f"Product-rotation control experiment")
    print(f"m={M} ({D} dims), p1={p1:.3f}, L={L}, {N_SEEDS} seeds\n")

    ising_eps, product_eps = [], []

    for seed in range(N_SEEDS):
        p = IsingDirectParams(m=M, L=L, hamming_threshold=THR, dt=DT, seed=seed)

        r_ising = simulate_and_analyze(p)
        r_product = simulate_product(p)

        ising_eps.append(r_ising.epsilon)
        product_eps.append(r_product.epsilon)

        born = round(r_ising.born_frequency)
        print(f"  seed={seed}:  Ising  eps_born={r_ising.epsilon[born]:.3f}  eps_comb={r_ising.epsilon[L//2]:.3f}")
        print(f"           Product eps_born={r_product.epsilon[born]:.3f}  eps_comb={r_product.epsilon[L//2]:.3f}")

    ising_eps = np.array(ising_eps)
    product_eps = np.array(product_eps)
    return ising_eps, product_eps, p1


def plot_comparison(ising_eps, product_eps, p1):
    freq = np.arange(L + 1) / L

    ie_mean = np.nanmean(ising_eps, axis=0)
    ie_std = np.nanstd(ising_eps, axis=0)
    pe_mean = np.nanmean(product_eps, axis=0)
    pe_std = np.nanstd(product_eps, axis=0)

    fig, ax = plt.subplots(figsize=(12, 6.5))

    ax.plot(freq, ie_mean, "s-", color="firebrick", lw=2, ms=7,
            label=f"Ising (scrambling)", zorder=5)
    ax.fill_between(freq, ie_mean - ie_std, ie_mean + ie_std,
                    color="firebrick", alpha=0.12)

    ax.plot(freq, pe_mean, "D-", color="slategray", lw=2, ms=7,
            label=f"Product rotation (non-scrambling)", zorder=5)
    ax.fill_between(freq, pe_mean - pe_std, pe_mean + pe_std,
                    color="slategray", alpha=0.12)

    ax.axvline(p1, color="crimson", ls="--", lw=2, alpha=0.6,
               label=f"Born p$_1$={p1:.2f}")
    ax.axvline(0.5, color="seagreen", ls=":", lw=2, alpha=0.6,
               label="Combinatorial peak")

    ax.set_xlabel("Frequency $n_1/L$", fontsize=14)
    ax.set_ylabel(r"$\varepsilon$ (recoherence parameter)", fontsize=14)
    ax.set_title(
        f"Scrambling control: Ising vs product rotation ($D$={2**M}, $L$={L}, $p_1$={p1:.2f})\n"
        f"Shaded = $\\pm 1\\sigma$ over {N_SEEDS} initial states",
        fontsize=14, fontweight="bold")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "product_control.png")
    fig.savefig(OUT / "product_control.pdf")
    plt.close(fig)
    print(f"\n  -> {OUT / 'product_control.png'}")


def print_summary(ising_eps, product_eps, p1):
    born = round(L * p1)
    comb = L // 2
    ie = np.nanmean(ising_eps, axis=0)
    pe = np.nanmean(product_eps, axis=0)

    print(f"\n{'='*60}")
    print(f"SUMMARY ({N_SEEDS} seeds, D={2**M}, L={L}, p1={p1:.3f})")
    print(f"{'='*60}")
    print(f"{'':>25} {'Ising':>12} {'Product':>12}")
    print(f"{'eps at Born (n1='+str(born)+')':>25} {ie[born]:>10.3f}   {pe[born]:>10.3f}")
    print(f"{'eps at comb (n1='+str(comb)+')':>25} {ie[comb]:>10.3f}   {pe[comb]:>10.3f}")
    r_i = ie[comb] / ie[born] if ie[born] > 0.001 else float("inf")
    r_p = pe[comb] / pe[born] if pe[born] > 0.001 else float("inf")
    print(f"{'Ratio comb/born':>25} {r_i:>10.1f}x  {r_p:>10.1f}x")
    print(f"{'='*60}")

    if r_p < 1.3:
        print("\n>>> Product rotation shows NO filtering.")
        print("    Scrambling dynamics are necessary — not just projector asymmetry.")
    elif r_p > 0.8 * r_i:
        print("\n>>> Product rotation shows COMPARABLE filtering to Ising.")
        print("    The effect may be primarily kinematic (projector-driven).")
    else:
        print(f"\n>>> Product filtering is weaker ({r_p:.1f}x vs {r_i:.1f}x).")
        print("    Dynamics contribute but projector asymmetry plays a role.")


if __name__ == "__main__":
    ising_eps, product_eps, p1 = run()
    plot_comparison(ising_eps, product_eps, p1)
    print_summary(ising_eps, product_eps, p1)
