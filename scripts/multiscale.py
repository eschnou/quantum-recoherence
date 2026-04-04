#!/usr/bin/env python3
"""Multi-scale Born filtering: does Born's rule emerge at every coarse-graining?

Scale 0: Ising Hamiltonian, unitary evolution in D=512
Scale 1: Hamming-weight projector -> binary history (0/1), p1 ~ 0.746
Scale 2: Pairs of scale-1 outcomes -> ternary history (A=00, B=11, C=01/10)

Key questions:
  1. Do scale-2 survivors cluster at Born-typical (nA, nB, nC)?
  2. Do scale-2 survivors consist only of scale-1 survivors? (nesting)
  3. How does the survival fraction compare across scales?
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from itertools import product as iterproduct

from recohere.branches import (
    BranchResult, Scale2Result, Scale2FreqClassResult,
    simulate_branches, analyze_scale2, analyze_scale2_freq_classes,
)
from recohere.ising_direct import IsingDirectParams, simulate_and_analyze

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
# Odd L supported: scale-2 drops the last outcome (uses first L-1)
L_VALUES = [6, 8, 10, 12, 14, 16]


def run_all() -> dict[int, tuple[BranchResult, Scale2Result]]:
    results = {}
    p1 = IsingDirectParams(m=M, L=1, hamming_threshold=THR).p1
    p0 = 1 - p1
    print(f"Multi-scale analysis: m={M} (D={D}), p1={p1:.3f}, seed={SEED}")
    print(f"Scale-2 Born probs: pA={p0**2:.4f}, pB={p1**2:.4f}, pC={2*p0*p1:.4f}\n")

    for L in L_VALUES:
        L2 = L // 2  # odd L drops last outcome
        print(f"  L={L}: {2**L} scale-1 branches, L₂={L2} ({3**L2} possible scale-2 histories)")
        p = IsingDirectParams(m=M, L=L, hamming_threshold=THR, dt=DT, seed=SEED)
        br = simulate_branches(p)
        s2 = analyze_scale2(br)

        n1_alive = np.sum(br.epsilon < EPS_THRESHOLD)
        n2_alive = np.sum(s2.epsilon < EPS_THRESHOLD)
        print(f"    Scale 1: {len(br.histories)} branches, {n1_alive} decoherent")
        print(f"    Scale 2: {len(s2.scale2_histories)} branches, {n2_alive} decoherent")
        results[L] = (br, s2)

    return results


def plot_scale2_death(results: dict[int, tuple[BranchResult, Scale2Result]]):
    """Scatter: epsilon vs Born-distance for scale-2 branches."""
    nrows = (len(L_VALUES) + 2) // 3
    fig, axes = plt.subplots(nrows, 3, figsize=(20, 6 * nrows))

    for ax, L in zip(axes.flatten(), L_VALUES):
        br, s2 = results[L]
        L2 = L // 2
        pA, pB, pC = s2.p_born

        # Born-typical frequency for each branch
        fB = s2.n_abc[:, 1] / L2  # fraction of B outcomes
        log_w = np.log10(np.maximum(s2.weights, 1e-50))

        sc = ax.scatter(fB, s2.epsilon, c=log_w, s=15, alpha=0.6,
                        cmap="viridis", vmin=log_w.min(), vmax=0, rasterized=True)
        ax.axhline(EPS_THRESHOLD, color="gray", ls="--", lw=1, alpha=0.5)
        ax.axvline(pB, color="crimson", ls="--", lw=2, alpha=0.7,
                   label=f"Born pB={pB:.3f}")
        ax.axvline(1/3, color="seagreen", ls=":", lw=2, alpha=0.7,
                   label="Uniform 1/3")

        n_alive = np.sum(s2.epsilon < EPS_THRESHOLD)
        ax.set_title(f"L={L} (L₂={L2})   {len(s2.scale2_histories)} branches, "
                     f"{n_alive} decoherent", fontsize=12)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.03, 1.05)
        ax.set_xlabel("fB = nB/L₂ (fraction of B=11)")
        ax.set_ylabel("ε (max overlap)")
        if L == L_VALUES[0]:
            ax.legend(fontsize=9)

    for i in range(len(L_VALUES), len(axes.flatten())):
        axes.flatten()[i].set_visible(False)

    fig.suptitle(
        f"Scale-2 branch decoherence — {M} qubits (D={D})\n"
        f"A=00, B=11, C=01/10. Red = Born frequency pB, color = log₁₀ weight",
        fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "multiscale_death.png")
    fig.savefig(OUT / "multiscale_death.pdf")
    plt.close(fig)
    print("  -> multiscale_death.png")


def plot_scale_comparison(results: dict[int, tuple[BranchResult, Scale2Result]]):
    """Side-by-side: scale-1 vs scale-2 death-of-worlds for largest L."""
    L = max(L_VALUES)
    br, s2 = results[L]
    L2 = L // 2

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Scale 1
    ax = axes[0]
    freq1 = br.n1 / L
    log_w1 = np.log10(np.maximum(br.weights, 1e-50))
    ax.scatter(freq1, br.epsilon, c=log_w1, s=8, alpha=0.6,
               cmap="viridis", vmin=log_w1.min(), vmax=0, rasterized=True)
    ax.axhline(EPS_THRESHOLD, color="gray", ls="--", lw=1, alpha=0.5)
    ax.axvline(br.p1, color="crimson", ls="--", lw=2, alpha=0.7)
    ax.axvline(0.5, color="seagreen", ls=":", lw=2, alpha=0.7)
    n1_alive = int(np.sum(br.epsilon < EPS_THRESHOLD))
    ax.set_title(f"Scale 1: L={L}, {len(br.histories)} branches, "
                 f"{n1_alive} decoherent\n"
                 f"x-axis = n₁/L (fraction of outcome '1')", fontsize=12)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("f₁ = n₁/L")
    ax.set_ylabel("ε (max overlap)")

    # Scale 2
    ax = axes[1]
    fB = s2.n_abc[:, 1] / L2
    log_w2 = np.log10(np.maximum(s2.weights, 1e-50))
    ax.scatter(fB, s2.epsilon, c=log_w2, s=15, alpha=0.6,
               cmap="viridis", vmin=log_w2.min(), vmax=0, rasterized=True)
    ax.axhline(EPS_THRESHOLD, color="gray", ls="--", lw=1, alpha=0.5)
    ax.axvline(s2.p_born[1], color="crimson", ls="--", lw=2, alpha=0.7)
    ax.axvline(1/3, color="seagreen", ls=":", lw=2, alpha=0.7)
    n2_alive = int(np.sum(s2.epsilon < EPS_THRESHOLD))
    ax.set_title(f"Scale 2: L₂={L2}, {len(s2.scale2_histories)} branches, "
                 f"{n2_alive} decoherent\n"
                 f"x-axis = nB/L₂ (fraction of B=11)", fontsize=12)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("fB = nB/L₂")
    ax.set_ylabel("ε (max overlap)")

    fig.suptitle(
        f"Scale comparison — {M} qubits (D={D}), ε < {EPS_THRESHOLD}\n"
        f"Born filtering at both scales: survivors cluster near Born-typical frequencies",
        fontsize=14, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(OUT / "multiscale_comparison.png")
    fig.savefig(OUT / "multiscale_comparison.pdf")
    plt.close(fig)
    print("  -> multiscale_comparison.png")


def plot_nesting(results: dict[int, tuple[BranchResult, Scale2Result]]):
    """Nesting analysis: do scale-2 survivors consist of scale-1 survivors?"""
    L = max(L_VALUES)
    br, s2 = results[L]

    scale1_alive = br.epsilon < EPS_THRESHOLD

    # For each scale-2 branch, compute fraction of its scale-1 members that survived
    s2_hists = s2.scale2_histories
    frac_alive = np.zeros(len(s2_hists))
    for i, s2h in enumerate(s2_hists):
        members = s2.scale1_members[s2h]
        if len(members) > 0:
            frac_alive[i] = np.mean(scale1_alive[members])

    scale2_alive = s2.epsilon < EPS_THRESHOLD

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Left: scatter of scale-2 epsilon vs fraction of scale-1 survivors
    ax = axes[0]
    colors = np.where(scale2_alive, "royalblue", "lightcoral")
    ax.scatter(frac_alive, s2.epsilon, c=colors, s=20, alpha=0.6, rasterized=True)
    ax.axhline(EPS_THRESHOLD, color="gray", ls="--", lw=1)
    ax.set_xlabel("Fraction of scale-1 members that are decoherent", fontsize=12)
    ax.set_ylabel("Scale-2 ε", fontsize=12)
    ax.set_title("Nesting: scale-2 ε vs scale-1 survival fraction", fontsize=13,
                 fontweight="bold")

    # Right: histogram of scale-1 survival fraction, split by scale-2 status
    ax = axes[1]
    bins = np.linspace(0, 1, 21)
    ax.hist(frac_alive[scale2_alive], bins=bins, color="royalblue", alpha=0.7,
            label=f"Scale-2 survivors (ε<{EPS_THRESHOLD})", edgecolor="white")
    ax.hist(frac_alive[~scale2_alive], bins=bins, color="lightcoral", alpha=0.5,
            label=f"Scale-2 dead (ε≥{EPS_THRESHOLD})", edgecolor="white")
    ax.set_xlabel("Fraction of scale-1 members that are decoherent", fontsize=12)
    ax.set_ylabel("Count of scale-2 branches", fontsize=12)
    ax.set_title("Distribution of scale-1 survival within scale-2 branches",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)

    # Summary stats
    if np.any(scale2_alive):
        mean_alive = np.mean(frac_alive[scale2_alive])
        mean_dead = np.mean(frac_alive[~scale2_alive]) if np.any(~scale2_alive) else 0
        fig.text(0.5, -0.02,
                 f"L={L}: Scale-2 survivors have {mean_alive:.1%} scale-1 survivors on average "
                 f"(vs {mean_dead:.1%} for dead scale-2 branches)",
                 ha="center", fontsize=12, style="italic")

    fig.suptitle(
        f"Nesting analysis — {M} qubits, L={L}\n"
        f"Do scale-2 surviving worlds consist of scale-1 surviving worlds?",
        fontsize=14, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(OUT / "multiscale_nesting.png")
    fig.savefig(OUT / "multiscale_nesting.pdf")
    plt.close(fig)
    print("  -> multiscale_nesting.png")


def plot_census(results: dict[int, tuple[BranchResult, Scale2Result]]):
    """Census: number of decoherent branches at each scale vs L."""
    L_all = sorted(results.keys())

    total_s1 = [2**L for L in L_all]
    alive_s1 = [int(np.sum(results[L][0].epsilon < EPS_THRESHOLD)) for L in L_all]
    total_s2 = [len(results[L][1].scale2_histories) for L in L_all]
    alive_s2 = [int(np.sum(results[L][1].epsilon < EPS_THRESHOLD)) for L in L_all]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.semilogy(L_all, total_s1, "o--", color="lightcoral", lw=1.5, ms=8,
                label="Total scale-1 (2ᴸ)", alpha=0.7)
    ax.semilogy(L_all, alive_s1, "o-", color="firebrick", lw=2.5, ms=10,
                label=f"Scale-1 decoherent (ε<{EPS_THRESHOLD})")
    ax.semilogy(L_all, total_s2, "s--", color="lightskyblue", lw=1.5, ms=8,
                label="Total scale-2 (3^(L/2))", alpha=0.7)
    ax.semilogy(L_all, alive_s2, "s-", color="royalblue", lw=2.5, ms=10,
                label=f"Scale-2 decoherent (ε<{EPS_THRESHOLD})")
    ax.axhline(D, color="black", ls="--", lw=2, alpha=0.5,
               label=f"D = {D}")

    ax.set_xlabel("History length L (scale-1 steps)", fontsize=14)
    ax.set_ylabel("Number of branches", fontsize=14)
    ax.set_title(
        f"Multi-scale census — {M} qubits (D={D})\n"
        f"Both scales: decoherent branches saturate while total grows exponentially",
        fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xticks(L_all)
    fig.tight_layout()
    fig.savefig(OUT / "multiscale_census.png")
    fig.savefig(OUT / "multiscale_census.pdf")
    plt.close(fig)
    print("  -> multiscale_census.png")


def plot_survivor_triangle(results: dict[int, tuple[BranchResult, Scale2Result]]):
    """Ternary-ish plot: scale-2 survivor frequencies in (nA, nB, nC) space."""
    L = max(L_VALUES)
    br, s2 = results[L]
    L2 = L // 2
    pA, pB, pC = s2.p_born

    alive = s2.epsilon < EPS_THRESHOLD
    fA = s2.n_abc[:, 0] / L2
    fB = s2.n_abc[:, 1] / L2

    fig, ax = plt.subplots(figsize=(10, 8))

    # All branches (faded)
    ax.scatter(fA[~alive], fB[~alive], c="lightgray", s=15, alpha=0.3,
               label="Dead", rasterized=True)
    # Survivors colored by epsilon
    sc = ax.scatter(fA[alive], fB[alive], c=s2.epsilon[alive], s=40, alpha=0.8,
                    cmap="viridis_r", vmin=0, vmax=EPS_THRESHOLD,
                    edgecolors="black", linewidths=0.3, label="Decoherent",
                    rasterized=True)
    plt.colorbar(sc, ax=ax, label="ε", shrink=0.8)

    # Born point
    ax.plot(pA, pB, "*", color="crimson", ms=20, zorder=10, label=f"Born ({pA:.3f}, {pB:.3f})")
    # Uniform point
    ax.plot(1/3, 1/3, "D", color="seagreen", ms=12, zorder=10, label="Uniform (1/3, 1/3)")

    ax.set_xlabel("fA = nA/L₂ (fraction of A=00)", fontsize=13)
    ax.set_ylabel("fB = nB/L₂ (fraction of B=11)", fontsize=13)
    ax.set_title(
        f"Scale-2 survivor map — L={L} (L₂={L2}), {int(alive.sum())} survivors\n"
        f"fC = 1 - fA - fB (implicit). Survivors cluster at Born point, not uniform.",
        fontsize=13, fontweight="bold")
    ax.legend(fontsize=11, loc="upper right")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    # Draw the simplex boundary (fA + fB <= 1)
    ax.plot([0, 1], [1, 0], "k-", lw=1, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "multiscale_triangle.png")
    fig.savefig(OUT / "multiscale_triangle.pdf")
    plt.close(fig)
    print("  -> multiscale_triangle.png")


def _epsilon_by_nB(fc: Scale2FreqClassResult) -> np.ndarray:
    """Weight-average frequency-class epsilon by nB value."""
    L2 = fc.L2
    eps_by_nB = np.full(L2 + 1, np.nan)
    counts = np.array([c[1] for c in fc.freq_classes])
    for nB in range(L2 + 1):
        mask = (counts == nB) & ~np.isnan(fc.epsilon) & (fc.weights > 1e-30)
        if np.any(mask):
            eps_by_nB[nB] = np.average(fc.epsilon[mask], weights=fc.weights[mask])
    return eps_by_nB


def plot_scale2_emergence(results: dict[int, tuple[BranchResult, Scale2Result]]):
    """Scale-2 emergence: frequency-class epsilon vs fB as L grows."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 10), sharey=True)

    for ax, L in zip(axes.flatten(), L_VALUES):
        br, s2 = results[L]
        fc = analyze_scale2_freq_classes(s2)
        L2 = fc.L2
        pB = fc.p_born[1]

        eps_by_nB = _epsilon_by_nB(fc)
        nB_values = np.arange(0, L2 + 1)
        fB_vals = nB_values / L2
        valid = ~np.isnan(eps_by_nB)
        ax.bar(fB_vals[valid], eps_by_nB[valid], width=0.7 / L2,
               color="steelblue", alpha=0.75, edgecolor="steelblue")
        ax.axvline(pB, color="crimson", ls="--", lw=2, alpha=0.7)
        ax.axvline(1/3, color="seagreen", ls=":", lw=2, alpha=0.7)
        ax.set_title(f"L = {L} (L₂ = {L2})", fontsize=14, fontweight="bold")
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlim(-0.05, 1.05)
        ax.set_xlabel("fB = nB/L₂", fontsize=12)
        if ax in axes[:, 0]:
            ax.set_ylabel("ε(nB)", fontsize=13)

    fig.suptitle(
        f"Emergence of Born filtering at scale 2 — {M} qubits, p₁={results[L_VALUES[0]][0].p1:.2f}\n"
        f"Red = Born pB, green = uniform 1/3. Frequency-class ε (weight-averaged over nA/nC splits).",
        fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "multiscale_emergence.png")
    fig.savefig(OUT / "multiscale_emergence.pdf")
    plt.close(fig)
    print("  -> multiscale_emergence.png")


def plot_scale2_overlay(results: dict[int, tuple[BranchResult, Scale2Result]]):
    """Overlay: scale-1 freq-class epsilon vs scale-2 freq-class epsilon."""
    L = max(L_VALUES)
    br, s2 = results[L]
    fc = analyze_scale2_freq_classes(s2)
    L2 = fc.L2
    pA, pB, pC = fc.p_born

    fig, ax = plt.subplots(figsize=(12, 6.5))

    # Scale 1: proper frequency-class epsilon from simulate_and_analyze
    p = IsingDirectParams(m=M, L=L, hamming_threshold=THR, dt=DT, seed=SEED)
    r1 = simulate_and_analyze(p)
    freq1 = np.arange(L + 1) / L
    valid1 = ~np.isnan(r1.epsilon)
    ax.plot(freq1[valid1], r1.epsilon[valid1], "s-", color="firebrick", lw=2, ms=7,
            label=f"Scale 1: ε(n₁/L), L={L}", zorder=5)

    # Scale 2: frequency-class epsilon, projected onto fB axis
    eps2 = _epsilon_by_nB(fc)
    freq2 = np.arange(L2 + 1) / L2
    valid2 = ~np.isnan(eps2)
    ax.plot(freq2[valid2], eps2[valid2], "o-", color="royalblue", lw=2, ms=7,
            label=f"Scale 2: ε(nB/L₂), L₂={L2}", zorder=5)

    # Reference lines
    ax.axvline(br.p1, color="crimson", ls="--", lw=2, alpha=0.6,
               label=f"Born p₁={br.p1:.3f}")
    ax.axvline(pB, color="navy", ls="--", lw=2, alpha=0.6,
               label=f"Born pB={pB:.3f}")
    ax.axvline(0.5, color="seagreen", ls=":", lw=2, alpha=0.6,
               label="Combinatorial / uniform")

    ax.set_xlabel("Frequency (n₁/L or nB/L₂)", fontsize=14)
    ax.set_ylabel("ε (recoherence parameter)", fontsize=14)
    ax.set_title(
        f"Born filtering at two scales — {M} qubits (D={D})\n"
        f"Frequency-class ε: both scales dip near their Born frequency",
        fontsize=14, fontweight="bold")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "multiscale_overlay.png")
    fig.savefig(OUT / "multiscale_overlay.pdf")
    plt.close(fig)
    print("  -> multiscale_overlay.png")


def print_summary(results: dict[int, tuple[BranchResult, Scale2Result]]):
    """Print a summary table."""
    L = max(L_VALUES)
    br, s2 = results[L]
    L2 = L // 2
    pA, pB, pC = s2.p_born

    print(f"\n{'='*70}")
    print(f"MULTI-SCALE SUMMARY — L={L}, m={M} (D={D})")
    print(f"{'='*70}")

    s1_alive = br.epsilon < EPS_THRESHOLD
    s2_alive = s2.epsilon < EPS_THRESHOLD

    print(f"\nScale 1 (binary 0/1):")
    print(f"  Born p1 = {br.p1:.4f}")
    print(f"  Branches: {len(br.histories)}, decoherent: {int(s1_alive.sum())}")
    if np.any(s1_alive):
        print(f"  Survivor mean n1/L = {np.mean(br.n1[s1_alive]/L):.3f} "
              f"(Born: {br.p1:.3f})")

    print(f"\nScale 2 (ternary A/B/C):")
    print(f"  Born: pA={pA:.4f}, pB={pB:.4f}, pC={pC:.4f}")
    print(f"  Branches: {len(s2.scale2_histories)}, decoherent: {int(s2_alive.sum())}")
    if np.any(s2_alive):
        mean_fA = np.mean(s2.n_abc[s2_alive, 0] / L2)
        mean_fB = np.mean(s2.n_abc[s2_alive, 1] / L2)
        mean_fC = np.mean(s2.n_abc[s2_alive, 2] / L2)
        print(f"  Survivor mean (fA, fB, fC) = ({mean_fA:.3f}, {mean_fB:.3f}, {mean_fC:.3f})")

    # Nesting
    frac_alive_in_s2_survivors = []
    frac_alive_in_s2_dead = []
    for i, s2h in enumerate(s2.scale2_histories):
        members = s2.scale1_members[s2h]
        if len(members) > 0:
            f = np.mean(s1_alive[members])
            if s2_alive[i]:
                frac_alive_in_s2_survivors.append(f)
            else:
                frac_alive_in_s2_dead.append(f)

    print(f"\nNesting:")
    if frac_alive_in_s2_survivors:
        print(f"  Scale-2 survivors: {np.mean(frac_alive_in_s2_survivors):.1%} "
              f"of scale-1 members are decoherent")
    if frac_alive_in_s2_dead:
        print(f"  Scale-2 dead:      {np.mean(frac_alive_in_s2_dead):.1%} "
              f"of scale-1 members are decoherent")
    print(f"{'='*70}")


if __name__ == "__main__":
    print("Multi-scale Born filtering analysis\n")
    results = run_all()
    print("\nGenerating plots...")
    plot_scale2_death(results)
    plot_scale_comparison(results)
    plot_nesting(results)
    plot_census(results)
    plot_survivor_triangle(results)
    plot_scale2_emergence(results)
    plot_scale2_overlay(results)
    print_summary(results)
    print(f"\nAll plots saved to {OUT}/")
