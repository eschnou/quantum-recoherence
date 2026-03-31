#!/usr/bin/env python3
"""Look inside individual branches: what does one reality experience?"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from recohere.branches import simulate_branches
from recohere.ising_direct import IsingDirectParams

OUT = Path("results")
OUT.mkdir(exist_ok=True)
plt.rcParams.update({
    "font.size": 13, "axes.grid": True, "grid.alpha": 0.25,
    "savefig.dpi": 200, "savefig.bbox": "tight", "font.family": "serif",
})

M, THR, L = 9, 4, 12
DT, SEED = 1.2, 42


def pick_branches(r):
    """Pick representative branches: Born-typical survivor, combinatorial, dying."""
    born_n1 = round(L * r.p1)  # ~9
    comb_n1 = L // 2  # 6

    # Best survivor near Born frequency
    born_mask = (r.n1 == born_n1)
    if not born_mask.any():
        born_mask = (np.abs(r.n1 - born_n1) <= 1)
    born_idx = np.where(born_mask)[0]
    best_born = born_idx[np.argmin(r.epsilon[born_idx])]

    # Branch near combinatorial peak with highest epsilon (dying world)
    comb_mask = (r.n1 == comb_n1)
    if not comb_mask.any():
        comb_mask = (np.abs(r.n1 - comb_n1) <= 1)
    comb_idx = np.where(comb_mask)[0]
    worst_comb = comb_idx[np.argmax(r.epsilon[comb_idx])]

    # A "lucky" extreme branch (very few 1s)
    low_mask = (r.n1 <= 3) & (r.weights > 1e-30)
    if low_mask.any():
        low_idx = np.where(low_mask)[0]
        extreme = low_idx[np.argmax(r.weights[low_idx])]
    else:
        extreme = np.argmin(r.n1)

    # A mediocre survivor: near Born but higher epsilon
    mediocre_mask = (np.abs(r.n1 - born_n1) <= 1) & (r.epsilon > 0.2) & (r.epsilon < 0.6)
    if mediocre_mask.any():
        med_idx = np.where(mediocre_mask)[0]
        mediocre = med_idx[0]
    else:
        mediocre = best_born  # fallback

    return {
        "Born survivor\n(decoherent)": best_born,
        "Combinatorial\n(recoherent)": worst_comb,
        "Rare extreme\n(n₁ very low)": extreme,
        "Marginal\n(borderline)": mediocre,
    }


def plot_realities(r, picks):
    fig, axes = plt.subplots(len(picks), 3, figsize=(18, 3.2 * len(picks)),
                             gridspec_kw={"width_ratios": [2, 3, 3]})

    for row, (label, idx) in enumerate(picks.items()):
        history = r.histories[idx]
        n1 = r.n1[idx]
        eps = r.epsilon[idx]
        weight = r.weights[idx]

        # Column 1: the outcome sequence as colored blocks
        ax = axes[row, 0]
        for k, outcome in enumerate(history):
            color = "crimson" if outcome == 1 else "steelblue"
            ax.barh(0, 1, left=k, height=0.6, color=color, edgecolor="white", lw=0.5)
            ax.text(k + 0.5, 0, str(outcome), ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold")
        ax.set_xlim(0, L)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.set_xlabel("Step")
        ax.set_title(f"{label}", fontsize=12, fontweight="bold")
        ax.text(L + 0.3, 0, f"n₁={n1}/{L}\nε={eps:.3f}\nw={weight:.2e}",
                fontsize=10, va="center", ha="left",
                transform=ax.transData)
        ax.set_xlim(0, L + 3.5)

        # Column 2: running frequency
        ax = axes[row, 1]
        cumulative_n1 = np.cumsum(history)
        steps = np.arange(1, L + 1)
        running_freq = cumulative_n1 / steps
        ax.plot(steps, running_freq, "o-", color="firebrick", lw=2, ms=5)
        ax.axhline(r.p1, color="crimson", ls="--", lw=2, alpha=0.6, label=f"Born p₁={r.p1:.3f}")
        ax.axhline(0.5, color="seagreen", ls=":", lw=2, alpha=0.6, label="0.5")
        ax.set_xlim(0.5, L + 0.5)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Step k")
        ax.set_ylabel("Running freq n₁(k)/k")
        ax.set_title("Running frequency", fontsize=12)
        if row == 0:
            ax.legend(fontsize=10, loc="upper right")

        # Column 3: weight evolution along this path
        ax = axes[row, 2]
        # Reconstruct intermediate weights by replaying the branch
        intermediate_norms = []
        state = r.branch_states[idx]  # final state
        # We need to reconstruct from scratch — replay the path
        # (we'll do this below in a helper)
        p = IsingDirectParams(m=M, L=L, hamming_threshold=THR,
                              dt=DT, seed=SEED)
        from recohere.ising_direct import build_ising_setup
        psi0, U, mask_1 = build_ising_setup(p)

        state = psi0.copy()
        step_weights = [np.vdot(state, state).real]
        for k, outcome in enumerate(history):
            evolved = U @ state
            if outcome == 1:
                state = np.where(mask_1, evolved, 0.0)
            else:
                state = np.where(~mask_1, evolved, 0.0)
            step_weights.append(np.vdot(state, state).real)

        ax.semilogy(range(L + 1), step_weights, "s-", color="royalblue", lw=2, ms=5)
        ax.set_xlabel("Step k")
        ax.set_ylabel("Weight ||ψ(x,k)||²")
        ax.set_title("Weight (how much 'reality')", fontsize=12)
        ax.set_xlim(-0.3, L + 0.3)

    fig.suptitle(
        f"Four realities from {2**L} branches — {M} qubits (D={2**M}), L={L}, p₁={r.p1:.3f}\n"
        f"Blue=outcome 0, Red=outcome 1",
        fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "one_reality.png")
    fig.savefig(OUT / "one_reality.pdf")
    plt.close(fig)
    print("  -> one_reality.png")


if __name__ == "__main__":
    print("Extracting individual realities...\n")
    p = IsingDirectParams(m=M, L=L, hamming_threshold=THR,
                          dt=DT, seed=SEED)
    r = simulate_branches(p)
    print(f"  {len(r.histories)} branches, p1={r.p1:.3f}")

    picks = pick_branches(r)
    for label, idx in picks.items():
        h = r.histories[idx]
        print(f"  {label.replace(chr(10), ' ')}: "
              f"{''.join(str(x) for x in h)}  n1={r.n1[idx]}  eps={r.epsilon[idx]:.3f}")

    print("\nGenerating plot...")
    plot_realities(r, picks)
    print(f"Saved to {OUT}/")
