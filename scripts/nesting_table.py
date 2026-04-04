#!/usr/bin/env python3
"""Compute the emergent decoherence nesting table and figure (paper Section V.A).

Tracks all 4^L fine-grained branches and 2^L coarse-grained branches,
then computes containment, emergence, and leakage metrics.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from recohere.spatial_multiscale import SpatialParams, build_spatial_setup
from recohere.branches import _normalize_states, _normalized_gram_epsilon, _chunked_epsilon

OUT = Path("results")
OUT.mkdir(exist_ok=True)

THR, DT = 2, 1.2
EPS = 0.3


def run_nesting(L, seed):
    p = SpatialParams(L=L, hamming_threshold=THR, dt=DT, seed=seed)
    D = p.D
    psi0, U, mask_left, mask_right = build_spatial_setup(p)

    masks_4 = [
        (~mask_left & ~mask_right, 0, 0),
        (~mask_left & mask_right,  0, 1),
        (mask_left & ~mask_right,  1, 0),
        (mask_left & mask_right,   1, 1),
    ]

    # Scale 1: all 4^L branches
    current = {(): psi0.copy()}
    for step in range(L):
        next_b = {}
        for hist, state in current.items():
            evolved = U @ state
            for mask, dl, dr in masks_4:
                proj = np.where(mask, evolved, 0.0)
                if np.vdot(proj, proj).real < 1e-30:
                    continue
                next_b[hist + ((dl, dr),)] = proj
        current = next_b

    s1_hists = list(current.keys())
    s1_states = np.array([current[h] for h in s1_hists])
    s1_weights = np.sum(np.abs(s1_states)**2, axis=1)
    s1_norm = _normalize_states(s1_states, s1_weights)
    N = len(s1_hists)
    if N <= 8192:
        _, s1_epsilon = _normalized_gram_epsilon(s1_norm)
    else:
        s1_epsilon = _chunked_epsilon(s1_norm, chunk_size=1024)
    s1_alive = s1_epsilon < EPS

    # Scale 2: group by OR history
    s2_groups = {}
    for idx, h in enumerate(s1_hists):
        s2h = tuple(1 if (dl or dr) else 0 for dl, dr in h)
        s2_groups.setdefault(s2h, []).append(idx)

    s2_hists = sorted(s2_groups.keys())
    s2_states_arr = np.zeros((len(s2_hists), D), dtype=complex)
    for i, s2h in enumerate(s2_hists):
        for idx in s2_groups[s2h]:
            s2_states_arr[i] += s1_states[idx]

    s2_weights = np.sum(np.abs(s2_states_arr)**2, axis=1)
    s2_norm = _normalize_states(s2_states_arr, s2_weights)
    _, s2_epsilon = _normalized_gram_epsilon(s2_norm)
    s2_alive = s2_epsilon < EPS

    n_s1_alive = int(s1_alive.sum())
    n_s2_alive = int(s2_alive.sum())

    # Containment
    if n_s1_alive > 0:
        s1_in_alive_s2 = sum(
            1 for i, s2h in enumerate(s2_hists) if s2_alive[i]
            for idx in s2_groups[s2h] if s1_alive[idx]
        )
        containment = s1_in_alive_s2 / n_s1_alive
    else:
        containment = float('nan')

    # Emergence
    if n_s2_alive > 0:
        emergent = sum(
            1 for i, s2h in enumerate(s2_hists) if s2_alive[i]
            and not any(s1_alive[idx] for idx in s2_groups[s2h])
        )
        emergence_frac = emergent / n_s2_alive
    else:
        emergence_frac = float('nan')

    # Leakage
    n_s2_dead = len(s2_hists) - n_s2_alive
    if n_s2_dead > 0:
        leakers = sum(
            1 for i, s2h in enumerate(s2_hists) if not s2_alive[i]
            and any(s1_alive[idx] for idx in s2_groups[s2h])
        )
        leakage = leakers / n_s2_dead
    else:
        leakage = float('nan')

    return n_s1_alive, n_s2_alive, containment, emergence_frac, leakage


def run_nesting_raw(L, seed):
    """Return full branch-level data for the nesting figure."""
    p = SpatialParams(L=L, hamming_threshold=THR, dt=DT, seed=seed)
    D = p.D
    psi0, U, mask_left, mask_right = build_spatial_setup(p)

    masks_4 = [
        (~mask_left & ~mask_right, 0, 0),
        (~mask_left & mask_right,  0, 1),
        (mask_left & ~mask_right,  1, 0),
        (mask_left & mask_right,   1, 1),
    ]

    current = {(): psi0.copy()}
    for step in range(L):
        next_b = {}
        for hist, state in current.items():
            evolved = U @ state
            for mask, dl, dr in masks_4:
                proj = np.where(mask, evolved, 0.0)
                if np.vdot(proj, proj).real < 1e-30:
                    continue
                next_b[hist + ((dl, dr),)] = proj
        current = next_b

    s1_hists = list(current.keys())
    s1_states = np.array([current[h] for h in s1_hists])
    s1_weights = np.sum(np.abs(s1_states)**2, axis=1)
    s1_norm = _normalize_states(s1_states, s1_weights)
    N = len(s1_hists)
    if N <= 8192:
        _, s1_epsilon = _normalized_gram_epsilon(s1_norm)
    else:
        s1_epsilon = _chunked_epsilon(s1_norm, chunk_size=1024)
    s1_alive = s1_epsilon < EPS

    s2_groups = {}
    for idx, h in enumerate(s1_hists):
        s2h = tuple(1 if (dl or dr) else 0 for dl, dr in h)
        s2_groups.setdefault(s2h, []).append(idx)

    s2_hists = sorted(s2_groups.keys())
    s2_states_arr = np.zeros((len(s2_hists), D), dtype=complex)
    for i, s2h in enumerate(s2_hists):
        for idx in s2_groups[s2h]:
            s2_states_arr[i] += s1_states[idx]

    s2_weights = np.sum(np.abs(s2_states_arr)**2, axis=1)
    s2_norm = _normalize_states(s2_states_arr, s2_weights)
    _, s2_epsilon = _normalized_gram_epsilon(s2_norm)
    s2_alive = s2_epsilon < EPS
    s2_n_either = np.array([sum(h) for h in s2_hists])

    return s1_alive, s2_hists, s2_groups, s2_epsilon, s2_alive, s2_n_either


def plot_nesting_figure():
    """Generate the nesting visualization (paper Fig. spatial-nesting)."""
    L, SEED = 6, 42
    s1_alive, s2_hists, s2_groups, s2_epsilon, s2_alive, s2_n_either = \
        run_nesting_raw(L, SEED)

    order = np.argsort(s2_n_either)

    plt.rcParams.update({"font.size": 13, "axes.grid": True, "grid.alpha": 0.25,
                          "font.family": "serif"})
    fig, axes = plt.subplots(1, 2, figsize=(16, 10),
                              gridspec_kw={"width_ratios": [1, 3]})

    # Left: scale-2 epsilon bars
    ax = axes[0]
    y_pos = np.arange(len(s2_hists))
    colors = ["royalblue" if s2_alive[i] else "lightcoral" for i in order]
    ax.barh(y_pos, s2_epsilon[order], color=colors, height=0.8,
            edgecolor="white", linewidth=0.3)
    ax.axvline(EPS, color="gray", ls="--", lw=1.5, alpha=0.7)
    ax.set_xlabel(r"Scale-2 $\varepsilon$", fontsize=13)
    ax.set_ylabel(r"Scale-2 branch (sorted by $n_\mathrm{either}$)", fontsize=13)
    ax.set_ylim(-0.5, len(s2_hists) - 0.5)
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_title(f"Scale 2\n({s2_alive.sum()} / {len(s2_hists)} decoherent)",
                 fontsize=12)

    # Right: scale-1 members
    ax = axes[1]
    for j, s2_idx in enumerate(order):
        s2h = s2_hists[s2_idx]
        members = s2_groups[s2h]
        if members:
            x_pos = np.linspace(0, 1, len(members))
            for k, m_idx in enumerate(members):
                if s1_alive[m_idx]:
                    ax.plot(x_pos[k], j, "o", color="royalblue", ms=5,
                            zorder=5, markeredgecolor="black", markeredgewidth=0.5)
                else:
                    ax.plot(x_pos[k], j, ".", color="lightgray", ms=1.5,
                            alpha=0.4, rasterized=True)
        if s2_alive[s2_idx]:
            ax.axhspan(j - 0.4, j + 0.4, color="royalblue", alpha=0.05)

    ax.set_xlabel("Scale-1 branches within each scale-2 branch", fontsize=13)
    ax.set_ylim(-0.5, len(s2_hists) - 0.5)
    ax.set_xlim(-0.02, 1.02)
    ax.set_yticks([])
    ax.set_title(f"Scale-1 composition\n(blue dots = decoherent, "
                 f"{s1_alive.sum()} / {len(s1_alive)} total)", fontsize=12)

    fig.suptitle(
        f"Nesting: scale-2 branches and their scale-1 composition\n"
        f"8 qubits, L={L}, D=256. "
        f"Dead scale-2 branches (red) contain zero decoherent scale-1 branches.",
        fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "spatial_nesting.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "spatial_nesting.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  -> spatial_nesting.png")


if __name__ == "__main__":
    print("Emergent decoherence nesting table and figure\n")
    plot_nesting_figure()
    print(f"{'L':>3} {'4^L':>6} {'S1':>5} {'S2':>5}  {'Contain':>8} {'Emerge':>8} {'Leak':>8}  seeds")
    print("-" * 65)

    for L in [4, 5, 6, 7, 8]:
        n_seeds = 20 if L <= 6 else (10 if L == 7 else 5)
        cont_all, emerg_all, leak_all = [], [], []
        s1_all, s2_all = [], []

        for seed in range(n_seeds):
            n_s1a, n_s2a, cont, emerg, leak = run_nesting(L, seed)
            cont_all.append(cont)
            emerg_all.append(emerg)
            leak_all.append(leak)
            s1_all.append(n_s1a)
            s2_all.append(n_s2a)

        c = np.nanmean(cont_all)
        e = np.nanmean(emerg_all)
        l = np.nanmean(leak_all)
        s1 = np.mean(s1_all)
        s2 = np.mean(s2_all)
        c_str = f"{c:7.0%}" if not np.isnan(c) else "     --"
        print(f"{L:3d} {4**L:6d} {s1:5.0f} {s2:5.0f}  {c_str}  {e:7.0%}  {l:7.0%}   {n_seeds}")
