#!/usr/bin/env python3
"""Projector search: all 4 experiments in one server-ready script.

Usage:
    poetry run python scripts/projector_search_all.py              # all experiments
    poetry run python scripts/projector_search_all.py --exp 1      # just experiment 1
    poetry run python scripts/projector_search_all.py --exp 1 2 3  # experiments 1-3

Outputs saved to results/projector_search/
"""

import argparse
import csv
import sys
import time
from math import comb
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm, norm as la_norm

# ---------------------------------------------------------------------------
# Inline physics core (no dependency on the recohere package)
# ---------------------------------------------------------------------------

def compute_gram_matrix(freq_states: list[np.ndarray]) -> np.ndarray:
    M = np.array(freq_states)
    return M @ M.conj().T


def compute_epsilon(gram_matrix: np.ndarray) -> np.ndarray:
    n = gram_matrix.shape[0]
    weights = gram_matrix.diagonal().real
    epsilon = np.full(n, np.nan)
    for i in range(n):
        if weights[i] < 1e-30:
            continue
        max_fidelity = 0.0
        for j in range(n):
            if j == i or weights[j] < 1e-30:
                continue
            fidelity = np.abs(gram_matrix[i, j]) / np.sqrt(weights[i] * weights[j])
            max_fidelity = max(max_fidelity, fidelity)
        epsilon[i] = max_fidelity
    return epsilon


def build_ising_hamiltonian(m, J=1.0, hx=0.9045, hz=0.0):
    D = 2**m
    H = np.zeros((D, D), dtype=complex)
    for i in range(m - 1):
        for v in range(D):
            bi = (v >> i) & 1
            bi1 = (v >> (i + 1)) & 1
            H[v, v] += (J / 2) * (1 - 2 * (bi ^ bi1))
    for i in range(m):
        for v in range(D):
            w = v ^ (1 << i)
            H[v, w] += hx
    for i in range(m):
        for v in range(D):
            bi = (v >> i) & 1
            H[v, v] += hz * (1 - 2 * bi)
    return H


def frequency_class_analysis(psi0, U, mask_1, L, p1):
    D = psi0.shape[0]
    current = {0: psi0.copy()}
    for step in range(L):
        next_states = {}
        for n1_so_far, state in current.items():
            evolved = U @ state
            state_0 = np.where(~mask_1, evolved, 0.0)
            state_1 = np.where(mask_1, evolved, 0.0)
            for outcome, projected in [(0, state_0), (1, state_1)]:
                if np.vdot(projected, projected).real < 1e-30:
                    continue
                new_n1 = n1_so_far + outcome
                if new_n1 not in next_states:
                    next_states[new_n1] = np.zeros(D, dtype=complex)
                next_states[new_n1] += projected
        current = next_states

    freq_states = [np.zeros(D, dtype=complex) for _ in range(L + 1)]
    for n1, state in current.items():
        if 0 <= n1 <= L:
            freq_states[n1] = state
    gram = compute_gram_matrix(freq_states)
    weights = gram.diagonal().real
    eps = compute_epsilon(gram)
    return {
        "gram": gram, "epsilon": eps, "weights": weights,
        "born_frequency": L * p1,
    }


# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------

M = 9
D = 2**M          # 512
L = 15
DT = 1.2
J = 1.0
HX = 0.9045
HZ = 0.0
D1 = 382           # rank (p1 ≈ 0.746)
P1 = D1 / D
BORN_N1 = round(L * P1)  # 11
COMB_N1 = L // 2         # 7

OUT = Path("results/projector_search")

plt.rcParams.update({
    "font.size": 13, "axes.grid": True, "grid.alpha": 0.25,
    "savefig.dpi": 200, "savefig.bbox": "tight", "font.family": "serif",
})


def make_initial_state(seed, D=D):
    rng = np.random.default_rng(seed)
    psi0 = rng.standard_normal(D) + 1j * rng.standard_normal(D)
    return (psi0 / np.linalg.norm(psi0)).astype(np.complex128)


def hamming_mask(m=M, threshold=4):
    return np.array([bin(v).count("1") >= threshold for v in range(2**m)])


def evaluate_gap(mask_1, U, seeds):
    """Mean gap ratio over given seed list."""
    eb_list, ec_list = [], []
    for seed in seeds:
        psi0 = make_initial_state(seed)
        r = frequency_class_analysis(psi0, U, mask_1, L, P1)
        eb, ec = r["epsilon"][BORN_N1], r["epsilon"][COMB_N1]
        if not np.isnan(eb): eb_list.append(eb)
        if not np.isnan(ec): ec_list.append(ec)
    eps_born = np.mean(eb_list) if eb_list else 1e-10
    eps_comb = np.mean(ec_list) if ec_list else np.nan
    return (eps_comb / eps_born) if eps_born > 0.001 else np.nan, eps_born, eps_comb


def setup_physics():
    """Build H and U once. Returns (H, U)."""
    print("Building Hamiltonian and unitary...", flush=True)
    H = build_ising_hamiltonian(M, J, HX, HZ)
    U = expm(-1j * H * DT)
    return H, U


# ===================================================================
# Experiment 1: Random computational-basis projectors
# ===================================================================

def experiment_1(U):
    N_PROJ = 2000
    N_SEEDS = 5
    seeds = list(range(N_SEEDS))

    print(f"\n{'='*70}", flush=True)
    print(f"EXPERIMENT 1: Random projector scan", flush=True)
    print(f"  {N_PROJ} projectors, d1={D1}, {N_SEEDS} seeds each", flush=True)
    est = N_PROJ * N_SEEDS * 0.018
    print(f"  Estimated runtime: {est/60:.1f} min", flush=True)
    print(f"{'='*70}", flush=True)

    rng = np.random.default_rng(12345)
    all_idx = np.arange(D)

    gap_ratios = np.empty(N_PROJ)
    eps_borns = np.empty(N_PROJ)
    eps_combs = np.empty(N_PROJ)
    masks = []

    t0 = time.time()
    for i in range(N_PROJ):
        subset = rng.choice(all_idx, size=D1, replace=False)
        subset.sort()
        masks.append(subset)
        mask_1 = np.zeros(D, dtype=bool)
        mask_1[subset] = True
        gap_ratios[i], eps_borns[i], eps_combs[i] = evaluate_gap(mask_1, U, seeds)
        if (i + 1) % 200 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{N_PROJ} ({el:.0f}s, ETA {(N_PROJ-i-1)/(i+1)*el:.0f}s)", flush=True)

    el = time.time() - t0
    print(f"  Done in {el:.0f}s", flush=True)

    # Hamming reference
    hm = hamming_mask()
    gap_h, eb_h, ec_h = evaluate_gap(hm, U, seeds)
    print(f"  Hamming reference: gap={gap_h:.2f}, eps_born={eb_h:.3f}, eps_comb={ec_h:.3f}", flush=True)

    valid = ~np.isnan(gap_ratios)
    vg = gap_ratios[valid]
    print(f"  Random projectors ({np.sum(valid)} valid):", flush=True)
    print(f"    gap: mean={np.mean(vg):.2f}, median={np.median(vg):.2f}, "
          f"std={np.std(vg):.2f}, range=[{np.min(vg):.2f}, {np.max(vg):.2f}]", flush=True)

    # Sort and save top/bottom 50
    order = np.argsort(gap_ratios)[::-1]
    top50_ids = order[:50]
    bot50_ids = order[-50:]

    np.savez(OUT / "exp1_results.npz",
             gap_ratios=gap_ratios, eps_borns=eps_borns, eps_combs=eps_combs,
             masks=np.array(masks, dtype=object),
             top50_ids=top50_ids, bottom50_ids=bot50_ids,
             top50_masks=np.array([masks[i] for i in top50_ids], dtype=object),
             bottom50_masks=np.array([masks[i] for i in bot50_ids], dtype=object),
             hamming_gap=gap_h, hamming_eps_born=eb_h, hamming_eps_comb=ec_h)

    # Plots
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(vg, bins=60, color="steelblue", alpha=0.75, edgecolor="white")
    ax.axvline(gap_h, color="crimson", ls="--", lw=2.5,
               label=f"Hamming projector (gap={gap_h:.2f})")
    ax.axvline(1.0, color="gray", ls=":", lw=1.5, alpha=0.5, label="No filtering")
    ax.set_xlabel("Gap ratio: ε(comb) / ε(Born)", fontsize=14)
    ax.set_ylabel("Count", fontsize=14)
    ax.set_title(f"Exp 1: Random projector scan ({N_PROJ} projectors, d₁={D1})",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "exp1_gap_histogram.png")
    fig.savefig(OUT / "exp1_gap_histogram.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(eps_borns[valid], eps_combs[valid], alpha=0.3, s=15, color="steelblue",
               label="Random projectors")
    ax.scatter(eb_h, ec_h, color="crimson", s=150, marker="*", zorder=10,
               label="Hamming projector")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("ε at Born frequency", fontsize=14)
    ax.set_ylabel("ε at combinatorial peak", fontsize=14)
    ax.set_title("Exp 1: Born vs comb-peak recoherence", fontsize=14, fontweight="bold")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.set_aspect("equal")
    ax.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "exp1_scatter.png")
    fig.savefig(OUT / "exp1_scatter.pdf")
    plt.close(fig)

    print("  Saved: exp1_results.npz, exp1_gap_histogram.png, exp1_scatter.png", flush=True)
    return gap_ratios, masks, gap_h


# ===================================================================
# Experiment 2: Simulated annealing
# ===================================================================

def experiment_2(U):
    N_STEPS = 10_000
    N_RUNS = 10
    ANNEAL_SEEDS = list(range(3))  # 3 seeds for speed

    print(f"\n{'='*70}", flush=True)
    print(f"EXPERIMENT 2: Simulated annealing", flush=True)
    print(f"  {N_STEPS} steps x {N_RUNS} runs (max + min), 3 seeds/eval", flush=True)
    est = N_STEPS * N_RUNS * 2 * 3 * 0.018
    print(f"  Estimated runtime: {est/60:.0f} min", flush=True)
    print(f"{'='*70}", flush=True)

    def anneal(run_id, maximize, rng):
        in_set = set(rng.choice(D, size=D1, replace=False).tolist())
        out_set = set(range(D)) - in_set
        mask_1 = np.zeros(D, dtype=bool)
        mask_1[list(in_set)] = True
        cur_gap, _, _ = evaluate_gap(mask_1, U, ANNEAL_SEEDS)
        best_gap, best_set = cur_gap, sorted(in_set)
        history = [cur_gap]
        tag = "max" if maximize else "min"
        t0 = time.time()

        for step in range(N_STEPS):
            T = 0.5 * (0.999 ** step)
            add_idx = rng.choice(list(out_set))
            rem_idx = rng.choice(list(in_set))
            mask_1[add_idx] = True
            mask_1[rem_idx] = False
            new_gap, _, _ = evaluate_gap(mask_1, U, ANNEAL_SEEDS)

            if np.isnan(new_gap):
                mask_1[add_idx] = False; mask_1[rem_idx] = True
                history.append(cur_gap); continue

            delta = (new_gap - cur_gap) if maximize else (cur_gap - new_gap)
            if delta > 0 or rng.random() < np.exp(delta / T):
                in_set.add(add_idx); in_set.discard(rem_idx)
                out_set.add(rem_idx); out_set.discard(add_idx)
                cur_gap = new_gap
            else:
                mask_1[add_idx] = False; mask_1[rem_idx] = True

            if (maximize and cur_gap > best_gap) or (not maximize and cur_gap < best_gap):
                best_gap = cur_gap; best_set = sorted(in_set)
            history.append(cur_gap)

            if (step + 1) % 2000 == 0:
                print(f"    Run {run_id} ({tag}): {step+1}/{N_STEPS}, "
                      f"cur={cur_gap:.3f}, best={best_gap:.3f} ({time.time()-t0:.0f}s)", flush=True)

        return {"best_gap": best_gap, "best_mask": np.array(best_set),
                "history": np.array(history)}

    print("\n  --- Maximizing ---", flush=True)
    max_results = []
    for i in range(N_RUNS):
        r = anneal(i, True, np.random.default_rng(1000 + i))
        max_results.append(r)
        print(f"    Run {i} done: best = {r['best_gap']:.3f}", flush=True)

    print("\n  --- Minimizing ---", flush=True)
    min_results = []
    for i in range(N_RUNS):
        r = anneal(i, False, np.random.default_rng(2000 + i))
        min_results.append(r)
        print(f"    Run {i} done: best = {r['best_gap']:.3f}", flush=True)

    max_gaps = [r["best_gap"] for r in max_results]
    min_gaps = [r["best_gap"] for r in min_results]
    best_max = max(max_results, key=lambda r: r["best_gap"])
    best_min = min(min_results, key=lambda r: r["best_gap"])

    # Re-evaluate with 5 seeds
    seeds5 = list(range(5))
    mask_tmp = np.zeros(D, dtype=bool)
    mask_tmp[best_max["best_mask"]] = True
    gap_max5, _, _ = evaluate_gap(mask_tmp, U, seeds5)
    mask_tmp[:] = False
    mask_tmp[best_min["best_mask"]] = True
    gap_min5, _, _ = evaluate_gap(mask_tmp, U, seeds5)
    gap_h5, _, _ = evaluate_gap(hamming_mask(), U, seeds5)

    print(f"\n  Best maximizer (5 seeds): {gap_max5:.3f}", flush=True)
    print(f"  Best minimizer (5 seeds): {gap_min5:.3f}", flush=True)
    print(f"  Hamming reference (5 seeds): {gap_h5:.3f}", flush=True)

    np.savez(OUT / "exp2_results.npz",
             max_best_gaps=np.array(max_gaps), min_best_gaps=np.array(min_gaps),
             max_best_mask=best_max["best_mask"], min_best_mask=best_min["best_mask"],
             all_max_masks=np.array([r["best_mask"] for r in max_results], dtype=object),
             all_min_masks=np.array([r["best_mask"] for r in min_results], dtype=object),
             max_histories=np.array([r["history"] for r in max_results], dtype=object),
             min_histories=np.array([r["history"] for r in min_results], dtype=object))

    # Convergence plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    for r in max_results: ax1.plot(r["history"], alpha=0.4, lw=0.8)
    ax1.set_xlabel("Step"); ax1.set_ylabel("Gap ratio")
    ax1.set_title("Maximizing gap ratio", fontweight="bold")
    for r in min_results: ax2.plot(r["history"], alpha=0.4, lw=0.8)
    ax2.set_xlabel("Step"); ax2.set_ylabel("Gap ratio")
    ax2.set_title("Minimizing gap ratio", fontweight="bold")
    fig.suptitle(f"Exp 2: Simulated annealing ({N_RUNS} runs x {N_STEPS} steps)",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "exp2_convergence.png"); fig.savefig(OUT / "exp2_convergence.pdf")
    plt.close(fig)

    # Overlay on Exp 1 histogram
    try:
        exp1 = np.load(OUT / "exp1_results.npz", allow_pickle=True)
        exp1_gaps = exp1["gap_ratios"]
        valid = ~np.isnan(exp1_gaps)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.hist(exp1_gaps[valid], bins=60, color="steelblue", alpha=0.5,
                edgecolor="white", density=True, label="Exp 1: random")
        for i, g in enumerate(max_gaps):
            ax.axvline(g, color="darkgreen", alpha=0.4, lw=1.5,
                       label="Exp 2: maximizers" if i == 0 else None)
        for i, g in enumerate(min_gaps):
            ax.axvline(g, color="darkorange", alpha=0.4, lw=1.5,
                       label="Exp 2: minimizers" if i == 0 else None)
        ax.axvline(gap_h5, color="crimson", ls="--", lw=2.5,
                   label=f"Hamming ({gap_h5:.2f})")
        ax.set_xlabel("Gap ratio"); ax.set_ylabel("Density")
        ax.set_title("Optimized projectors vs random", fontweight="bold")
        ax.legend(fontsize=11)
        fig.tight_layout()
        fig.savefig(OUT / "exp2_comparison.png"); fig.savefig(OUT / "exp2_comparison.pdf")
        plt.close(fig)
    except FileNotFoundError:
        pass

    print("  Saved: exp2_results.npz, exp2_convergence.png, exp2_comparison.png", flush=True)
    return max_results, min_results


# ===================================================================
# Experiment 3: Analyze winners vs losers
# ===================================================================

def experiment_3(H, U):
    print(f"\n{'='*70}", flush=True)
    print(f"EXPERIMENT 3: Analyze winners vs losers", flush=True)
    print(f"{'='*70}", flush=True)

    # Load masks from experiments 1 and 2
    top_masks, bot_masks = [], []
    labels_top, labels_bot = [], []

    try:
        exp1 = np.load(OUT / "exp1_results.npz", allow_pickle=True)
        for m in exp1["top50_masks"]:
            top_masks.append(np.array(m))
        for m in exp1["bottom50_masks"]:
            bot_masks.append(np.array(m))
        labels_top += [f"exp1_top_{i}" for i in range(len(exp1["top50_masks"]))]
        labels_bot += [f"exp1_bot_{i}" for i in range(len(exp1["bottom50_masks"]))]
        print(f"  Loaded {len(exp1['top50_masks'])} top + {len(exp1['bottom50_masks'])} bottom from Exp 1", flush=True)
    except FileNotFoundError:
        print("  WARNING: exp1_results.npz not found, skipping Exp 1 projectors", flush=True)

    try:
        exp2 = np.load(OUT / "exp2_results.npz", allow_pickle=True)
        for m in exp2["all_max_masks"]:
            top_masks.append(np.array(m))
        for m in exp2["all_min_masks"]:
            bot_masks.append(np.array(m))
        labels_top += [f"exp2_max_{i}" for i in range(len(exp2["all_max_masks"]))]
        labels_bot += [f"exp2_min_{i}" for i in range(len(exp2["all_min_masks"]))]
        print(f"  Loaded {len(exp2['all_max_masks'])} max + {len(exp2['all_min_masks'])} min from Exp 2", flush=True)
    except FileNotFoundError:
        print("  WARNING: exp2_results.npz not found, skipping Exp 2 projectors", flush=True)

    if not top_masks and not bot_masks:
        print("  No data — run experiments 1 and/or 2 first.", flush=True)
        return

    # Add Hamming reference
    hm = hamming_mask()
    hamming_indices = np.where(hm)[0]

    # Energy eigenstates
    print("  Computing energy eigenstates...", flush=True)
    eigvals, eigvecs = np.linalg.eigh(H.real)  # H is real for hz=0

    def analyze_one(mask_indices):
        S = set(mask_indices.tolist())
        mask_bool = np.zeros(D, dtype=bool)
        mask_bool[list(S)] = True

        # 1. Energy alignment: for each eigenstate, overlap with S
        energy_overlap = np.array([
            np.sum(np.abs(eigvecs[list(S), n])**2) for n in range(D)
        ])

        # 2. Magnetization alignment
        mag_sectors = {}  # magnetization -> count in S
        for v in range(D):
            mag = M - 2 * bin(v).count("1")  # Σ Z_i eigenvalue
            if mag not in mag_sectors:
                mag_sectors[mag] = [0, 0]  # [in_S, total]
            mag_sectors[mag][1] += 1
            if v in S:
                mag_sectors[mag][0] += 1

        # Fraction of each sector captured
        mag_values = sorted(mag_sectors.keys())
        mag_fractions = np.array([mag_sectors[m][0] / mag_sectors[m][1] for m in mag_values])

        # Summary: how non-uniform is the magnetization distribution?
        # If S were random, each sector would have fraction D1/D ≈ 0.746
        mag_std = np.std(mag_fractions)

        # 3. Spatial locality: mutual information proxy
        # Split into left (qubits 0-4) and right (qubits 5-8)
        n_left = 5
        left_mask_bits = (1 << n_left) - 1
        # For states in S, compute distribution of (left_pattern, right_pattern)
        left_counts = {}
        right_counts = {}
        joint_counts = {}
        for v in S:
            l = v & left_mask_bits
            r = v >> n_left
            left_counts[l] = left_counts.get(l, 0) + 1
            right_counts[r] = right_counts.get(r, 0) + 1
            joint_counts[(l, r)] = joint_counts.get((l, r), 0) + 1
        n_total = len(S)
        # Mutual information
        mi = 0.0
        for (l, r), c in joint_counts.items():
            p_joint = c / n_total
            p_l = left_counts[l] / n_total
            p_r = right_counts[r] / n_total
            if p_joint > 0:
                mi += p_joint * np.log2(p_joint / (p_l * p_r))

        # 4. Commutator norm
        Pi = np.diag(mask_bool.astype(float))
        comm = Pi @ H - H @ Pi
        comm_norm = la_norm(comm, 'fro') / la_norm(H, 'fro')

        # 5. Initial-state overlap (average over 5 seeds)
        psi0_overlap = np.mean([
            np.sum(np.abs(make_initial_state(s)[list(S)])**2) for s in range(5)
        ])

        return {
            "energy_overlap": energy_overlap,
            "eigvals": eigvals,
            "mag_values": mag_values,
            "mag_fractions": mag_fractions,
            "mag_std": mag_std,
            "mutual_info": mi,
            "commutator_norm": comm_norm,
            "psi0_overlap": psi0_overlap,
        }

    print("  Analyzing top projectors...", flush=True)
    top_analysis = [analyze_one(m) for m in top_masks]
    print("  Analyzing bottom projectors...", flush=True)
    bot_analysis = [analyze_one(m) for m in bot_masks]
    hamming_analysis = analyze_one(hamming_indices)

    # --- Summary statistics ---
    def stats(analyses, key):
        vals = [a[key] for a in analyses]
        return np.mean(vals), np.std(vals)

    print(f"\n  {'Metric':<25} {'Top (winners)':<20} {'Bottom (losers)':<20} {'Hamming':<15}", flush=True)
    print(f"  {'-'*80}", flush=True)
    for key in ["mag_std", "mutual_info", "commutator_norm", "psi0_overlap"]:
        tm, ts = stats(top_analysis, key)
        bm, bs = stats(bot_analysis, key)
        hv = hamming_analysis[key]
        print(f"  {key:<25} {tm:.4f} ± {ts:.4f}   {bm:.4f} ± {bs:.4f}   {hv:.4f}", flush=True)

    # --- CSV output ---
    seeds5 = list(range(5))
    rows = []
    for label, mask_idx, analysis in (
        [(l, m, a) for l, m, a in zip(labels_top, top_masks, top_analysis)] +
        [(l, m, a) for l, m, a in zip(labels_bot, bot_masks, bot_analysis)] +
        [("hamming", hamming_indices, hamming_analysis)]
    ):
        mask_bool = np.zeros(D, dtype=bool)
        mask_bool[mask_idx] = True
        gap, eb, ec = evaluate_gap(mask_bool, U, seeds5)
        rows.append({
            "id": label, "d1": int(np.sum(mask_bool)),
            "gap_ratio": f"{gap:.4f}", "eps_born": f"{eb:.4f}", "eps_comb": f"{ec:.4f}",
            "energy_alignment_score": f"{analysis['mag_std']:.4f}",
            "magnetization_alignment_score": f"{analysis['mag_std']:.4f}",
            "commutator_norm": f"{analysis['commutator_norm']:.4f}",
            "mutual_info": f"{analysis['mutual_info']:.4f}",
            "psi0_overlap": f"{analysis['psi0_overlap']:.4f}",
        })

    with open(OUT / "exp3_projector_analysis.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    # --- Plots ---

    # Energy overlap spectrum: top vs bottom vs Hamming
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    # Average energy overlap for top
    top_eo = np.mean([a["energy_overlap"] for a in top_analysis], axis=0)
    bot_eo = np.mean([a["energy_overlap"] for a in bot_analysis], axis=0)
    ham_eo = hamming_analysis["energy_overlap"]

    for ax, data, title, color in [
        (axes[0], top_eo, "Top projectors (avg)", "darkgreen"),
        (axes[1], bot_eo, "Bottom projectors (avg)", "darkorange"),
        (axes[2], ham_eo, "Hamming projector", "crimson"),
    ]:
        ax.scatter(eigvals, data, s=8, alpha=0.6, color=color)
        ax.axhline(P1, color="gray", ls="--", alpha=0.5, label=f"Random baseline ({P1:.2f})")
        ax.set_xlabel("Energy eigenvalue", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
    axes[0].set_ylabel("Overlap of projector with eigenstate", fontsize=12)
    fig.suptitle("Exp 3: Energy alignment of projectors", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "exp3_energy_alignment.png"); fig.savefig(OUT / "exp3_energy_alignment.pdf")
    plt.close(fig)

    # Magnetization fractions
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, analyses, title, color in [
        (axes[0], top_analysis, "Top projectors", "darkgreen"),
        (axes[1], bot_analysis, "Bottom projectors", "darkorange"),
        (axes[2], [hamming_analysis], "Hamming projector", "crimson"),
    ]:
        for a in analyses[:10]:  # plot up to 10
            ax.plot(a["mag_values"], a["mag_fractions"], "o-", alpha=0.3, color=color, ms=4)
        ax.axhline(P1, color="gray", ls="--", alpha=0.5)
        ax.set_xlabel("Total magnetization", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Fraction of sector in S", fontsize=12)
    fig.suptitle("Exp 3: Magnetization alignment", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "exp3_magnetization.png"); fig.savefig(OUT / "exp3_magnetization.pdf")
    plt.close(fig)

    # Box plots for key metrics
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, key, ylabel in [
        (axes[0], "commutator_norm", "‖[Π,H]‖_F / ‖H‖_F"),
        (axes[1], "mutual_info", "MI(left, right) [bits]"),
        (axes[2], "mag_std", "σ(mag fractions)"),
        (axes[3], "psi0_overlap", "Tr(Π|ψ₀⟩⟨ψ₀|)"),
    ]:
        top_vals = [a[key] for a in top_analysis]
        bot_vals = [a[key] for a in bot_analysis]
        bp = ax.boxplot([top_vals, bot_vals], labels=["Top", "Bottom"],
                        patch_artist=True, widths=0.5)
        bp["boxes"][0].set_facecolor("lightgreen")
        bp["boxes"][1].set_facecolor("moccasin")
        ax.axhline(hamming_analysis[key], color="crimson", ls="--", lw=2,
                   label="Hamming")
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(fontsize=10)
    fig.suptitle("Exp 3: Winner vs loser metrics", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "exp3_metrics.png"); fig.savefig(OUT / "exp3_metrics.pdf")
    plt.close(fig)

    print("  Saved: exp3_projector_analysis.csv, exp3_energy_alignment.png, "
          "exp3_magnetization.png, exp3_metrics.png", flush=True)


# ===================================================================
# Experiment 4: Vary rank
# ===================================================================

def experiment_4(U):
    D1_VALUES = [50, 100, 150, 200, 256, 300, 350, 400, 462]
    N_PROJ = 500
    N_SEEDS = 5
    seeds = list(range(N_SEEDS))

    print(f"\n{'='*70}", flush=True)
    print(f"EXPERIMENT 4: Vary projector rank", flush=True)
    print(f"  Ranks: {D1_VALUES}", flush=True)
    print(f"  {N_PROJ} random projectors per rank, {N_SEEDS} seeds each", flush=True)
    est = len(D1_VALUES) * N_PROJ * N_SEEDS * 0.018
    print(f"  Estimated runtime: {est/60:.1f} min", flush=True)
    print(f"{'='*70}", flush=True)

    rng = np.random.default_rng(54321)
    all_idx = np.arange(D)

    rank_results = {}
    hamming_refs = {}

    for d1 in D1_VALUES:
        p1 = d1 / D
        born_n1 = round(L * p1)
        comb_n1 = L // 2
        print(f"\n  d1={d1} (p1={p1:.3f}), born_n1={born_n1}, comb_n1={comb_n1}", flush=True)

        gaps = []
        t0 = time.time()
        for i in range(N_PROJ):
            subset = rng.choice(all_idx, size=d1, replace=False)
            mask_1 = np.zeros(D, dtype=bool)
            mask_1[subset] = True

            eb_list, ec_list = [], []
            for s in seeds:
                psi0 = make_initial_state(s)
                r = frequency_class_analysis(psi0, U, mask_1, L, p1)
                eb, ec = r["epsilon"][born_n1], r["epsilon"][comb_n1]
                if not np.isnan(eb): eb_list.append(eb)
                if not np.isnan(ec): ec_list.append(ec)
            eps_born = np.mean(eb_list) if eb_list else 1e-10
            eps_comb = np.mean(ec_list) if ec_list else np.nan
            gap = (eps_comb / eps_born) if eps_born > 0.001 else np.nan
            gaps.append(gap)

            if (i + 1) % 100 == 0:
                el = time.time() - t0
                print(f"    {i+1}/{N_PROJ} ({el:.0f}s)", flush=True)

        gaps = np.array(gaps)
        valid = ~np.isnan(gaps)
        rank_results[d1] = gaps
        print(f"    mean gap={np.nanmean(gaps):.3f}, median={np.nanmedian(gaps):.3f}", flush=True)

        # Hamming reference at closest matching threshold
        # Find threshold that gives d1 closest to target
        best_thr, best_d1h = 0, 0
        for thr in range(M + 1):
            d1h = sum(comb(M, k) for k in range(thr, M + 1))
            if abs(d1h - d1) < abs(best_d1h - d1):
                best_thr, best_d1h = thr, d1h
        if best_d1h == d1:
            hm = np.array([bin(v).count("1") >= best_thr for v in range(D)])
            p1h = best_d1h / D
            born_h = round(L * p1h)
            eb_list, ec_list = [], []
            for s in seeds:
                psi0 = make_initial_state(s)
                r = frequency_class_analysis(psi0, U, hm, L, p1h)
                eb, ec = r["epsilon"][born_h], r["epsilon"][comb_n1]
                if not np.isnan(eb): eb_list.append(eb)
                if not np.isnan(ec): ec_list.append(ec)
            eb_m = np.mean(eb_list) if eb_list else 1e-10
            ec_m = np.mean(ec_list) if ec_list else np.nan
            hamming_refs[d1] = (ec_m / eb_m) if eb_m > 0.001 else np.nan
            print(f"    Hamming (thr={best_thr}, d1={best_d1h}): gap={hamming_refs[d1]:.3f}", flush=True)

    # Save
    np.savez(OUT / "exp4_results.npz",
             d1_values=np.array(D1_VALUES),
             **{f"gaps_d1_{d1}": rank_results[d1] for d1 in D1_VALUES})

    # Plot: mean gap vs p1
    fig, ax = plt.subplots(figsize=(12, 6))
    p1_vals = [d1 / D for d1 in D1_VALUES]
    mean_gaps = [np.nanmean(rank_results[d1]) for d1 in D1_VALUES]
    std_gaps = [np.nanstd(rank_results[d1]) for d1 in D1_VALUES]
    med_gaps = [np.nanmedian(rank_results[d1]) for d1 in D1_VALUES]

    ax.errorbar(p1_vals, mean_gaps, yerr=std_gaps, fmt="o-", color="steelblue",
                lw=2, ms=8, capsize=5, label="Random projectors (mean ± σ)")
    ax.plot(p1_vals, med_gaps, "s--", color="navy", ms=6, alpha=0.6,
            label="Random projectors (median)")

    # Hamming references
    for d1, gap in hamming_refs.items():
        ax.scatter(d1 / D, gap, color="crimson", s=120, marker="*", zorder=10)
    ax.scatter([], [], color="crimson", s=120, marker="*", label="Hamming-weight projectors")

    ax.axhline(1.0, color="gray", ls=":", alpha=0.5, label="No filtering")
    ax.set_xlabel("p₁ = d₁/D", fontsize=14)
    ax.set_ylabel("Mean gap ratio", fontsize=14)
    ax.set_title(f"Exp 4: Gap ratio vs projector rank (D={D}, L={L})",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "exp4_rank_scan.png"); fig.savefig(OUT / "exp4_rank_scan.pdf")
    plt.close(fig)

    # Violin plot
    fig, ax = plt.subplots(figsize=(14, 6))
    positions = list(range(len(D1_VALUES)))
    data = [rank_results[d1][~np.isnan(rank_results[d1])] for d1 in D1_VALUES]
    vp = ax.violinplot(data, positions=positions, showmedians=True, showextrema=False)
    for body in vp["bodies"]:
        body.set_facecolor("steelblue"); body.set_alpha(0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{d1}\n({d1/D:.2f})" for d1 in D1_VALUES])
    ax.set_xlabel("d₁ (p₁)", fontsize=14)
    ax.set_ylabel("Gap ratio", fontsize=14)
    ax.set_title(f"Exp 4: Gap distribution vs rank", fontsize=14, fontweight="bold")
    ax.axhline(1.0, color="gray", ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "exp4_violin.png"); fig.savefig(OUT / "exp4_violin.pdf")
    plt.close(fig)

    print("  Saved: exp4_results.npz, exp4_rank_scan.png, exp4_violin.png", flush=True)


# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Projector search experiments")
    parser.add_argument("--exp", nargs="+", type=int, default=[1, 2, 3, 4],
                        help="Which experiments to run (default: all)")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    exps = set(args.exp)
    t_total = time.time()

    H, U = setup_physics()

    if 1 in exps:
        experiment_1(U)
    if 2 in exps:
        experiment_2(U)
    if 3 in exps:
        experiment_3(H, U)
    if 4 in exps:
        experiment_4(U)

    print(f"\nTotal time: {(time.time() - t_total)/60:.1f} min", flush=True)
    print(f"All results in {OUT}/", flush=True)
