#!/usr/bin/env python3
"""Experiment 2: Simulated annealing optimization of projectors.

Starting from random projectors (rank 382), swap one basis state in/out at each step.
Run 10,000 steps x 10 independent runs, optimizing for both max and min gap ratio.
Use 3 seeds per evaluation for speed.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import time
from pathlib import Path
from scipy.linalg import expm

from recohere.ising_direct import (
    build_ising_hamiltonian,
    frequency_class_analysis,
)

OUT = Path("results/projector_search")

plt.rcParams.update({
    "font.size": 13, "axes.grid": True, "grid.alpha": 0.25,
    "savefig.dpi": 200, "savefig.bbox": "tight", "font.family": "serif",
})

# Physics parameters
M = 9
D = 2**M  # 512
L = 15
DT = 1.2
J = 1.0
HX = 0.9045

# Experiment parameters
D1 = 382
P1 = D1 / D
N_SEEDS_EVAL = 3       # fewer seeds for speed during annealing
N_STEPS = 10_000
N_RUNS = 10
BORN_N1 = round(L * P1)  # 11
COMB_N1 = L // 2         # 7


def make_initial_state(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    psi0 = rng.standard_normal(D) + 1j * rng.standard_normal(D)
    return (psi0 / np.linalg.norm(psi0)).astype(np.complex128)


# Pre-compute initial states
INITIAL_STATES = [make_initial_state(s) for s in range(N_SEEDS_EVAL)]


def evaluate_gap(mask_1: np.ndarray, U: np.ndarray) -> float:
    """Compute mean gap ratio over seeds."""
    eps_born_list = []
    eps_comb_list = []
    for psi0 in INITIAL_STATES:
        result = frequency_class_analysis(psi0, U, mask_1, L, P1)
        eb = result.epsilon[BORN_N1]
        ec = result.epsilon[COMB_N1]
        if not np.isnan(eb):
            eps_born_list.append(eb)
        if not np.isnan(ec):
            eps_comb_list.append(ec)
    eps_born = np.mean(eps_born_list) if eps_born_list else 1e-10
    eps_comb = np.mean(eps_comb_list) if eps_comb_list else np.nan
    if eps_born < 0.001:
        return np.nan
    return eps_comb / eps_born


def anneal_one_run(
    run_id: int, U: np.ndarray, maximize: bool, rng: np.random.Generator,
) -> dict:
    """Single simulated annealing run."""
    # Random initial projector
    in_set = set(rng.choice(D, size=D1, replace=False).tolist())
    out_set = set(range(D)) - in_set

    mask_1 = np.zeros(D, dtype=bool)
    mask_1[list(in_set)] = True
    current_gap = evaluate_gap(mask_1, U)

    best_gap = current_gap
    best_mask_indices = sorted(in_set)
    history = [current_gap]

    direction = "max" if maximize else "min"
    t0 = time.time()

    for step in range(N_STEPS):
        # Temperature schedule: exponential decay
        T = 0.5 * (0.999 ** step)

        # Propose: swap one in for one out
        add_idx = rng.choice(list(out_set))
        remove_idx = rng.choice(list(in_set))

        # Apply swap
        mask_1[add_idx] = True
        mask_1[remove_idx] = False

        new_gap = evaluate_gap(mask_1, U)

        if np.isnan(new_gap):
            # Reject
            mask_1[add_idx] = False
            mask_1[remove_idx] = True
            history.append(current_gap)
            continue

        # Accept/reject
        if maximize:
            delta = new_gap - current_gap
        else:
            delta = current_gap - new_gap  # want to decrease

        if delta > 0 or rng.random() < np.exp(delta / T):
            # Accept
            in_set.add(add_idx)
            in_set.discard(remove_idx)
            out_set.add(remove_idx)
            out_set.discard(add_idx)
            current_gap = new_gap
        else:
            # Reject
            mask_1[add_idx] = False
            mask_1[remove_idx] = True

        if (maximize and current_gap > best_gap) or (not maximize and current_gap < best_gap):
            best_gap = current_gap
            best_mask_indices = sorted(in_set)

        history.append(current_gap)

        if (step + 1) % 2000 == 0:
            elapsed = time.time() - t0
            print(f"    Run {run_id} ({direction}): step {step+1}/{N_STEPS}, "
                  f"current={current_gap:.3f}, best={best_gap:.3f} ({elapsed:.0f}s)")

    return {
        "run_id": run_id,
        "maximize": maximize,
        "best_gap": best_gap,
        "best_mask_indices": np.array(best_mask_indices),
        "final_gap": current_gap,
        "history": np.array(history),
    }


def run_experiment_2():
    print("Experiment 2: Simulated annealing optimization")
    print(f"  D={D}, L={L}, d1={D1}, {N_STEPS} steps x {N_RUNS} runs, {N_SEEDS_EVAL} seeds/eval")

    # Build unitary
    print("  Building Hamiltonian and unitary...")
    H = build_ising_hamiltonian(M, J, HX)
    U = expm(-1j * H * DT)

    # Estimate: ~0.05s per step (3 seeds), 10k steps x 20 runs
    est_per_step = N_SEEDS_EVAL * 0.018
    est_total = N_STEPS * N_RUNS * 2 * est_per_step
    print(f"  Estimated runtime: {est_total/60:.0f} min")

    # Run maximization
    print("\n  === Maximizing gap ratio ===")
    max_results = []
    for i in range(N_RUNS):
        rng = np.random.default_rng(1000 + i)
        r = anneal_one_run(i, U, maximize=True, rng=rng)
        max_results.append(r)
        print(f"    Run {i} done: best gap = {r['best_gap']:.3f}")

    # Run minimization
    print("\n  === Minimizing gap ratio ===")
    min_results = []
    for i in range(N_RUNS):
        rng = np.random.default_rng(2000 + i)
        r = anneal_one_run(i, U, maximize=False, rng=rng)
        min_results.append(r)
        print(f"    Run {i} done: best gap = {r['best_gap']:.3f}")

    # Summary
    max_gaps = [r["best_gap"] for r in max_results]
    min_gaps = [r["best_gap"] for r in min_results]
    print(f"\n  Maximizers: best={max(max_gaps):.3f}, mean={np.mean(max_gaps):.3f}")
    print(f"  Minimizers: best={min(min_gaps):.3f}, mean={np.mean(min_gaps):.3f}")

    # Get overall best/worst
    best_max = max(max_results, key=lambda r: r["best_gap"])
    best_min = min(min_results, key=lambda r: r["best_gap"])

    # Save
    np.savez(
        OUT / "experiment2_results.npz",
        max_best_gaps=np.array(max_gaps),
        min_best_gaps=np.array(min_gaps),
        max_best_mask=best_max["best_mask_indices"],
        min_best_mask=best_min["best_mask_indices"],
        max_histories=np.array([r["history"] for r in max_results], dtype=object),
        min_histories=np.array([r["history"] for r in min_results], dtype=object),
        all_max_masks=np.array([r["best_mask_indices"] for r in max_results], dtype=object),
        all_min_masks=np.array([r["best_mask_indices"] for r in min_results], dtype=object),
    )

    # Re-evaluate best/worst with 5 seeds for accurate numbers
    print("\n  Re-evaluating best/worst with 5 seeds...")
    global INITIAL_STATES
    INITIAL_STATES_5 = [make_initial_state(s) for s in range(5)]
    old_states = INITIAL_STATES
    INITIAL_STATES = INITIAL_STATES_5

    mask_max = np.zeros(D, dtype=bool)
    mask_max[best_max["best_mask_indices"]] = True
    gap_max_5 = evaluate_gap(mask_max, U)

    mask_min = np.zeros(D, dtype=bool)
    mask_min[best_min["best_mask_indices"]] = True
    gap_min_5 = evaluate_gap(mask_min, U)

    # Hamming reference
    from recohere.ising_direct import _build_mask
    hamming_mask = _build_mask(M, "hamming", 4)
    gap_hamming = evaluate_gap(hamming_mask, U)

    INITIAL_STATES = old_states

    print(f"  Best maximizer (5 seeds): gap = {gap_max_5:.3f}")
    print(f"  Best minimizer (5 seeds): gap = {gap_min_5:.3f}")
    print(f"  Hamming reference (5 seeds): gap = {gap_hamming:.3f}")

    # Plots
    plot_convergence(max_results, min_results)
    plot_comparison(max_gaps, min_gaps, gap_hamming)

    return max_results, min_results


def plot_convergence(max_results, min_results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for r in max_results:
        ax1.plot(r["history"], alpha=0.4, lw=0.8)
    ax1.set_xlabel("Step", fontsize=14)
    ax1.set_ylabel("Gap ratio", fontsize=14)
    ax1.set_title("Maximizing gap ratio", fontsize=14, fontweight="bold")

    for r in min_results:
        ax2.plot(r["history"], alpha=0.4, lw=0.8)
    ax2.set_xlabel("Step", fontsize=14)
    ax2.set_ylabel("Gap ratio", fontsize=14)
    ax2.set_title("Minimizing gap ratio", fontsize=14, fontweight="bold")

    fig.suptitle(f"Experiment 2: Simulated annealing ({N_RUNS} runs x {N_STEPS} steps)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "exp2_convergence.png")
    fig.savefig(OUT / "exp2_convergence.pdf")
    plt.close(fig)
    print("  -> exp2_convergence.png")


def plot_comparison(max_gaps, min_gaps, hamming_gap):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Load experiment 1 data for context
    exp1 = np.load(OUT / "experiment1_results.npz", allow_pickle=True)
    exp1_gaps = exp1["gap_ratios"]
    valid = ~np.isnan(exp1_gaps)

    ax.hist(exp1_gaps[valid], bins=60, color="steelblue", alpha=0.5,
            edgecolor="white", label="Exp 1: random projectors", density=True)

    for i, g in enumerate(max_gaps):
        ax.axvline(g, color="darkgreen", alpha=0.4, lw=1.5,
                   label="Exp 2: maximizers" if i == 0 else None)
    for i, g in enumerate(min_gaps):
        ax.axvline(g, color="darkorange", alpha=0.4, lw=1.5,
                   label="Exp 2: minimizers" if i == 0 else None)
    ax.axvline(hamming_gap, color="crimson", ls="--", lw=2.5,
               label=f"Hamming projector ({hamming_gap:.2f})")

    ax.set_xlabel("Gap ratio", fontsize=14)
    ax.set_ylabel("Density", fontsize=14)
    ax.set_title("Optimized projectors vs random distribution",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "exp2_comparison.png")
    fig.savefig(OUT / "exp2_comparison.pdf")
    plt.close(fig)
    print("  -> exp2_comparison.png")


if __name__ == "__main__":
    run_experiment_2()
