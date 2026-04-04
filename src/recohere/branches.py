"""Branch-level analysis: track ALL 2^L individual history states.

Instead of aggregating into frequency classes |psi(n1)>, this module tracks
every individual branch |psi(x)> = Pi_{x_L} U ... Pi_{x_1} U |psi_0>.

This lets us directly observe which branches remain decoherent (orthogonal)
as L grows and the Hilbert space "runs out of room" for all 2^L worlds.

Multi-scale analysis: coarse-grain pairs of scale-1 outcomes into a ternary
alphabet (A=00, B=11, C=01/10) to test whether Born filtering is self-similar.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from recohere.analysis import CoherenceResult, compute_epsilon
from recohere.ising_direct import IsingDirectParams, build_ising_setup


@dataclass(frozen=True)
class BranchResult:
    """Results of branch-level decoherence analysis."""

    histories: list[tuple[int, ...]]  # each is an L-tuple of 0/1
    branch_states: np.ndarray  # (N, D) complex, unnormalized
    n1: np.ndarray  # (N,) int, number of "1" outcomes per branch
    weights: np.ndarray  # (N,) float, q(x) = ||psi(x)||^2
    gram_normalized: np.ndarray | None  # (N, N) complex or None for large L
    epsilon: np.ndarray  # (N,) float, max off-diagonal |G_norm| per branch
    p1: float


# Scale-2 ternary alphabet: A=00, B=11, C=01 or 10
SCALE2_MAP = {(0, 0): "A", (1, 1): "B", (0, 1): "C", (1, 0): "C"}


@dataclass(frozen=True)
class Scale2Result:
    """Results of scale-2 (coarse-grained) branch analysis."""

    scale2_histories: list[tuple[str, ...]]  # e.g. ("B", "C", "A", ...)
    scale2_states: np.ndarray  # (N2, D) complex, unnormalized
    n_abc: np.ndarray  # (N2, 3) int, counts of A/B/C per branch
    weights: np.ndarray  # (N2,) float
    gram_normalized: np.ndarray  # (N2, N2) complex
    epsilon: np.ndarray  # (N2,) float
    p_born: tuple[float, float, float]  # (pA, pB, pC) Born probabilities
    # Nesting: for each scale-2 branch, which scale-1 branches compose it
    scale1_members: dict[tuple[str, ...], list[int]]  # scale2_hist -> list of scale1 indices


def _normalize_states(states: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Normalize state vectors, handling near-zero weights safely."""
    norms = np.sqrt(weights)
    safe_norms = np.where(norms > 1e-15, norms, 1.0)
    return states / safe_norms[:, None]


def _normalized_gram_epsilon(states_norm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute normalized Gram matrix and per-row max off-diagonal (epsilon)."""
    gram_norm = states_norm @ states_norm.conj().T
    abs_gram = np.abs(gram_norm)
    np.fill_diagonal(abs_gram, 0.0)
    epsilon = np.max(abs_gram, axis=1)
    return gram_norm, epsilon


def _chunked_epsilon(states_norm: np.ndarray, chunk_size: int = 512) -> np.ndarray:
    """Compute epsilon without materializing the full Gram matrix.

    Processes rows in chunks: for each chunk, computes overlap with ALL branches,
    then takes the max off-diagonal per row. Memory: O(chunk_size * N) instead of O(N^2).
    """
    N = states_norm.shape[0]
    epsilon = np.zeros(N)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        overlap = np.abs(states_norm[start:end] @ states_norm.conj().T)
        chunk_indices = np.arange(end - start)
        overlap[chunk_indices, chunk_indices + start] = 0.0
        epsilon[start:end] = np.max(overlap, axis=1)

    return epsilon


def simulate_branches(params: IsingDirectParams, gram: bool | None = None) -> BranchResult:
    """Track all individual branches through the Ising evolution.

    For L<=13, stores the full normalized Gram matrix.
    For L>13 (up to 16), uses chunked epsilon to avoid O(N^2) memory.
    Set gram=True/False to override.
    """
    D = 2**params.m
    L = params.L

    if L > 16:
        raise NotImplementedError(
            f"L={L} gives 2^{L}={2**L} branches — too many. Use L<=16."
        )

    store_gram = gram if gram is not None else (L <= 13)

    psi0, U, mask_1 = build_ising_setup(params)

    # Branch tracking: dict from history tuple -> state vector
    current: dict[tuple[int, ...], np.ndarray] = {(): psi0.copy()}

    for step in range(L):
        next_branches: dict[tuple[int, ...], np.ndarray] = {}

        for history, state in current.items():
            evolved = U @ state

            state_0 = np.where(~mask_1, evolved, 0.0)
            state_1 = np.where(mask_1, evolved, 0.0)

            for outcome, projected in [(0, state_0), (1, state_1)]:
                if np.vdot(projected, projected).real < 1e-30:
                    continue
                next_branches[history + (outcome,)] = projected

        current = next_branches

    # Collect into arrays
    histories = list(current.keys())
    N = len(histories)
    states = np.array([current[h] for h in histories])
    n1 = np.array([sum(h) for h in histories])
    weights = np.sum(np.abs(states) ** 2, axis=1)

    states_norm = _normalize_states(states, weights)

    if store_gram:
        gram_norm, epsilon = _normalized_gram_epsilon(states_norm)
    else:
        gram_norm = None
        epsilon = _chunked_epsilon(states_norm)

    return BranchResult(
        histories=histories,
        branch_states=states,
        n1=n1,
        weights=weights,
        gram_normalized=gram_norm,
        epsilon=epsilon,
        p1=params.p1,
    )


def _history_to_scale2(hist: tuple[int, ...], L_used: int | None = None) -> tuple[str, ...]:
    """Map a scale-1 binary history to a scale-2 ternary history.

    Uses the first L_used outcomes (must be even). Defaults to all if even.
    """
    n = L_used if L_used is not None else len(hist)
    return tuple(SCALE2_MAP[(hist[i], hist[i + 1])] for i in range(0, n, 2))


def analyze_scale2(branch_result: BranchResult) -> Scale2Result:
    """Coarse-grain scale-1 branches into scale-2 ternary histories.

    Pairs consecutive scale-1 outcomes: A=00, B=11, C=01/10.
    Aggregates branch states sharing the same scale-2 history.
    For odd L, the last scale-1 outcome is ignored (uses first L-1 outcomes).
    """
    L = len(branch_result.histories[0])
    L_used = L if L % 2 == 0 else L - 1  # drop last outcome if odd

    p1 = branch_result.p1
    p0 = 1.0 - p1
    p_born = (p0 * p0, p1 * p1, 2 * p0 * p1)  # (pA, pB, pC)

    # Group scale-1 branches by their scale-2 history
    scale2_groups: dict[tuple[str, ...], list[int]] = {}
    for idx, hist in enumerate(branch_result.histories):
        s2 = _history_to_scale2(hist, L_used)
        scale2_groups.setdefault(s2, []).append(idx)

    # Aggregate states
    scale2_histories = sorted(scale2_groups.keys())
    D = branch_result.branch_states.shape[1]
    N2 = len(scale2_histories)
    states = np.zeros((N2, D), dtype=complex)

    for i, s2h in enumerate(scale2_histories):
        indices = np.array(scale2_groups[s2h])
        states[i] = np.sum(branch_result.branch_states[indices], axis=0)

    n_abc = np.array([[h.count("A"), h.count("B"), h.count("C")]
                       for h in scale2_histories])

    weights = np.sum(np.abs(states) ** 2, axis=1)
    states_norm = _normalize_states(states, weights)
    gram_norm, epsilon = _normalized_gram_epsilon(states_norm)

    return Scale2Result(
        scale2_histories=scale2_histories,
        scale2_states=states,
        n_abc=n_abc,
        weights=weights,
        gram_normalized=gram_norm,
        epsilon=epsilon,
        p_born=p_born,
        scale1_members=scale2_groups,
    )


@dataclass(frozen=True)
class Scale2FreqClassResult:
    """Frequency-class analysis at scale 2.

    Aggregates all scale-2 branches with the same (nA, nB, nC) counts
    into a single frequency-class state, then computes the Gram matrix
    and epsilon — the direct analogue of scale-1 frequency-class analysis.
    """

    freq_classes: list[tuple[int, int, int]]  # (nA, nB, nC) for each class
    states: np.ndarray  # (N_classes, D) complex, unnormalized
    weights: np.ndarray  # (N_classes,) float
    gram_matrix: np.ndarray  # (N_classes, N_classes) complex (unnormalized)
    epsilon: np.ndarray  # (N_classes,) float
    p_born: tuple[float, float, float]  # (pA, pB, pC)
    L2: int  # number of scale-2 steps


def analyze_scale2_freq_classes(s2: Scale2Result) -> Scale2FreqClassResult:
    """Aggregate scale-2 branches into frequency classes (nA, nB, nC).

    This is the scale-2 analogue of the scale-1 frequency-class decomposition:
    all branches with the same (nA, nB, nC) are coherently summed.
    """
    D = s2.scale2_states.shape[1]
    L2 = s2.n_abc[0].sum()  # all rows sum to L2

    # Group by (nA, nB, nC)
    freq_groups: dict[tuple[int, int, int], list[int]] = {}
    for i, abc in enumerate(s2.n_abc):
        key = (int(abc[0]), int(abc[1]), int(abc[2]))
        freq_groups.setdefault(key, []).append(i)

    freq_classes = sorted(freq_groups.keys())
    N_classes = len(freq_classes)
    states = np.zeros((N_classes, D), dtype=complex)

    for i, fc in enumerate(freq_classes):
        indices = np.array(freq_groups[fc])
        states[i] = np.sum(s2.scale2_states[indices], axis=0)

    gram = states @ states.conj().T
    weights = gram.diagonal().real
    epsilon = compute_epsilon(gram)

    return Scale2FreqClassResult(
        freq_classes=freq_classes,
        states=states,
        weights=weights,
        gram_matrix=gram,
        epsilon=epsilon,
        p_born=s2.p_born,
        L2=L2,
    )


def simulate_scale2_freq_classes(params: IsingDirectParams, L2: int) -> Scale2FreqClassResult:
    """Directly simulate scale-2 frequency classes without tracking 2^L branches.

    At each super-step (pair of scale-1 steps), each (nA, nB, nC) state branches
    into three outcomes: A=00→(nA+1), B=11→(nB+1), C=01or10→(nC+1).
    The C branch coherently sums the 01 and 10 paths.

    Cost: O(L2^3 * D^2) — the number of frequency classes grows as L2^2/2,
    each needing 3 matrix-vector multiplies per super-step.
    For L2=50, D=512: ~1326 classes, trivially cheap.
    """
    D = 2**params.m
    psi0, U, mask_1 = build_ising_setup(params)

    # Track frequency classes: (nA, nB, nC) -> state vector
    current: dict[tuple[int, int, int], np.ndarray] = {(0, 0, 0): psi0.copy()}

    for step in range(L2):
        next_states: dict[tuple[int, int, int], np.ndarray] = {}

        for (nA, nB, nC), state in current.items():
            # First scale-1 step: evolve and project
            evolved1 = U @ state
            s0 = np.where(~mask_1, evolved1, 0.0)  # outcome 0
            s1 = np.where(mask_1, evolved1, 0.0)    # outcome 1

            # Second scale-1 step from outcome 0
            evolved_0x = U @ s0
            s00 = np.where(~mask_1, evolved_0x, 0.0)  # A = 00
            s01 = np.where(mask_1, evolved_0x, 0.0)    # C = 01

            # Second scale-1 step from outcome 1
            evolved_1x = U @ s1
            s10 = np.where(~mask_1, evolved_1x, 0.0)  # C = 10
            s11 = np.where(mask_1, evolved_1x, 0.0)    # B = 11

            # A = 00
            for key, proj in [
                ((nA + 1, nB, nC), s00),
                ((nA, nB + 1, nC), s11),
                ((nA, nB, nC + 1), s01),  # C gets both 01 and 10
            ]:
                if np.vdot(proj, proj).real < 1e-30:
                    continue
                if key not in next_states:
                    next_states[key] = np.zeros(D, dtype=complex)
                next_states[key] += proj

            # C also gets s10
            key_C = (nA, nB, nC + 1)
            if np.vdot(s10, s10).real >= 1e-30:
                if key_C not in next_states:
                    next_states[key_C] = np.zeros(D, dtype=complex)
                next_states[key_C] += s10

        current = next_states

    # Collect into arrays
    freq_classes = sorted(current.keys())
    N = len(freq_classes)
    states = np.array([current[fc] for fc in freq_classes])

    gram = states @ states.conj().T
    weights = gram.diagonal().real
    epsilon = compute_epsilon(gram)

    # Compute actual Born probabilities from single super-step weights
    p1 = params.p1
    p0 = 1.0 - p1
    p_born = (p0 * p0, p1 * p1, 2 * p0 * p1)

    return Scale2FreqClassResult(
        freq_classes=freq_classes,
        states=states,
        weights=weights,
        gram_matrix=gram,
        epsilon=epsilon,
        p_born=p_born,
        L2=L2,
    )


def simulate_scale2_binary(params: IsingDirectParams, L2: int) -> CoherenceResult:
    """Scale-2 frequency-class simulation with binary alphabet.

    A = "same" (00 or 11), B = "different" (01 or 10).
    Tracks n_A (number of "same" outcomes) — only L2+1 frequency classes,
    same scaling as scale-1. Can push to large L2 without hitting D.

    Returns a CoherenceResult identical in structure to scale-1 analysis,
    with born_frequency = L2 * p_A.
    """
    D = 2**params.m
    psi0, U, mask_1 = build_ising_setup(params)

    # Track frequency classes: n_same -> state vector
    current: dict[int, np.ndarray] = {0: psi0.copy()}

    for step in range(L2):
        next_states: dict[int, np.ndarray] = {}

        for n_same, state in current.items():
            # First scale-1 step
            evolved1 = U @ state
            s0 = np.where(~mask_1, evolved1, 0.0)
            s1 = np.where(mask_1, evolved1, 0.0)

            # Second scale-1 step
            evolved_0x = U @ s0
            s00 = np.where(~mask_1, evolved_0x, 0.0)  # same (A)
            s01 = np.where(mask_1, evolved_0x, 0.0)    # diff (B)

            evolved_1x = U @ s1
            s10 = np.where(~mask_1, evolved_1x, 0.0)  # diff (B)
            s11 = np.where(mask_1, evolved_1x, 0.0)    # same (A)

            # A = "same" → n_same + 1 (coherent sum of 00 and 11)
            key_A = n_same + 1
            for proj in [s00, s11]:
                if np.vdot(proj, proj).real < 1e-30:
                    continue
                if key_A not in next_states:
                    next_states[key_A] = np.zeros(D, dtype=complex)
                next_states[key_A] += proj

            # B = "different" → n_same unchanged (coherent sum of 01 and 10)
            key_B = n_same
            for proj in [s01, s10]:
                if np.vdot(proj, proj).real < 1e-30:
                    continue
                if key_B not in next_states:
                    next_states[key_B] = np.zeros(D, dtype=complex)
                next_states[key_B] += proj

        current = next_states

    # Collect into frequency-class states
    freq_states = [np.zeros(D, dtype=complex) for _ in range(L2 + 1)]
    for n_same, state in current.items():
        if 0 <= n_same <= L2:
            freq_states[n_same] = state

    gram = np.array(freq_states) @ np.array(freq_states).conj().T
    weights = gram.diagonal().real
    epsilon = compute_epsilon(gram)

    p1 = params.p1
    p0 = 1.0 - p1
    p_same = p0 * p0 + p1 * p1  # independent assumption

    return CoherenceResult(
        frequency_classes=np.arange(L2 + 1),
        gram_matrix=gram,
        epsilon=epsilon,
        weights=weights,
        born_frequency=L2 * p_same,
    )
