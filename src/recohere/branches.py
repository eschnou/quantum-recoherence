"""Branch-level analysis: track ALL 2^L individual history states.

Instead of aggregating into frequency classes |psi(n1)>, this module tracks
every individual branch |psi(x)> = Pi_{x_L} U ... Pi_{x_1} U |psi_0>.

This lets us directly observe which branches remain decoherent (orthogonal)
as L grows and the Hilbert space "runs out of room" for all 2^L worlds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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


