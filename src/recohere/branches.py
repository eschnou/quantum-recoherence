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
    gram_normalized: np.ndarray  # (N, N) complex, diagonal = 1
    epsilon: np.ndarray  # (N,) float, max off-diagonal |G_norm| per branch
    p1: float


def simulate_branches(params: IsingDirectParams) -> BranchResult:
    """Track all individual branches through the Ising evolution."""
    D = 2**params.m
    L = params.L

    if L > 13:
        raise NotImplementedError(
            f"L={L} gives 2^{L}={2**L} branches — too many. Use L<=13."
        )

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

    # Normalized Gram matrix (diagonal = 1)
    norms = np.sqrt(weights)
    safe_norms = np.where(norms > 1e-15, norms, 1.0)
    states_norm = states / safe_norms[:, None]
    gram_norm = states_norm @ states_norm.conj().T

    # Epsilon: max off-diagonal |G_norm| per branch
    abs_gram = np.abs(gram_norm)
    np.fill_diagonal(abs_gram, 0.0)
    epsilon = np.max(abs_gram, axis=1)

    return BranchResult(
        histories=histories,
        branch_states=states,
        n1=n1,
        weights=weights,
        gram_normalized=gram_norm,
        epsilon=epsilon,
        p1=params.p1,
    )
