"""Spatial multi-scale Born filtering.

Split 8 qubits into two groups of 4 (left: 0-3, right: 4-7).
At each step, apply independent Hamming-weight projectors to each group.
This gives nested scales of proper orthogonal projectors:

  Scale 1a/1b: each group independently (HW >= threshold)
  Scale 2:     "at least one group fires" (OR projector)

All are genuine projector decompositions of the same Hilbert space.
Born filtering operates independently at each level.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from recohere.analysis import CoherenceResult, compute_epsilon
from recohere.ising_direct import IsingDirectParams, build_ising_setup


@dataclass(frozen=True)
class SpatialParams:
    m_left: int = 4
    m_right: int = 4
    hamming_threshold: int = 2  # HW >= threshold on each group
    L: int = 15
    J: float = 1.0
    hx: float = 0.9045
    hz: float = 0.0
    dt: float = 1.2
    seed: int = 42

    @property
    def m(self) -> int:
        return self.m_left + self.m_right

    @property
    def D(self) -> int:
        return 2**self.m

    @property
    def d1_left(self) -> int:
        from math import comb
        return sum(comb(self.m_left, k) for k in range(self.hamming_threshold, self.m_left + 1))

    @property
    def d1_right(self) -> int:
        from math import comb
        return sum(comb(self.m_right, k) for k in range(self.hamming_threshold, self.m_right + 1))

    @property
    def p1_left(self) -> float:
        return self.d1_left / (2**self.m_left)

    @property
    def p1_right(self) -> float:
        return self.d1_right / (2**self.m_right)

    @property
    def d1_both(self) -> int:
        return self.d1_left * self.d1_right

    @property
    def p1_both(self) -> float:
        return self.d1_both / self.D

    @property
    def d1_either(self) -> int:
        """Rank of the OR projector: at least one group fires."""
        d0_left = 2**self.m_left - self.d1_left
        d0_right = 2**self.m_right - self.d1_right
        return self.D - d0_left * d0_right

    @property
    def p1_either(self) -> float:
        return self.d1_either / self.D


def build_spatial_setup(params: SpatialParams) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build initial state, unitary, and spatial masks.

    Reuses the Ising Hamiltonian from build_ising_setup, replacing the
    global Hamming mask with two spatial group masks.

    Returns (psi0, U, mask_left, mask_right).
    """
    D = params.D
    ising_p = IsingDirectParams(
        m=params.m, L=1, hamming_threshold=params.hamming_threshold,
        J=params.J, hx=params.hx, hz=params.hz, dt=params.dt,
        seed=params.seed, exact_expm=True,
    )
    psi0, U, _ = build_ising_setup(ising_p)

    # Spatial masks (replacing the global Hamming mask)
    left_bits = (1 << params.m_left) - 1
    mask_left = np.array([
        bin(v & left_bits).count("1") >= params.hamming_threshold
        for v in range(D)
    ])
    mask_right = np.array([
        bin(v >> params.m_left).count("1") >= params.hamming_threshold
        for v in range(D)
    ])

    return psi0, U, mask_left, mask_right


@dataclass(frozen=True)
class SpatialMultiscaleResult:
    """Results at all scales simultaneously."""

    scale1_left: CoherenceResult
    scale1_right: CoherenceResult
    scale2_either: CoherenceResult  # OR: at least one group fires
    params: SpatialParams


def simulate_spatial_multiscale(params: SpatialParams) -> SpatialMultiscaleResult:
    """Frequency-class analysis at three spatial scales simultaneously.

    Tracks the 4-outcome branching (00, 01, 10, 11) at each step,
    then aggregates into frequency classes for each scale:
      - Scale 1a: n_left (number of steps where left group had outcome 1)
      - Scale 1b: n_right
      - Scale 2:  n_both (number of steps where BOTH groups had outcome 1)
    """
    D = params.D
    L = params.L

    psi0, U, mask_left, mask_right = build_spatial_setup(params)

    # Four outcome masks (proper orthogonal partition of H)
    mask_00 = ~mask_left & ~mask_right
    mask_01 = ~mask_left & mask_right
    mask_10 = mask_left & ~mask_right
    mask_11 = mask_left & mask_right

    outcomes = [
        # (mask, delta_left, delta_right, delta_both)
        (mask_00, 0, 0, 0),
        (mask_01, 0, 1, 0),
        (mask_10, 1, 0, 0),
        (mask_11, 1, 1, 1),
    ]

    # Track by (n_left, n_right, n_both) -> state vector
    current: dict[tuple[int, int, int], np.ndarray] = {(0, 0, 0): psi0.copy()}

    for step in range(L):
        next_states: dict[tuple[int, int, int], np.ndarray] = {}

        for (nl, nr, nb), state in current.items():
            evolved = U @ state

            for mask, dl, dr, db in outcomes:
                proj = np.where(mask, evolved, 0.0)
                if np.vdot(proj, proj).real < 1e-30:
                    continue
                key = (nl + dl, nr + dr, nb + db)
                if key not in next_states:
                    next_states[key] = np.zeros(D, dtype=complex)
                next_states[key] += proj

        current = next_states

    # Aggregate into frequency classes for each scale
    def aggregate(current: dict, L: int, index: int) -> list[np.ndarray]:
        """Sum states by the value at position `index` in the key tuple."""
        freq_states = [np.zeros(D, dtype=complex) for _ in range(L + 1)]
        for key, state in current.items():
            n = key[index]
            if 0 <= n <= L:
                freq_states[n] += state
        return freq_states

    def aggregate_either(current: dict, L: int) -> list[np.ndarray]:
        """Sum states by n_either = n_left + n_right - n_both."""
        freq_states = [np.zeros(D, dtype=complex) for _ in range(L + 1)]
        for (nl, nr, nb), state in current.items():
            ne = nl + nr - nb
            if 0 <= ne <= L:
                freq_states[ne] += state
        return freq_states

    def to_result(freq_states: list[np.ndarray], born_freq: float) -> CoherenceResult:
        L = len(freq_states) - 1
        states = np.array(freq_states)
        gram = states @ states.conj().T
        weights = gram.diagonal().real
        epsilon = compute_epsilon(gram)
        return CoherenceResult(
            frequency_classes=np.arange(L + 1),
            gram_matrix=gram,
            epsilon=epsilon,
            weights=weights,
            born_frequency=born_freq,
        )

    return SpatialMultiscaleResult(
        scale1_left=to_result(aggregate(current, L, 0), L * params.p1_left),
        scale1_right=to_result(aggregate(current, L, 1), L * params.p1_right),
        scale2_either=to_result(aggregate_either(current, L), L * params.p1_either),
        params=params,
    )
