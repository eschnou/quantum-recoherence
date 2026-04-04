"""Spatial multi-scale Born filtering.

Split 8 qubits into two groups of 4 (left: 0-3, right: 4-7).
At each step, apply independent Hamming-weight projectors to each group.
This gives three nested scales of proper orthogonal projectors:

  Scale 1a: left group HW >= threshold  (p1 = d1_left / D)
  Scale 1b: right group HW >= threshold (p1 = d1_right / D)
  Scale 2:  both groups >= threshold    (p1 = d1_left * d1_right / D^2... but also = d_both / D)

All three are genuine projector decompositions of the same Hilbert space.
Born filtering operates independently at each level.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from recohere.analysis import CoherenceResult, compute_epsilon


@dataclass(frozen=True)
class SpatialParams:
    m_left: int = 4
    m_right: int = 4
    hamming_threshold: int = 3  # HW >= threshold on each group
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

    Returns (psi0, U, mask_left, mask_right).
    """
    m = params.m
    D = params.D

    rng = np.random.default_rng(params.seed)
    psi0 = rng.standard_normal(D) + 1j * rng.standard_normal(D)
    psi0 = (psi0 / np.linalg.norm(psi0)).astype(np.complex128)

    # Ising Hamiltonian on m qubits
    H = np.zeros((D, D), dtype=complex)
    for i in range(m - 1):
        for v in range(D):
            bi = (v >> i) & 1
            bi1 = (v >> (i + 1)) & 1
            H[v, v] += (params.J / 2) * (1 - 2 * (bi ^ bi1))
    for i in range(m):
        for v in range(D):
            w = v ^ (1 << i)
            H[v, w] += params.hx
    for i in range(m):
        for v in range(D):
            bi = (v >> i) & 1
            H[v, v] += params.hz * (1 - 2 * bi)
    U = expm(-1j * H * params.dt)

    # Spatial masks
    left_bits = (1 << params.m_left) - 1  # 0x0F for m_left=4
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
    scale2_both: CoherenceResult    # AND: both groups fire
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
        # (mask, delta_left, delta_right, delta_both, delta_either)
        (mask_00, 0, 0, 0, 0),
        (mask_01, 0, 1, 0, 1),
        (mask_10, 1, 0, 0, 1),
        (mask_11, 1, 1, 1, 1),
    ]

    # Track by (n_left, n_right, n_both, n_either) -> state vector
    current: dict[tuple[int, int, int, int], np.ndarray] = {(0, 0, 0, 0): psi0.copy()}

    for step in range(L):
        next_states: dict[tuple[int, int, int, int], np.ndarray] = {}

        for (nl, nr, nb, ne), state in current.items():
            evolved = U @ state

            for mask, dl, dr, db, de in outcomes:
                proj = np.where(mask, evolved, 0.0)
                if np.vdot(proj, proj).real < 1e-30:
                    continue
                key = (nl + dl, nr + dr, nb + db, ne + de)
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
        scale2_both=to_result(aggregate(current, L, 2), L * params.p1_both),
        scale2_either=to_result(aggregate(current, L, 3), L * params.p1_either),
        params=params,
    )
