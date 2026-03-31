"""Strasberg's random matrix model for Born-rule filtering.

Implements the toy model from Strasberg & Schindler, "Shearing Off the Tree"
(arXiv:2310.06755), Section "Toy model" and "Numerical evidence."

The Hilbert space H of dimension D is split into two subspaces:
  - H_0 of dimension d_0 (outcome "0")
  - H_1 of dimension d_1 (outcome "1")
  - D = d_0 + d_1

Born probability: p_1 = d_1 / D

The Hamiltonian has block structure:
  H = [[H_00, H_01],
       [H_10, H_11]]

where H_00 (d_0 x d_0) and H_11 (d_1 x d_1) are diagonal with evenly spaced
eigenvalues in [0, δε], and H_01 = H_10† = λR with R a random matrix
(Gaussian, zero mean, unit variance entries).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from recohere.analysis import CoherenceResult, compute_epsilon, compute_gram_matrix


@dataclass(frozen=True)
class StrasbergParams:
    """Parameters for Strasberg's random matrix model."""

    D: int  # total Hilbert space dimension
    d1_ratio: float  # d1/D, controls Born probability p1 = d1/D
    L: int  # number of events (history length)
    delta_epsilon: float = 0.5  # energy spread
    c: float = 0.0025  # weak coupling parameter (paper default)

    @property
    def d0(self) -> int:
        return self.D - round(self.D * self.d1_ratio)

    @property
    def d1(self) -> int:
        return round(self.D * self.d1_ratio)

    @property
    def lam(self) -> float:
        """Coupling λ derived from c = 8λ²d₀d₁/(Dδε²)."""
        return np.sqrt(self.c * self.D * self.delta_epsilon**2 / (8 * self.d0 * self.d1))

    @property
    def p1(self) -> float:
        return self.d1 / self.D


def build_hamiltonian(params: StrasbergParams, rng: np.random.Generator) -> np.ndarray:
    """Build the block Hamiltonian."""
    d0, d1, D = params.d0, params.d1, params.D

    H = np.zeros((D, D), dtype=complex)
    H[:d0, :d0] = np.diag(np.linspace(0, params.delta_epsilon, d0))
    H[d0:, d0:] = np.diag(np.linspace(0, params.delta_epsilon, d1))

    R = (rng.standard_normal((d0, d1)) + 1j * rng.standard_normal((d0, d1))) / np.sqrt(2)
    H[:d0, d0:] = params.lam * R
    H[d0:, :d0] = params.lam * R.conj().T

    return H


def run_strasberg(
    params: StrasbergParams, seed: int = 42
) -> CoherenceResult:
    """Run Strasberg's model and compute coherence analysis."""
    rng = np.random.default_rng(seed)
    d0, d1, D, L = params.d0, params.d1, params.D, params.L

    H = build_hamiltonian(params, rng)

    psi0 = rng.standard_normal(D) + 1j * rng.standard_normal(D)
    psi0 /= np.linalg.norm(psi0)

    # Eigendecompose once, then reconstruct U(t) cheaply per step
    eigvals, V = np.linalg.eigh(H)
    Vdag = V.conj().T

    tau = params.delta_epsilon / (2 * np.pi * params.lam**2 * D)
    times = rng.uniform(19.5 * tau, 20.5 * tau, size=L)

    # Step-by-step branching
    current = {0: psi0.copy()}

    for k in range(L):
        # U_k = V @ diag(exp(-i λ_j t_k)) @ V†  — O(D²) instead of O(D³) expm
        U = (V * np.exp(-1j * eigvals * times[k])) @ Vdag
        next_states: dict[int, np.ndarray] = {}

        for n1_so_far, state in current.items():
            evolved = U @ state

            # Project via slicing instead of D×D matmul
            proj_0 = np.zeros(D, dtype=complex)
            proj_0[:d0] = evolved[:d0]
            proj_1 = np.zeros(D, dtype=complex)
            proj_1[d0:] = evolved[d0:]

            for outcome, projected in [(0, proj_0), (1, proj_1)]:
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

    return CoherenceResult(
        frequency_classes=np.arange(L + 1),
        gram_matrix=gram,
        epsilon=eps,
        weights=weights,
        born_frequency=L * params.p1,
    )
