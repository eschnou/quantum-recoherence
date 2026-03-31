"""Direct Strasberg-style analysis on Ising dynamics.

No separate recorder register. The system IS the recorder.
The coarse-grained "outcome" at each step is defined analytically
by projectors, not by a physical gate.

This exactly mirrors Strasberg's formalism:
  |Ψ_L⟩ = U_L ... U_1 |Ψ₀⟩   (physical evolution)
  |ψ(n₁)⟩ = Σ_{x: n₁(x)=n₁} Π_{x_L} U_L ... Π_{x_1} U_1 |Ψ₀⟩   (analysis)
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

import cirq
import numpy as np
from scipy.linalg import expm

from recohere.analysis import CoherenceResult, compute_epsilon, compute_gram_matrix


@dataclass(frozen=True)
class IsingDirectParams:
    m: int  # total qubits
    L: int  # number of events
    hamming_threshold: int  # "outcome 1" if HW >= threshold
    J: float = 1.0  # effective coupling is J/2 due to ZZPowGate convention
    hx: float = 0.9045  # near critical point
    hz: float = 0.0  # no longitudinal field (preserves p1 = d1/D)
    trotter_steps: int = 1
    dt: float = 1.2
    seed: int = 42
    exact_expm: bool = True  # use exact matrix exponential

    @property
    def p1(self) -> float:
        d1 = sum(comb(self.m, k) for k in range(self.hamming_threshold, self.m + 1))
        return d1 / (2**self.m)


def build_ising_setup(params: IsingDirectParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build initial state, unitary, and outcome mask from params.

    Returns (psi0, U, mask_1) where:
      psi0: (D,) complex initial state vector
      U: (D, D) complex Ising Trotter unitary
      mask_1: (D,) bool mask for outcome "1" (Hamming weight >= threshold)
    """
    m = params.m
    D = 2**m

    rng = np.random.default_rng(params.seed)
    # Haar-random initial state on the full Hilbert space
    psi0 = rng.standard_normal(D) + 1j * rng.standard_normal(D)
    psi0 = (psi0 / np.linalg.norm(psi0)).astype(np.complex128)

    if params.exact_expm:
        # Build H explicitly and compute U = expm(-i H dt)
        H = np.zeros((D, D), dtype=complex)
        # ZZ couplings
        for i in range(m - 1):
            for v in range(D):
                bi = (v >> i) & 1
                bi1 = (v >> (i + 1)) & 1
                # Z_i Z_{i+1} eigenvalue: (+1 if same, -1 if different)
                H[v, v] += (params.J / 2) * (1 - 2 * (bi ^ bi1))
            # hx X_i terms
        for i in range(m):
            for v in range(D):
                w = v ^ (1 << i)  # flip bit i
                H[v, w] += params.hx
            # hz Z_i terms
        for i in range(m):
            for v in range(D):
                bi = (v >> i) & 1
                H[v, v] += params.hz * (1 - 2 * bi)
        U = expm(-1j * H * params.dt)
    else:
        qubits = cirq.LineQubit.range(m)
        sc = cirq.Circuit()
        dt_step = params.dt / params.trotter_steps
        for _ in range(params.trotter_steps):
            for i in range(m - 1):
                sc.append(cirq.ZZPowGate(exponent=params.J * dt_step / np.pi).on(qubits[i], qubits[i + 1]))
            for q in qubits:
                sc.append(cirq.rx(2 * params.hx * dt_step).on(q))
            for q in qubits:
                sc.append(cirq.rz(2 * params.hz * dt_step).on(q))
        U = cirq.unitary(sc)

    mask_1 = np.array([bin(v).count("1") >= params.hamming_threshold for v in range(D)])
    return psi0, U, mask_1


def simulate_and_analyze(params: IsingDirectParams) -> CoherenceResult:
    """Pure Ising evolution + analytical frequency-class decomposition."""
    L = params.L
    D = 2**params.m

    if L > 25:
        raise NotImplementedError(f"L={L} too large for branching")

    psi0, U, mask_1 = build_ising_setup(params)

    # Step-by-step branching
    current = {0: psi0.copy()}

    for step in range(L):
        next_states: dict[int, np.ndarray] = {}

        for n1_so_far, state in current.items():
            evolved = U @ state

            # Vectorized projection
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

    return CoherenceResult(
        frequency_classes=np.arange(L + 1),
        gram_matrix=gram,
        epsilon=eps,
        weights=weights,
        born_frequency=L * params.p1,
    )
