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
from typing import Literal

import cirq
import numpy as np
from scipy.linalg import expm

from recohere.analysis import CoherenceResult, compute_epsilon, compute_gram_matrix

Projector = Literal["hamming", "spatial_majority", "left_heavy", "parity"]


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
    projector: Projector = "hamming"

    @property
    def p1(self) -> float:
        if self.projector == "hamming":
            d1 = sum(comb(self.m, k) for k in range(self.hamming_threshold, self.m + 1))
            return d1 / (2**self.m)
        mask = _build_mask(self.m, self.projector, self.hamming_threshold)
        return int(np.sum(mask)) / (2**self.m)


def _build_mask(m: int, projector: Projector, hamming_threshold: int) -> np.ndarray:
    """Build the boolean outcome mask for the given projector type.

    Returns (D,) bool array where True means outcome "1".
    """
    D = 2**m
    if projector == "hamming":
        return np.array([bin(v).count("1") >= hamming_threshold for v in range(D)])
    elif projector == "spatial_majority":
        # "1" if majority of the first ceil(m/2) qubits are |1>
        n_sub = (m + 1) // 2
        sub_threshold = (n_sub + 1) // 2  # strict majority
        return np.array([
            bin(v & ((1 << n_sub) - 1)).count("1") >= sub_threshold
            for v in range(D)
        ])
    elif projector == "left_heavy":
        # "1" if first ceil(m/2) qubits have HW >= hamming_threshold
        # Structurally different from global Hamming: only sees a spatial subsystem
        n_sub = (m + 1) // 2
        sub_mask = (1 << n_sub) - 1
        return np.array([
            bin(v & sub_mask).count("1") >= hamming_threshold
            for v in range(D)
        ])
    elif projector == "parity":
        return np.array([bin(v).count("1") % 2 == 1 for v in range(D)])
    else:
        raise ValueError(f"Unknown projector: {projector}")


def build_product_unitary(m: int, theta: float) -> np.ndarray:
    """Build product rotation U = exp(-i theta X)^{otimes m}.

    Each qubit rotates independently — no entanglement, no scrambling.
    This is the non-scrambling control for testing whether Born-rule
    filtering requires dynamical complexity or is purely kinematic.
    """
    D = 2**m
    # Single-qubit rotation: exp(-i theta X) = cos(theta) I - i sin(theta) X
    c, s = np.cos(theta), np.sin(theta)
    u1 = np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)
    # Build m-qubit product unitary via tensor product
    U = np.array([[1.0]], dtype=complex)
    for _ in range(m):
        U = np.kron(U, u1)
    assert U.shape == (D, D)
    return U


def build_ising_hamiltonian(m: int, J: float = 1.0, hx: float = 0.9045,
                            hz: float = 0.0) -> np.ndarray:
    """Build the transverse-field Ising Hamiltonian explicitly.

    H = (J/2) Σ Z_i Z_{i+1} + hx Σ X_i + hz Σ Z_i

    Returns (D, D) complex Hamiltonian matrix.
    """
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
        H = build_ising_hamiltonian(m, params.J, params.hx, params.hz)
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

    mask_1 = _build_mask(m, params.projector, params.hamming_threshold)
    return psi0, U, mask_1


def frequency_class_analysis(
    psi0: np.ndarray, U: np.ndarray, mask_1: np.ndarray, L: int, p1: float,
) -> CoherenceResult:
    """Core frequency-class decomposition for an arbitrary projector mask.

    Args:
        psi0: (D,) initial state
        U: (D, D) unitary
        mask_1: (D,) bool mask for outcome "1"
        L: number of steps
        p1: probability of outcome "1" (= d1/D for Born frequency)
    """
    D = psi0.shape[0]
    if L > 25:
        raise NotImplementedError(f"L={L} too large for branching")

    current = {0: psi0.copy()}
    for step in range(L):
        next_states: dict[int, np.ndarray] = {}
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
    return CoherenceResult(
        frequency_classes=np.arange(L + 1),
        gram_matrix=gram,
        epsilon=eps,
        weights=weights,
        born_frequency=L * p1,
    )


def simulate_and_analyze(params: IsingDirectParams) -> CoherenceResult:
    """Pure Ising evolution + analytical frequency-class decomposition."""
    psi0, U, mask_1 = build_ising_setup(params)
    return frequency_class_analysis(psi0, U, mask_1, params.L, params.p1)


def simulate_product(params: IsingDirectParams, theta: float | None = None) -> CoherenceResult:
    """Product-rotation control: same projectors, no scrambling.

    Uses U = exp(-i theta X)^{otimes m} instead of the Ising unitary.
    Default theta = hx * dt to match the X-rotation strength of the Ising model.
    """
    L = params.L
    D = 2**params.m

    if L > 25:
        raise NotImplementedError(f"L={L} too large for branching")

    if theta is None:
        theta = params.hx * params.dt

    # Same Haar-random initial state (same seed → same psi0)
    rng = np.random.default_rng(params.seed)
    psi0 = rng.standard_normal(D) + 1j * rng.standard_normal(D)
    psi0 = (psi0 / np.linalg.norm(psi0)).astype(np.complex128)

    U = build_product_unitary(params.m, theta)
    mask_1 = _build_mask(params.m, params.projector, params.hamming_threshold)

    # Identical branching logic
    current = {0: psi0.copy()}

    for step in range(L):
        next_states: dict[int, np.ndarray] = {}

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

    return CoherenceResult(
        frequency_classes=np.arange(L + 1),
        gram_matrix=gram,
        epsilon=eps,
        weights=weights,
        born_frequency=L * params.p1,
    )
