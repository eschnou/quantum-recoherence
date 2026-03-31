"""Coherence analysis: Gram matrix and recoherence parameter epsilon.

This is the physics core. Computes the recoherence parameter epsilon(n1)
that determines whether a frequency class is decoherent or recoherent.

The frequency-class decomposition itself is model-specific and lives in
each model's module (ising_direct.py, strasberg.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CoherenceResult:
    """Results of coherence analysis for a single parameter point."""

    frequency_classes: np.ndarray  # n1 values: 0, 1, ..., L
    gram_matrix: np.ndarray  # (L+1, L+1) complex
    epsilon: np.ndarray  # (L+1,) recoherence parameter
    weights: np.ndarray  # (L+1,) total weight per frequency class
    born_frequency: float  # L * p1


def compute_gram_matrix(freq_states: list[np.ndarray]) -> np.ndarray:
    """Gram matrix G[i,j] = <psi(i)|psi(j)>."""
    M = np.array(freq_states)  # (L+1, d_full)
    return M @ M.conj().T


def compute_epsilon(gram_matrix: np.ndarray) -> np.ndarray:
    """Recoherence parameter for each frequency class.

    epsilon = 0: fully decoherent (no overlap with others).
    epsilon = 1: fully recoherent.

    Definition from Strasberg & Schindler, "Shearing Off the Tree", eq. (6):
        epsilon(n) = max_{m != n} |<psi(m)|psi(n)>| / sqrt(<psi(m)|psi(m)> * <psi(n)|psi(n)>)

    This is the maximum normalized overlap (fidelity) between frequency-class
    state n and any other frequency-class state m.

    For zero-weight classes, epsilon is set to NaN.
    """
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
