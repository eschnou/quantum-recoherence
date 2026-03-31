"""Tests for decohere.analysis — Gram matrix and epsilon computation."""

import numpy as np

from recohere.analysis import compute_epsilon, compute_gram_matrix
from recohere.ising_direct import IsingDirectParams, simulate_and_analyze


def test_gram_hermitian():
    """Gram matrix of random vectors should be Hermitian."""
    rng = np.random.default_rng(42)
    states = [rng.standard_normal(16) + 1j * rng.standard_normal(16) for _ in range(5)]
    G = compute_gram_matrix(states)
    np.testing.assert_allclose(G, G.conj().T, atol=1e-14)


def test_gram_diagonal_nonneg():
    rng = np.random.default_rng(42)
    states = [rng.standard_normal(8) + 1j * rng.standard_normal(8) for _ in range(4)]
    G = compute_gram_matrix(states)
    assert np.all(G.diagonal().real >= -1e-15)


def test_gram_orthogonal_states():
    """Orthogonal states should give a diagonal Gram matrix."""
    states = [np.zeros(4, dtype=complex) for _ in range(4)]
    for i in range(4):
        states[i][i] = 1.0
    G = compute_gram_matrix(states)
    np.testing.assert_allclose(G, np.eye(4), atol=1e-15)


def test_epsilon_orthogonal():
    """Orthogonal states: epsilon = 0 (fully decoherent)."""
    states = [np.zeros(4, dtype=complex) for _ in range(4)]
    for i in range(4):
        states[i][i] = 1.0
    G = compute_gram_matrix(states)
    eps = compute_epsilon(G)
    np.testing.assert_allclose(eps, 0.0, atol=1e-15)


def test_epsilon_identical():
    """Identical states: epsilon = 1 (fully recoherent)."""
    v = np.array([1, 0, 0, 0], dtype=complex)
    states = [v.copy(), v.copy(), v.copy()]
    G = compute_gram_matrix(states)
    eps = compute_epsilon(G)
    np.testing.assert_allclose(eps, 1.0, atol=1e-15)


def test_epsilon_range():
    """Epsilon should be in [0, 1] for non-zero-weight classes."""
    rng = np.random.default_rng(99)
    states = [rng.standard_normal(16) + 1j * rng.standard_normal(16) for _ in range(6)]
    G = compute_gram_matrix(states)
    eps = compute_epsilon(G)
    valid = eps[~np.isnan(eps)]
    assert np.all(valid >= -1e-10)
    assert np.all(valid <= 1.0 + 1e-10)


def test_epsilon_zero_weight():
    """Zero-weight states should get NaN epsilon."""
    states = [np.array([1, 0], dtype=complex), np.zeros(2, dtype=complex)]
    G = compute_gram_matrix(states)
    eps = compute_epsilon(G)
    assert not np.isnan(eps[0])
    assert np.isnan(eps[1])


def test_ising_gram_trace_one():
    """Ising model: Gram matrix trace should be 1 (completeness)."""
    p = IsingDirectParams(m=6, L=8, hamming_threshold=3, seed=42)
    result = simulate_and_analyze(p)
    np.testing.assert_allclose(np.trace(result.gram_matrix).real, 1.0, atol=1e-10)
