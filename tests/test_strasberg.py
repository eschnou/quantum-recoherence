"""Tests for decohere.strasberg — Strasberg random matrix model."""

import numpy as np
import pytest

from recohere.strasberg import StrasbergParams, build_hamiltonian, run_strasberg


def test_params():
    p = StrasbergParams(D=100, d1_ratio=0.5, L=10)
    assert p.d0 == 50
    assert p.d1 == 50
    assert p.p1 == 0.5


def test_params_asymmetric():
    p = StrasbergParams(D=100, d1_ratio=0.8, L=10)
    assert p.d0 == 20
    assert p.d1 == 80
    assert p.p1 == 0.8


def test_hamiltonian_hermitian():
    p = StrasbergParams(D=50, d1_ratio=0.5, L=5)
    rng = np.random.default_rng(42)
    H = build_hamiltonian(p, rng)
    np.testing.assert_allclose(H, H.conj().T, atol=1e-14)


def test_gram_trace_one():
    """Gram matrix trace should be 1 (completeness)."""
    p = StrasbergParams(D=50, d1_ratio=0.5, L=10)
    result = run_strasberg(p, seed=42)
    np.testing.assert_allclose(np.trace(result.gram_matrix).real, 1.0, atol=1e-10)


def test_epsilon_range():
    p = StrasbergParams(D=50, d1_ratio=0.8, L=15)
    result = run_strasberg(p, seed=42)
    valid = result.epsilon[~np.isnan(result.epsilon)]
    assert np.all(valid >= -1e-10)
    assert np.all(valid <= 1.0 + 1e-10)


def test_born_filtering_symmetric():
    """At p1=0.5, epsilon minimum should be near L/2."""
    p = StrasbergParams(D=200, d1_ratio=0.5, L=25)
    result = run_strasberg(p, seed=42)
    min_n1 = int(np.nanargmin(result.epsilon))
    # Should be near L/2 = 12.5
    assert abs(min_n1 - p.L / 2) <= 4, f"Expected min near {p.L/2}, got {min_n1}"


def test_born_filtering_asymmetric():
    """At p1=0.8 with large D, epsilon minimum should be near Born frequency.

    This is the key positive control: Strasberg's mechanism should show
    epsilon dip at n1/L ≈ 0.8, NOT at 0.5.
    """
    p = StrasbergParams(D=200, d1_ratio=0.8, L=25)
    result = run_strasberg(p, seed=42)

    born_n1 = round(result.born_frequency)
    min_n1 = int(np.nanargmin(result.epsilon))

    print(f"\nStrasberg positive control: D={p.D}, p1={p.p1}, L={p.L}")
    print(f"  Born freq n1={born_n1}, epsilon={result.epsilon[born_n1]:.4f}")
    print(f"  Min epsilon at n1={min_n1} (freq={min_n1/p.L:.2f})")
    print(f"  Epsilon profile: {np.array2string(result.epsilon, precision=3)}")

    # The minimum should be closer to Born than to combinatorial peak
    dist_to_born = abs(min_n1 - born_n1)
    dist_to_comb = abs(min_n1 - p.L // 2)
    assert dist_to_born < dist_to_comb, (
        f"Min epsilon at n1={min_n1}: closer to comb peak ({p.L//2}) "
        f"than Born freq ({born_n1})"
    )
