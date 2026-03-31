# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project demonstrating **Born-rule filtering from finite Hilbert space geometry**, based on Strasberg et al.'s
recoherence mechanism. A 9-qubit system evolving under deterministic Ising dynamics shows that Born-typical frequency
classes remain more decoherent than combinatorial-peak frequencies — deriving Born's rule from unitarity + finite
dimensions without additional postulates.

## Key References

- `references/Strasberg_Decoherence.pdf` — Main paper (arXiv:2601.19703)
- `references/Strasberg_Everything_all_at_once.pdf` — D^(-alpha) scaling (Phys. Rev. X 14, 041027, 2024)
- `references/Strasberg_Shearing Off the Tree.pdf` — Born-rule filtering, epsilon definition (eq. 6)

## Architecture

**No separate recorder register.** The 9-qubit system (D=512) IS the finite-dimensional recorder, exactly as in
Strasberg's formalism. The "outcome" at each step is defined by the Hamming weight of the qubits (≥ 4 → outcome "1",
p₁ ≈ 0.746), computed analytically in the frequency-class decomposition — not by a physical gate.

The evolution is L layers of exact matrix exponentiation of the transverse-field Ising Hamiltonian
(H = J/2·ZZ + hx·X, integrable near-critical regime; J/2 = 0.5, hx = 0.9045, hz = 0.0).

- **Frequency-class result**: 3.1× gap between ε_born and ε_comb at L=15, robust over 10 seeds
- **Branch-level result**: At L=12 (4096 branches, D=512), only 451 branches remain decoherent (ε < 0.3).
  Survivors cluster at Born-typical frequencies, not the combinatorial peak.

## Code

```
src/recohere/
  analysis.py       — Gram matrix, epsilon (Strasberg eq. 6)
  ising_direct.py   — Ising circuit with analytical coarse-graining
  strasberg.py      — Strasberg's random matrix model (positive control)
  branches.py       — Branch-level analysis: tracks all 2^L individual histories

tests/
  test_analysis.py  — Gram matrix and epsilon unit tests
  test_strasberg.py — Strasberg model tests including Born filtering assertion

scripts/
  simulate.py       — Frequency-class simulation: multi-seed Ising + Strasberg + plots
  branches.py       — Branch-level analysis: death of worlds, census, Gram heatmap
  one_reality.py    — Extract and visualize individual branches (one "reality")
```

## Build & Run

```bash
poetry install                                    # install deps
poetry run pytest tests/ -v                       # all tests (15)
poetry run python scripts/simulate.py             # frequency-class results + plots (~5 min)
poetry run python scripts/branches.py             # branch-level analysis (~30s)
poetry run python scripts/one_reality.py          # individual branch visualization (~30s)
```

## Key Physics

### Why naive circuits fail
A single shared system qubit creates a symmetric Markov chain: P(flip) = p₁ regardless of current state, so the
stationary distribution is always 0.5. Born filtering requires multi-qubit scrambling.

### What works
The system's finite Hilbert space (D=512) can't keep all 2^L histories decoherent when L is large. The histories that
remain decoherent are the Born-typical ones. No separate recorder needed — the system IS the recorder.

### Branch-level picture ("death of worlds")
At L=12 there are 4096 branches but only D=512 dimensions — 8× oversubscribed. The Gram matrix of all individual
branch states shows that:
- **Born-typical branches** (n₁/L ≈ 0.75) remain approximately orthogonal (ε < 0.3) — distinct worlds
- **Combinatorial-peak branches** (n₁/L ≈ 0.5) become highly coherent (ε ≈ 0.8) — indistinguishable, "dead" worlds
- **Extreme branches** (e.g. all-zeros) lose both weight AND identity — absorbed into neighboring branches

The all-zeros branch (never got outcome "1") has weight 7×10⁻¹² and ε = 0.85 (85% overlap with neighbors).
The all-ones branch (always got "1") has weight 0.024 and ε = 0.24 — still a distinct world, because p₁ = 0.746
makes this outcome unlikely but not impossible, and it has no frequency-class siblings to compete with.

Anti-Born branches don't just have low probability — they lose their ability to exist as separate realities.

### Critical: weak coupling in Strasberg's model
Strasberg requires c = 8λ²d₀d₁/(Dδε²) ≈ 0.0025. Too-strong coupling prevents thermalization and shifts the epsilon
dip away from the Born frequency.
