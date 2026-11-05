from __future__ import annotations

import itertools
import pytest
import torch

from benchmarks.mnist_prototype_retrieval import covering_radius, k_center_greedy


def test_k_center_greedy_toy() -> None:
    # Proves that on a hand-computed 1D toy instance, greedy k-center selects indices [0, 6, 5] with covering radius 7.
    x = torch.tensor([[0.0], [1.0], [2.0], [3.0], [8.0], [15.0], [30.0]])
    gen = torch.Generator().manual_seed(0)
    idx = k_center_greedy(x, 3, generator=gen, first=0)
    assert idx.tolist() == [0, 6, 5]
    assert covering_radius(x, x[idx]) == 7.0


def test_k_center_greedy_distinct_and_real_rows() -> None:
    # Proves that picked indices are strictly distinct and index into real rows of x.
    x = torch.randn(25, 4)
    gen = torch.Generator().manual_seed(42)
    idx = k_center_greedy(x, 10, generator=gen)
    assert len(idx) == 10
    assert len(set(idx.tolist())) == 10
    assert idx.dtype == torch.long
    for i in idx.tolist():
        assert torch.equal(x[i], x[i])


def test_k_center_greedy_step_by_step_farthest() -> None:
    # Proves that each pick maximizes the exact distance to the set of centers chosen so far.
    x = torch.randn(30, 5)
    gen = torch.Generator().manual_seed(123)
    idx = k_center_greedy(x, 7, generator=gen, first=2)
    for t in range(1, 7):
        current_centers = x[idx[:t]]
        dists = torch.cdist(x, current_centers).min(dim=1).values
        unpicked_mask = torch.ones(len(x), dtype=torch.bool)
        unpicked_mask[idx[:t]] = False
        max_dist = dists[unpicked_mask].max()
        assert torch.isclose(dists[idx[t]], max_dist)


def test_k_center_greedy_prefix_and_monotonic_radius() -> None:
    # Proves that choices share a common prefix for fixed seed/first and covering radius is non-increasing.
    x = torch.randn(20, 3)
    gen1 = torch.Generator().manual_seed(99)
    gen2 = torch.Generator().manual_seed(99)
    idx_k5 = k_center_greedy(x, 5, generator=gen1, first=1)
    idx_k8 = k_center_greedy(x, 8, generator=gen2, first=1)
    assert idx_k8[:5].tolist() == idx_k5.tolist()
    radii = [covering_radius(x, x[idx_k8[:k]]) for k in range(1, 9)]
    for prev, cur in zip(radii[:-1], radii[1:]):
        assert cur <= prev + 1e-6


def test_k_center_greedy_2_approximation_brute_force() -> None:
    # Proves that the greedy covering radius is within 2*OPT across 20 synthetic 2D instances.
    gen = torch.Generator().manual_seed(2026)
    n, d, k = 12, 2, 3
    for _ in range(20):
        x = torch.randn(n, d, generator=gen)
        combos = list(itertools.combinations(range(n), k))
        opt_radius = min(covering_radius(x, x[list(c)]) for c in combos)
        greedy_idx = k_center_greedy(x, k, generator=gen)
        greedy_radius = covering_radius(x, x[greedy_idx])
        assert greedy_radius <= 2.0 * opt_radius + 1e-6


def test_k_center_greedy_determinism_and_first_override() -> None:
    # Proves that identical seeds produce identical picks and first= overrides random initialization.
    x = torch.randn(15, 3)
    idx1 = k_center_greedy(x, 4, generator=torch.Generator().manual_seed(7))
    idx2 = k_center_greedy(x, 4, generator=torch.Generator().manual_seed(7))
    assert torch.equal(idx1, idx2)
    idx3 = k_center_greedy(x, 4, generator=torch.Generator().manual_seed(7), first=9)
    assert idx3[0].item() == 9


def test_k_center_greedy_k_equals_n() -> None:
    # Proves that k == N selects all rows and yields zero covering radius.
    x = torch.randn(10, 4)
    idx = k_center_greedy(x, 10, generator=torch.Generator().manual_seed(0))
    assert len(set(idx.tolist())) == 10
    assert covering_radius(x, x[idx]) == 0.0


def test_k_center_greedy_duplicate_rows() -> None:
    # Proves that duplicate rows still result in distinct selected indices without error.
    x = torch.tensor([[1.0, 2.0], [1.0, 2.0], [5.0, 6.0], [5.0, 6.0]])
    idx = k_center_greedy(x, 4, generator=torch.Generator().manual_seed(0), first=0)
    assert len(set(idx.tolist())) == 4


def test_k_center_greedy_invalid_args() -> None:
    # Proves that invalid dimensions, k out of bounds, or invalid first argument raise ValueError.
    x = torch.randn(10, 3)
    gen = torch.Generator().manual_seed(0)
    with pytest.raises(ValueError, match="2-D"):
        k_center_greedy(torch.randn(10), 3, generator=gen)
    with pytest.raises(ValueError, match="2-D"):
        k_center_greedy(torch.randn(10, 3, 2), 3, generator=gen)
    with pytest.raises(ValueError, match="integer in"):
        k_center_greedy(x, 0, generator=gen)
    with pytest.raises(ValueError, match="integer in"):
        k_center_greedy(x, 11, generator=gen)
    with pytest.raises(ValueError, match="integer in"):
        k_center_greedy(x, True, generator=gen)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="first must be an integer"):
        k_center_greedy(x, 3, generator=gen, first=-1)
    with pytest.raises(ValueError, match="first must be an integer"):
        k_center_greedy(x, 3, generator=gen, first=10)
    with pytest.raises(ValueError, match="first must be an integer"):
        k_center_greedy(x, 3, generator=gen, first=False)  # type: ignore[arg-type]


def test_k_center_greedy_input_not_modified() -> None:
    # Proves that k_center_greedy does not modify the input tensor x in-place.
    x = torch.randn(15, 3)
    clone = x.clone()
    _ = k_center_greedy(x, 5, generator=torch.Generator().manual_seed(42))
    assert torch.equal(x, clone)
