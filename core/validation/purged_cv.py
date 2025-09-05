# core/validation/purged_cv.py
"""
Purged cross-validation utilities for time-series / event-based ML.

This module provides:
- purged_kfold_indices: time-ordered k-fold generator that purges training samples
  whose "future windows" would overlap the test fold, and applies an embargo.
- naive_time_kfold_indices: simple contiguous time folds (for comparison / testing)
- helpers: compute_embargo_size, purge_train_indices

The implementation is intentionally small, dependency-free, and well-documented so
it can be imported by training scripts and unit tests.

Example usage:
    from core.validation.purged_cv import purged_kfold_indices
    for train_idx, test_idx in purged_kfold_indices(n_samples=1000, n_splits=5, lookahead=10, embargo=0.01):
        ...  # use train_idx/test_idx

References:
- Lopez de Prado, "Advances in Financial Machine Learning" (purged k-fold & embargo)
"""
from typing import Iterable, Tuple
import numpy as np
import math

__all__ = [
    "purged_kfold_indices",
    "naive_time_kfold_indices",
    "compute_embargo_size",
    "purge_train_indices",
]


def compute_embargo_size(n_samples: int, embargo: float) -> int:
    """Return embargo size in samples given fraction. """
    if embargo is None or embargo <= 0:
        return 0
    return int(math.ceil(embargo * n_samples))


def naive_time_kfold_indices(n_samples: int, n_splits: int = 5) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    """Yield time-ordered contiguous (naive) train/test splits.

    Parameters
    ----------
    n_samples: int
        Number of samples (time-ordered indices 0..n_samples-1)
    n_splits: int
        Number of folds

    Yields
    ------
    (train_idx, test_idx)
        numpy integer arrays of indices
    """
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    fold_sizes = [(n_samples // n_splits) + (1 if i < n_samples % n_splits else 0) for i in range(n_splits)]
    cursor = 0
    for fs in fold_sizes:
        test_idx = np.arange(cursor, cursor + fs, dtype=int)
        train_idx = np.setdiff1d(np.arange(n_samples), test_idx, assume_unique=True)
        cursor += fs
        yield train_idx, test_idx


def purge_train_indices(train_candidates: np.ndarray, test_idx: np.ndarray, lookahead: int) -> np.ndarray:
    """Return train_candidates with indices purged if their forward window [j, j+lookahead]
    overlaps the test_idx range.

    Both inputs are expected to be integer numpy arrays.
    """
    if len(train_candidates) == 0:
        return train_candidates
    test_min = int(test_idx.min())
    test_max = int(test_idx.max())
    to_purge = []
    for j in train_candidates:
        j_future_min = int(j)
        j_future_max = int(min(train_candidates.max() + lookahead, j + lookahead))
        # overlap if intersect
        if not (j_future_max < test_min or j_future_min > test_max):
            to_purge.append(int(j))
    if not to_purge:
        return train_candidates
    return np.setdiff1d(train_candidates, np.array(to_purge, dtype=int), assume_unique=True)


def purged_kfold_indices(n_samples: int, n_splits: int = 5, lookahead: int = 5, embargo: float = 0.01) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    """Yield purged train/test indices implementing purged k-fold with embargo.

    Parameters
    ----------
    n_samples : int
        Number of time-ordered samples (indices 0..n_samples-1)
    n_splits : int
        Number of folds
    lookahead : int
        Maximum forward look (in number of samples) used when building labels/features.
        Any training index j whose future window [j, j+lookahead] intersects the test set
        will be removed from the training set for that fold.
    embargo : float
        Fraction of the dataset to embargo after the test set (e.g. 0.01 = 1%). Embargo
        prevents overlap through nearby events and is expressed as fraction of n_samples.

    Yields
    ------
    (train_idx, test_idx)
        numpy arrays of integer indices safe for time-series CV

    Notes
    -----
    This implementation is intentionally conservative and simple. For event-driven
    labels where each sample has an explicit end time, you should compute a per-sample
    end index and purge any training sample that overlaps any test sample's end time.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    fold_sizes = [(n_samples // n_splits) + (1 if i < n_samples % n_splits else 0) for i in range(n_splits)]

    test_starts = []
    cursor = 0
    for fs in fold_sizes:
        test_starts.append(cursor)
        cursor += fs

    embargo_size = compute_embargo_size(n_samples, embargo)

    for i, start in enumerate(test_starts):
        test_size = fold_sizes[i]
        test_idx = np.arange(start, start + test_size, dtype=int)
        embargo_end = min(n_samples, start + test_size + embargo_size)
        train_candidates = np.setdiff1d(np.arange(n_samples), np.arange(start, embargo_end), assume_unique=True)
        train_idx = purge_train_indices(train_candidates, test_idx, lookahead)
        yield train_idx, test_idx


# Small self-test when run directly
if __name__ == "__main__":
    import pprint
    print("Demo purged_kfold for n_samples=50, n_splits=5, lookahead=3, embargo=0.05")
    for tr, te in purged_kfold_indices(50, n_splits=5, lookahead=3, embargo=0.05):
        print("TEST:", te[:3], "...", te[-3:])
        print("TRAIN sample count:", len(tr))
    print("Done")
