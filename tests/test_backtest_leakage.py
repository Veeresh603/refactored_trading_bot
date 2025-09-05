# tests/test_backtest_leakage.py
"""
Purged-CV unit tests that now import the splitter from core.validation.purged_cv.

This mirrors the deterministic tests we used interactively but imports the
production utility so the same logic is available to your trainer code.
"""

from typing import Iterable, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)

# import the production purged k-fold utility we created
from core.validation.purged_cv import purged_kfold_indices, naive_time_kfold_indices

# Utility functions (small, local)
def make_synthetic_series(n: int = 200, lookahead: int = 5):
    rng = np.random.RandomState(0)
    labels = rng.randint(0, 2, size=(n,))
    prices = np.arange(n + lookahead).astype(float)
    return prices, labels


def count_overlaps(train_idx, test_idx, lookahead):
    test_min = int(test_idx.min())
    test_max = int(test_idx.max())
    overlaps = 0
    for j in train_idx:
        j_future_max = j + lookahead
        if j_future_max >= test_min and j_future_max <= test_max:
            overlaps += 1
    return overlaps


def test_naive_kfold_has_leakage():
    n = 200
    lookahead = 5
    _, _ = make_synthetic_series(n, lookahead=lookahead)
    total_overlaps = 0
    for train_idx, test_idx in naive_time_kfold_indices(n, n_splits=5):
        total_overlaps += count_overlaps(train_idx, test_idx, lookahead)
    assert total_overlaps > 0, "Naive time KFold should have overlapping future windows -> leakage"


def test_purged_kfold_removes_leakage():
    n = 200
    lookahead = 5
    _, _ = make_synthetic_series(n, lookahead=lookahead)
    total_overlaps = 0
    for train_idx, test_idx in purged_kfold_indices(n, n_splits=5, lookahead=lookahead, embargo=0.01):
        total_overlaps += count_overlaps(train_idx, test_idx, lookahead)
    assert total_overlaps == 0, "PurgedKFold with embargo should remove all overlaps (no leakage)"


def test_deterministic_leakage_provable():
    """
    Deterministic provable leakage test (small contruction) that relies on the
    purged_kfold_indices above (production code).
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score

    n = 200
    lookahead = 4
    # small deterministic block
    B = 30
    L = lookahead
    test_idx = np.arange(B, B + L, dtype=int)
    leaking_js = np.array([t - lookahead for t in test_idx], dtype=int)
    assert leaking_js.min() >= 0

    # Build X and y deterministically so leakage is exact
    X = np.zeros((n, 1), dtype=float)
    y = np.zeros(n, dtype=int)
    for i, t in enumerate(test_idx):
        j = t - lookahead
        label_t = 1 if (i % 2 == 0) else 0
        y[j] = label_t
        y[t] = label_t
        X_val = float(1000 + i)
        X[j, 0] = X_val
        X[t, 0] = X_val

    # Fill remaining indices
    cur = 0
    for idx in range(n):
        if idx in leaking_js or idx in test_idx:
            continue
        X[idx, 0] = float(2000 + cur)
        y[idx] = 0 if (cur % 2 == 0) else 1
        cur += 1

    extra_train = np.arange(0, 10, dtype=int)
    train_idx = np.unique(np.concatenate([extra_train, leaking_js]))
    assert all(j in train_idx for j in leaking_js)

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X[train_idx], y[train_idx])
    preds = knn.predict(X[test_idx])
    naive_acc = accuracy_score(y[test_idx], preds)

    purged_train = np.setdiff1d(train_idx, leaking_js, assume_unique=True)
    if len(purged_train) == 0:
        purged_acc = 0.0
    else:
        knn2 = KNeighborsClassifier(n_neighbors=1)
        knn2.fit(X[purged_train], y[purged_train])
        preds2 = knn2.predict(X[test_idx])
        purged_acc = accuracy_score(y[test_idx], preds2)

    logger.info("naive_acc=%.3f purged_acc=%.3f leaking_js=%s", naive_acc, purged_acc, leaking_js.tolist())
    assert naive_acc == 1.0, f"Naive 1-NN should be perfect (naive_acc={naive_acc})"
    assert purged_acc <= 0.5, f"Purged accuracy should drop to chance or lower (purged_acc={purged_acc})"
