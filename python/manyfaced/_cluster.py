"""Graph clustering: dlib's chinese whispers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import numpy.typing as npt

from . import _manyfaced

__all__ = ["chinese_whispers", "chinese_whispers_clustering"]


def chinese_whispers(
    edges: Iterable[tuple[int, int] | tuple[int, int, float]], num_iterations: int = 100
) -> np.ndarray:
    """Cluster a graph of ``(i, j)`` or ``(i, j, weight)`` edges, as dlib's ``chinese_whispers``.

    The result has one int64 label for each node from 0 to the largest node index. The labels
    are numbered in the order in which they first occur. They are identical to the dlib labels.
    """
    labels = _manyfaced._chinese_whispers(list(edges), num_iterations)
    return np.asarray(labels, dtype=np.int64)


def chinese_whispers_clustering(
    descriptors: npt.ArrayLike | Sequence[npt.ArrayLike], threshold: float
) -> np.ndarray:
    """Cluster face descriptors, as dlib's ``chinese_whispers_clustering``.

    Two descriptors have a connection when their Euclidean distance is less than ``threshold``.
    ``threshold`` is rounded to float32, as in the dlib binding. The result has one int64 label
    for each descriptor. Descriptors with values that are not finite cause ``ValueError``.
    """
    arr = np.asarray(descriptors, dtype=np.float64)
    if arr.size == 0:
        return np.zeros(0, dtype=np.int64)
    if arr.ndim != 2:
        raise ValueError(f"expected an (N, D) array of descriptors, got shape {arr.shape}")
    # dlib gives no label to a last descriptor that is not finite, so that the labels and the
    # descriptors do not align. Refuse such input.
    if not np.isfinite(arr).all():
        raise ValueError("all descriptor values must be finite")
    labels = _manyfaced._chinese_whispers_clustering(np.ascontiguousarray(arr), threshold)
    return np.asarray(labels, dtype=np.int64)
