"""Golden chinese whispers labels from dlib .

Inputs are the committed golden descriptors (tests/golden/descriptors.npz), so the labels do not
depend on manyfaced's descriptors, plus synthetic edge lists for dlib.chinese_whispers.
"""

from __future__ import annotations

import sys

import numpy as np
from golden_common import GOLDEN_DIR, TestImage, dlib

THRESHOLDS = [0.3, 0.45, 0.5, 0.55, 0.6, 0.7]
SUBSETS = [8, 50, None]  # first N descriptors; None = all


def _graphs(descriptors: np.ndarray) -> list[list[tuple]]:
    """Edge lists for dlib.chinese_whispers: weighted, unweighted, gaps and duplicates."""
    n = len(descriptors)
    dist = np.linalg.norm(descriptors[:, None] - descriptors[None], axis=2)
    weighted = [
        (i, j, float(1.0 / (1.0 + dist[i, j])))
        for i in range(n)
        for j in range(i, n)
        if dist[i, j] < 0.7
    ]
    # A deterministic sparse "random" graph (LCG), with node 3 isolated and a gap before 60.
    unweighted, state = [], 12345
    for _ in range(150):
        state = (1103515245 * state + 12345) % (1 << 31)
        a = state % 60
        state = (1103515245 * state + 12345) % (1 << 31)
        b = state % 60
        if 3 not in (a, b):
            unweighted.append((a, b))
    unweighted.append((61, 61))
    duplicates = [(0, 1), (1, 0), (0, 1, 2.5), (2, 2), (1, 2, 0.5), (1, 2, 0.25)]
    return [weighted, unweighted, duplicates]


def generate(images: list[TestImage]) -> None:
    g = np.load(GOLDEN_DIR / "descriptors.npz")
    descriptors = g["single"].astype(np.float64)
    vectors = [dlib.vector(d.tolist()) for d in descriptors]

    cases_n, cases_t, labels = [], [], []
    for n in SUBSETS:
        subset = vectors if n is None else vectors[:n]
        for t in THRESHOLDS:
            cases_n.append(len(subset))
            cases_t.append(t)
            labels.append(np.array(dlib.chinese_whispers_clustering(subset, t), dtype=np.int64))

    graphs = _graphs(descriptors)
    graph_labels = [np.array(dlib.chinese_whispers(e), dtype=np.int64) for e in graphs]

    out = GOLDEN_DIR / "clusters.npz"
    payload = {
        "dlib_version": np.array(dlib.__version__),
        "case_num_descriptors": np.array(cases_n, dtype=np.int64),
        "case_threshold": np.array(cases_t, dtype=np.float64),
    }
    for k, lab in enumerate(labels):
        payload[f"labels_{k}"] = lab
    for k, (edges, lab) in enumerate(zip(graphs, graph_labels, strict=True)):
        payload[f"graph_{k}_edges"] = np.array(
            [(e[0], e[1], e[2] if len(e) == 3 else np.nan) for e in edges], dtype=np.float64
        )
        payload[f"graph_{k}_labels"] = lab
    np.savez_compressed(out, **payload)
    sizes = [int(lab.max()) + 1 for lab in labels]
    print(
        f"{out.name}: {len(labels)} clusterings (clusters: {sizes}), {len(graphs)} graphs",
        file=sys.stderr,
    )
