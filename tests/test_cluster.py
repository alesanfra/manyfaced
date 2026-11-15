"""chinese_whispers parity with dlib. Needs only committed golden files (no models/images)."""

import numpy as np
import pytest

import manyfaced
import testimages


@pytest.fixture(scope="module")
def clusters(golden):
    return golden("clusters")


@pytest.fixture(scope="module")
def descriptors(golden):
    return golden("descriptors")["single"].astype(np.float64)


def test_clustering_labels_match_dlib(clusters, descriptors):
    n_cases = len(clusters["case_threshold"])
    assert n_cases > 0
    for k in range(n_cases):
        n = int(clusters["case_num_descriptors"][k])
        t = float(clusters["case_threshold"][k])
        got = manyfaced.chinese_whispers_clustering(descriptors[:n], t)
        np.testing.assert_array_equal(got, clusters[f"labels_{k}"], err_msg=f"n={n} t={t}")


def test_raw_graph_labels_match_dlib(clusters):
    k = 0
    while f"graph_{k}_edges" in clusters:
        edges = [
            (int(a), int(b)) if np.isnan(w) else (int(a), int(b), float(w))
            for a, b, w in clusters[f"graph_{k}_edges"]
        ]
        got = manyfaced.chinese_whispers(edges)
        np.testing.assert_array_equal(got, clusters[f"graph_{k}_labels"], err_msg=f"graph {k}")
        k += 1
    assert k == 3


def test_edge_cases():
    assert manyfaced.chinese_whispers([]).tolist() == []
    assert manyfaced.chinese_whispers_clustering(np.zeros((0, 128)), 0.6).tolist() == []
    assert manyfaced.chinese_whispers_clustering([[0.0, 0.0], [3.0, 4.0]], 5.5).tolist() == [
        0,
        0,
    ]
    with pytest.raises(ValueError, match="threshold"):
        manyfaced.chinese_whispers_clustering([[0.0]], 0.0)
    with pytest.raises(ValueError, match="threshold"):
        manyfaced.chinese_whispers_clustering([[0.0]], float("nan"))
    with pytest.raises(ValueError, match=r"\(N, D\)"):
        manyfaced.chinese_whispers_clustering([0.0, 1.0], 0.5)
    with pytest.raises(TypeError, match="tuples"):
        manyfaced.chinese_whispers([(0, 1, 2, 3)])
    with pytest.raises(TypeError, match="tuples"):
        manyfaced.chinese_whispers([(-1, 1)])
    with pytest.raises(ValueError, match="finite"):
        manyfaced.chinese_whispers_clustering([[0.0], [1.0], [float("nan")]], 0.5)
    # The node table has one entry for each index up to the largest: too large for the memory.
    with pytest.raises(MemoryError):
        manyfaced.chinese_whispers([(0, 2**62)])


def test_golden_files_exist():
    assert (testimages.GOLDEN_DIR / "clusters.npz").exists()
