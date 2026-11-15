from __future__ import annotations

import numpy as np
import pytest

import testimages


def load_golden(name: str) -> dict[str, np.ndarray]:
    path = testimages.GOLDEN_DIR / f"{name}.npz"
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


@pytest.fixture(scope="session")
def images() -> dict[str, testimages.TestImage]:
    """All test images by id. Needs third_party/dlib (parity tests only)."""
    return {img.id: img for img in testimages.load_images()}


@pytest.fixture(scope="session")
def golden():
    return load_golden
