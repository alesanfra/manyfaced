"""Check that the images we decode are bit-identical to the ones dlib saw."""

import numpy as np
import pytest

import testimages


def _uses_images(path) -> bool:
    with np.load(path) as data:
        return "image_ids" in data.files


# Golden files built from test images (clusters.npz, for example, only uses descriptors).
GOLDEN_FILES = sorted(p.stem for p in testimages.GOLDEN_DIR.glob("*.npz") if _uses_images(p))


@pytest.mark.parity
@pytest.mark.parametrize("name", GOLDEN_FILES)
def test_image_digests_match(name, images, golden):
    g = golden(name)
    for image_id, expected in zip(g["image_ids"], g["image_digests"], strict=True):
        actual = testimages.digest(images[str(image_id)].array)
        assert actual == expected, (
            f"{image_id}: decoded pixels differ from the ones used to generate {name}.npz; "
            "Pillow/libjpeg probably changed, regenerate with tools/gen_golden.py"
        )
