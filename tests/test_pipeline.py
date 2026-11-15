"""manyfaced.find_faces: the full pipeline in one call, from an array, a file or a Pillow image."""

from __future__ import annotations

import threading

import numpy as np
import pytest
from PIL import Image, ImageOps

import manyfaced
import manyfaced.compat.dlib
import testimages
from manyfaced import _image, _pipeline

ORIENTATION = 0x0112  # EXIF tag


def _gradient(h: int, w: int) -> np.ndarray:
    y, x = np.mgrid[0:h, 0:w]
    return np.stack([x * 255 // (w - 1), y * 255 // (h - 1), (x + y) % 256], axis=-1).astype(
        np.uint8
    )


def _assert_same(a: list[manyfaced.Face], b: list[manyfaced.Face]) -> None:
    assert [f.rect.to_tuple() for f in a] == [f.rect.to_tuple() for f in b]
    for fa, fb in zip(a, b, strict=True):
        np.testing.assert_array_equal(fa.landmarks, fb.landmarks)
        np.testing.assert_array_equal(fa.descriptor, fb.descriptor)


def test_no_faces():
    assert manyfaced.find_faces(_gradient(240, 320)) == []


def test_gray_input():
    gray = _gradient(240, 320)[:, :, 0]
    assert manyfaced.find_faces(gray, upsample=0) == []


def test_path_equals_array(tmp_path):
    img = _gradient(120, 160)
    path = tmp_path / "gradient.png"
    Image.fromarray(img).save(path)
    np.testing.assert_array_equal(_image.load_rgb(path, exif_transpose=True), img)
    assert manyfaced.find_faces(path) == manyfaced.find_faces(img) == []
    assert manyfaced.find_faces(str(path)) == []


def test_pillow_image_equals_array():
    img = _gradient(120, 160)
    pil = Image.fromarray(img)
    np.testing.assert_array_equal(_image.pil_to_rgb(pil, exif_transpose=True), img)
    assert manyfaced.find_faces(pil) == []
    # Other modes are changed into RGB.
    gray = Image.fromarray(img[:, :, 0])
    rgb = _image.pil_to_rgb(gray, exif_transpose=True)
    assert rgb.shape == (120, 160, 3)
    np.testing.assert_array_equal(rgb[:, :, 2], img[:, :, 0])
    assert manyfaced.find_faces(gray.convert("RGBA")) == []


def test_pillow_image_exif_orientation_is_applied(tmp_path):
    path = tmp_path / "rotated.jpg"
    exif = Image.Exif()
    exif[ORIENTATION] = 6
    Image.fromarray(_gradient(40, 60)).save(path, exif=exif)
    with Image.open(path) as im:
        assert _image.pil_to_rgb(im, exif_transpose=True).shape == (60, 40, 3)
        # An image that is already upright has no Orientation tag: it does not turn again.
        upright = ImageOps.exif_transpose(im)
        assert _image.pil_to_rgb(upright, exif_transpose=True).shape == (60, 40, 3)


def test_is_pil_image():
    assert _image.is_pil_image(Image.new("RGB", (2, 2)))
    assert not _image.is_pil_image(np.zeros((2, 2, 3), np.uint8))
    assert not _image.is_pil_image("photo.jpg")


@pytest.mark.parametrize("fmt", ["JPEG", "PNG"])
def test_exif_orientation_is_applied(tmp_path, fmt):
    path = tmp_path / f"rotated.{fmt.lower()}"
    exif = Image.Exif()
    exif[ORIENTATION] = 6  # displayed rotated 90 degrees clockwise
    Image.fromarray(_gradient(40, 60)).save(path, format=fmt, exif=exif)

    upright = _image.load_rgb(path, exif_transpose=True)
    assert upright.shape == (60, 40, 3) and upright.dtype == np.uint8
    assert manyfaced.find_faces(path) == []
    # dlib's load_rgb_image ignores the orientation, and so does the compat layer.
    assert manyfaced.compat.dlib.load_rgb_image(path).shape == (40, 60, 3)


def test_default_models_are_created_once():
    found: list[manyfaced.FaceRecognizer] = []

    def get() -> None:
        found.append(_pipeline._default(manyfaced.FaceRecognizer))

    threads = [threading.Thread(target=get) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(found) == 8 and all(r is found[0] for r in found)


def test_face_is_frozen():
    face = manyfaced.Face(manyfaced.Rect(0, 0, 9, 9), np.zeros((5, 2)), np.zeros(128))
    with pytest.raises(AttributeError):
        face.rect = manyfaced.Rect(1, 1, 2, 2)  # ty: ignore[invalid-assignment]


@pytest.mark.parity
@pytest.mark.parametrize("name", testimages.FACE_IMAGES)
def test_matches_the_three_steps(images, name):
    img = images[f"faces/{name}"].array
    faces = manyfaced.find_faces(img)

    rects = manyfaced.FhogDetector()(img, upsample=1)
    predictor = manyfaced.ShapePredictor()
    landmarks = [predictor(img, r) for r in rects]
    assert [f.rect.to_tuple() for f in faces] == [r.to_tuple() for r in rects]
    if not rects:
        return
    descriptors = manyfaced.FaceRecognizer().embed_many(img, landmarks)
    for face, lm, d in zip(faces, landmarks, descriptors, strict=True):
        assert not face.landmarks.flags.writeable and not face.descriptor.flags.writeable
        assert face.landmarks.shape == (5, 2) and face.landmarks.dtype == np.int64
        assert face.descriptor.shape == (128,) and face.descriptor.dtype == np.float64
        np.testing.assert_array_equal(face.landmarks, lm)
        np.testing.assert_array_equal(face.descriptor, d)


@pytest.mark.parity
def test_jitters_and_upsample_are_passed_through(images):
    img = images["faces/2008_002079.jpg"].array  # 1 face without upsampling, 6 with
    faces = manyfaced.find_faces(img, upsample=0, num_jitters=2)
    rects = manyfaced.FhogDetector()(img, upsample=0)
    assert len(faces) == 1 and faces[0].rect.to_tuple() == rects[0].to_tuple()
    expected = manyfaced.FaceRecognizer().embed_many(
        img, [f.landmarks for f in faces], num_jitters=2
    )
    np.testing.assert_array_equal(np.stack([f.descriptor for f in faces]), expected)


@pytest.mark.parity
@pytest.mark.parametrize("name", [n for n in testimages.FACE_IMAGES if n != "dogs.jpg"])
def test_path_matches_decoded_array(images, name):
    path = testimages.FACES_DIR / name
    with Image.open(path) as im:
        assert im.getexif().get(ORIENTATION, 1) == 1  # no rotation to apply
    faces = manyfaced.find_faces(path)
    assert faces
    _assert_same(faces, manyfaced.find_faces(images[f"faces/{name}"].array))
    with Image.open(path) as im:
        _assert_same(faces, manyfaced.find_faces(im))
