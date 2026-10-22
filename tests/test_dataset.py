import cv2
import numpy as np
import pytest

from facial_recognition import dataset


def _write_img(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.zeros((10, 10, 3), np.uint8))


def test_list_authorized_images_by_subfolder(tmp_path):
    _write_img(tmp_path / "maria" / "1.jpg")
    _write_img(tmp_path / "maria" / "2.png")
    _write_img(tmp_path / "joao.jpg")
    (tmp_path / "maria" / "notas.txt").write_text("ignorar")

    pairs = dataset.list_authorized_images(tmp_path)
    assert sorted(n for n, _ in pairs) == ["joao", "maria", "maria"]


def test_empty_folder_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Nenhuma imagem"):
        dataset.list_authorized_images(tmp_path)


def test_missing_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        dataset.list_test_images(tmp_path / "nao_existe")


def test_load_image_validates(tmp_path):
    with pytest.raises(FileNotFoundError):
        dataset.load_image(tmp_path / "x.jpg")
    bad = tmp_path / "bad.jpg"
    bad.write_text("nao sou imagem")
    with pytest.raises(ValueError):
        dataset.load_image(bad)
    _write_img(tmp_path / "ok.jpg")
    assert dataset.load_image(tmp_path / "ok.jpg").shape == (10, 10, 3)
