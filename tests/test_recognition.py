"""Testes da logica de comparacao (nao precisam do DeepFace)."""
import numpy as np
import pytest

from facial_recognition import config
from facial_recognition.recognition import (
    FaceDatabase, _normalize, compare, cosine_distances, is_authorized,
)


def _db():
    rng = np.random.default_rng(0)
    emb = _normalize(rng.normal(size=(4, 128)).astype(np.float32))
    return FaceDatabase(["maria", "maria", "joao", "ana"], ["a", "b", "c", "d"],
                        emb, config.MODEL_NAME, config.DETECTOR_BACKEND)


def test_identical_vector_has_zero_distance():
    db = _db()
    assert cosine_distances(db.embeddings[2], db.embeddings)[2] == pytest.approx(0, abs=1e-6)


def test_compare_finds_correct_person():
    db = _db()
    noisy = db.embeddings[2] + np.random.default_rng(1).normal(scale=0.01, size=128)
    name, dist = compare(noisy, db)
    assert name == "joao" and dist < config.get_threshold()


def test_unknown_vector_is_denied():
    db = _db()
    unknown = np.random.default_rng(99).normal(size=128).astype(np.float32)
    _, dist = compare(unknown, db)
    assert not is_authorized(dist)


def test_threshold_boundary():
    assert is_authorized(0.1, threshold=0.1)
    assert not is_authorized(0.11, threshold=0.1)
