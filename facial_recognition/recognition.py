"""Embeddings faciais, banco de autorizados e comparacao (nucleo do sistema)."""
import pickle
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from facial_recognition import config
from facial_recognition.dataset import list_authorized_images, load_image
from facial_recognition.detection import detect_faces, largest_face


@dataclass
class FaceDatabase:
    names: List[str]
    paths: List[str]
    embeddings: np.ndarray        # (N, D), cada linha normalizada (norma 1)
    model_name: str
    detector_backend: str

    def __len__(self) -> int:
        return len(self.names)

    @property
    def people(self) -> List[str]:
        return sorted(set(self.names))


# --------------------------------------------------------------------------- #
# Embeddings e distancia
# --------------------------------------------------------------------------- #
def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(norm, 1e-12, None)


def get_embedding(face_bgr: np.ndarray, model_name: Optional[str] = None) -> np.ndarray:
    """Gera o vetor (embedding) de um rosto ja recortado/alinhado."""
    from deepface import DeepFace

    rep = DeepFace.represent(
        img_path=face_bgr,
        model_name=model_name or config.MODEL_NAME,
        detector_backend="skip",     # o rosto ja foi detectado em detection.py
        enforce_detection=False,
        align=False,
    )
    return np.asarray(rep[0]["embedding"], dtype=np.float32)


def cosine_distances(embedding: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Distancia de cosseno (1 - similaridade) entre um vetor e varias linhas."""
    return 1.0 - _normalize(matrix) @ _normalize(np.asarray(embedding, dtype=np.float32))


def compare(embedding: np.ndarray, database: FaceDatabase) -> Tuple[str, float]:
    """Retorna (nome do mais parecido, menor distancia)."""
    if len(database) == 0:
        raise ValueError("Banco de autorizados vazio")
    distances = cosine_distances(embedding, database.embeddings)
    best = int(np.argmin(distances))
    return database.names[best], float(distances[best])


def is_authorized(distance: float, threshold: Optional[float] = None) -> bool:
    limit = config.get_threshold() if threshold is None else threshold
    return distance <= limit


# --------------------------------------------------------------------------- #
# Banco de dados de autorizados
# --------------------------------------------------------------------------- #
def build_database(save: bool = True, verbose: bool = True) -> FaceDatabase:
    """Calcula o embedding de cada foto autorizada e (opcionalmente) salva em disco."""
    names: List[str] = []
    paths: List[str] = []
    vectors: List[np.ndarray] = []

    for name, path in list_authorized_images():
        faces = detect_faces(load_image(path))
        face = largest_face(faces)
        if face is None:
            if verbose:
                print(f"[aviso] nenhum rosto detectado, ignorada: {path}")
            continue
        if len(faces) > 1 and verbose:
            print(f"[aviso] {len(faces)} rostos em {path.name}; usando o maior.")

        vectors.append(get_embedding(face.face))
        names.append(name)
        paths.append(str(path))
        if verbose:
            print(f"[ok] {name}: {path.name}")

    if not vectors:
        raise RuntimeError("Nenhum rosto valido nas fotos autorizadas.")

    database = FaceDatabase(
        names=names,
        paths=paths,
        embeddings=_normalize(np.vstack(vectors)),
        model_name=config.MODEL_NAME,
        detector_backend=config.DETECTOR_BACKEND,
    )
    if save:
        save_database(database)
    return database


def save_database(database: FaceDatabase) -> None:
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "names": database.names,
        "paths": database.paths,
        "embeddings": database.embeddings,
        "model_name": database.model_name,
        "detector_backend": database.detector_backend,
    }
    with open(config.EMBEDDINGS_PATH, "wb") as f:
        pickle.dump(payload, f)


def load_database(rebuild_if_missing: bool = True) -> FaceDatabase:
    """Carrega o cache; reconstroi se nao existir ou se o modelo/detector mudou."""
    path = config.EMBEDDINGS_PATH
    if path.exists():
        with open(path, "rb") as f:
            payload = pickle.load(f)
        if (
            payload.get("model_name") == config.MODEL_NAME
            and payload.get("detector_backend") == config.DETECTOR_BACKEND
        ):
            return FaceDatabase(**payload)
        print("[aviso] modelo/detector mudou desde o ultimo cadastro; reconstruindo.")
    elif not rebuild_if_missing:
        raise FileNotFoundError(f"Banco nao encontrado: {path}")
    return build_database()
