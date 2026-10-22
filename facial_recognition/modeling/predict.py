"""Orquestracao: deteccao -> embedding -> comparacao -> decisao."""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from facial_recognition.detection import detect_faces
from facial_recognition.recognition import (
    FaceDatabase,
    compare,
    get_embedding,
    is_authorized,
    load_database,
)


@dataclass
class Prediction:
    bbox: Tuple[int, int, int, int]   # (x, y, w, h) na imagem original
    name: str                         # pessoa autorizada mais parecida
    distance: float                   # distancia de cosseno ate ela
    authorized: bool
    confidence: float                 # confianca da deteccao do rosto


def predict_image(
    image: np.ndarray,
    database: Optional[FaceDatabase] = None,
    scale: float = 1.0,
) -> List[Prediction]:
    """Preve o acesso para cada rosto da imagem.

    `scale` < 1 reduz a imagem antes de processar (mais rapido); as caixas
    retornadas sempre estao nas coordenadas da imagem original.
    """
    if database is None:
        database = load_database()

    work = image
    if scale != 1.0:
        work = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    predictions: List[Prediction] = []
    for face in detect_faces(work):
        embedding = get_embedding(face.face)
        name, distance = compare(embedding, database)

        x, y, w, h = face.bbox
        if scale != 1.0:
            x, y, w, h = (int(round(v / scale)) for v in (x, y, w, h))

        predictions.append(
            Prediction(
                bbox=(x, y, w, h),
                name=name,
                distance=distance,
                authorized=is_authorized(distance),
                confidence=face.confidence,
            )
        )
    return predictions
