"""Deteccao de rostos usando o DeepFace (detectores configuraveis)."""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from facial_recognition import config


@dataclass
class DetectedFace:
    bbox: Tuple[int, int, int, int]   # (x, y, w, h) na imagem original
    confidence: float
    face: np.ndarray                  # recorte alinhado, BGR uint8


def _to_uint8_bgr(face: np.ndarray) -> np.ndarray:
    """Converte a saida do DeepFace (RGB, float 0-1 ou uint8) para BGR uint8."""
    arr = np.asarray(face)
    if arr.dtype != np.uint8:
        if arr.size and arr.max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr[:, :, ::-1].copy()


def detect_faces(
    image: np.ndarray,
    detector_backend: Optional[str] = None,
    min_confidence: Optional[float] = None,
) -> List[DetectedFace]:
    """Detecta rostos em uma imagem BGR. Retorna lista vazia se nao houver."""
    from deepface import DeepFace  # import tardio: TensorFlow e pesado

    backend = detector_backend or config.DETECTOR_BACKEND
    min_conf = config.MIN_FACE_CONFIDENCE if min_confidence is None else min_confidence
    img_h, img_w = image.shape[:2]

    try:
        results = DeepFace.extract_faces(
            img_path=image,
            detector_backend=backend,
            enforce_detection=False,   # nao levanta erro; checamos manualmente
            align=config.ALIGN,
        )
    except ValueError:
        return []

    faces: List[DetectedFace] = []
    for r in results:
        area = r["facial_area"]
        x, y, w, h = int(area["x"]), int(area["y"]), int(area["w"]), int(area["h"])
        confidence = float(r.get("confidence", 0.0))

        # Com enforce_detection=False, "sem rosto" volta como a imagem inteira.
        is_fallback = x <= 0 and y <= 0 and w >= img_w - 1 and h >= img_h - 1
        if is_fallback:
            continue
        if 0 < confidence < min_conf:
            continue

        faces.append(DetectedFace((x, y, w, h), confidence, _to_uint8_bgr(r["face"])))
    return faces


def largest_face(faces: List[DetectedFace]) -> Optional[DetectedFace]:
    """Rosto com maior area (assume ser o principal da foto)."""
    if not faces:
        return None
    return max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
