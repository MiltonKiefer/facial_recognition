"""Desenho das caixas (verde = liberado, vermelho = negado) e utilitarios de imagem."""
import unicodedata
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from facial_recognition import config
from facial_recognition.modeling.predict import Prediction

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _ascii(text: str) -> str:
    """O OpenCV nao desenha acentos: 'Joao' em vez de 'Joao' com til."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def draw_label(frame, text, top_left, bg_color, text_color) -> None:
    """Texto com fundo preenchido, acima da caixa (ou dentro, se nao couber)."""
    (tw, th), baseline = cv2.getTextSize(
        text, FONT, config.FONT_SCALE, config.FONT_THICKNESS
    )
    x, y = top_left
    pad = 6
    box_h = th + baseline + pad
    top = y - box_h
    if top < 0:
        top = y
    cv2.rectangle(frame, (x, top), (x + tw + 8, top + box_h), bg_color, -1)
    cv2.putText(
        frame, text, (x + 4, top + th + pad // 2), FONT,
        config.FONT_SCALE, text_color, config.FONT_THICKNESS, cv2.LINE_AA,
    )


def draw_prediction(frame: np.ndarray, prediction: Prediction, show_distance: bool = True):
    """Desenha a caixa e o texto de uma predicao (altera `frame` in-place)."""
    x, y, w, h = prediction.bbox

    if prediction.authorized:
        color, text_color = config.GREEN, config.BLACK
        text = f"{config.TEXT_GRANTED}: {_ascii(prediction.name)}"
    else:
        color, text_color = config.RED, config.WHITE
        text = config.TEXT_DENIED

    if show_distance:
        text += f" ({prediction.distance:.2f})"

    cv2.rectangle(frame, (x, y), (x + w, y + h), color, config.BOX_THICKNESS)
    draw_label(frame, text, (x, y), color, text_color)
    return frame


def draw_predictions(frame: np.ndarray, predictions: Iterable[Prediction], copy: bool = True):
    out = frame.copy() if copy else frame
    for p in predictions:
        draw_prediction(out, p)
    return out


def show_image(image: np.ndarray, window_name: str = "Resultado") -> None:
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)


def save_image(image: np.ndarray, path) -> Path:
    """Salva imagem (suporta caminhos com acentos)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".jpg", image)
    if not ok:
        raise ValueError(f"Falha ao codificar a imagem: {path}")
    buf.tofile(str(path))
    return path
