"""Leitura e organizacao das imagens (autorizadas e de teste)."""
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from facial_recognition import config


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in config.IMAGE_EXTENSIONS


def list_authorized_images(directory: Optional[Path] = None) -> List[Tuple[str, Path]]:
    """Retorna pares (nome, caminho) das fotos autorizadas.

    Estrutura aceita:
        authorized/maria/1.jpg  -> nome "maria"
        authorized/joao.jpg     -> nome "joao"
    """
    directory = Path(directory) if directory else config.AUTHORIZED_DIR
    if not directory.exists():
        raise FileNotFoundError(f"Pasta de autorizados nao encontrada: {directory}")

    pairs: List[Tuple[str, Path]] = []
    for item in sorted(directory.iterdir()):
        if item.is_dir():
            pairs += [(item.name, img) for img in sorted(item.iterdir()) if _is_image(img)]
        elif _is_image(item):
            pairs.append((item.stem, item))

    if not pairs:
        raise FileNotFoundError(
            f"Nenhuma imagem encontrada em {directory}. "
            "Adicione fotos em subpastas, ex.: authorized/maria/1.jpg"
        )
    return pairs


def list_test_images(directory: Optional[Path] = None) -> List[Path]:
    """Retorna os caminhos das imagens de teste."""
    directory = Path(directory) if directory else config.TEST_DIR
    if not directory.exists():
        raise FileNotFoundError(f"Pasta de teste nao encontrada: {directory}")

    images = sorted(p for p in directory.rglob("*") if _is_image(p))
    if not images:
        raise FileNotFoundError(f"Nenhuma imagem de teste encontrada em {directory}")
    return images


def load_image(path) -> np.ndarray:
    """Le uma imagem em BGR. Funciona com caminhos com acentos (Windows)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Imagem nao encontrada: {path}")
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Arquivo nao e uma imagem valida: {path}")
    return image


def save_interim(face: np.ndarray, name: str) -> Path:
    """Salva um rosto recortado em data/interim/ (util para depuracao)."""
    config.INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    out = config.INTERIM_DIR / f"{name}.jpg"
    ok, buf = cv2.imencode(".jpg", face)
    if not ok:
        raise ValueError("Falha ao codificar a imagem do rosto")
    buf.tofile(str(out))
    return out
