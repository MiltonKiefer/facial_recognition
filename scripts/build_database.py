"""Gera data/processed/embeddings.pkl a partir de data/raw/authorized/."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from facial_recognition import config  # noqa: E402
from facial_recognition.recognition import build_database  # noqa: E402


def main() -> int:
    print(f"Modelo: {config.MODEL_NAME} | Detector: {config.DETECTOR_BACKEND}")
    db = build_database()
    print(f"\nIndexadas {len(db)} fotos de {len(db.people)} pessoa(s): {', '.join(db.people)}")
    print(f"Salvo em: {config.EMBEDDINGS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
