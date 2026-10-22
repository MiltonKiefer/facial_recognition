"""Configuracoes centralizadas do projeto.

Todos os outros modulos leem daqui: caminhos, modelo, limiar, cores e camera.
"""
from pathlib import Path

# --------------------------------------------------------------------------- #
# Caminhos
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
AUTHORIZED_DIR = RAW_DIR / "authorized"   # fotos das pessoas autorizadas
TEST_DIR = RAW_DIR / "test"               # fotos para teste em lote
INTERIM_DIR = DATA_DIR / "interim"        # rostos recortados (debug)
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROCESSED_DIR / "results"   # imagens de teste anotadas
EMBEDDINGS_PATH = PROCESSED_DIR / "embeddings.pkl"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# --------------------------------------------------------------------------- #
# Modelo de reconhecimento (DeepFace)
# --------------------------------------------------------------------------- #
MODEL_NAME = "Facenet512"          # alternativas: "ArcFace", "VGG-Face", "SFace"
DETECTOR_BACKEND = "retinaface"    # mais rapido: "opencv" | "ssd" | "mediapipe"
ALIGN = True                       # alinha o rosto pelos olhos antes do embedding
MIN_FACE_CONFIDENCE = 0.90         # descarta deteccoes pouco confiaveis

# Limiares de distancia de COSSENO (valores de referencia do DeepFace).
# Distancia <= limiar  ->  mesma pessoa.
COSINE_THRESHOLDS = {
    "VGG-Face": 0.68,
    "Facenet": 0.40,
    "Facenet512": 0.30,
    "ArcFace": 0.68,
    "Dlib": 0.07,
    "SFace": 0.593,
    "OpenFace": 0.10,
    "DeepFace": 0.23,
    "DeepID": 0.015,
    "GhostFaceNet": 0.65,
}
# Defina um numero (ex.: 0.25) para sobrescrever o padrao apos calibrar.
# Mais baixo = mais rigoroso (menos falsos positivos, mais falsos negativos).
THRESHOLD_OVERRIDE = None


def get_threshold() -> float:
    """Limiar efetivo em uso."""
    if THRESHOLD_OVERRIDE is not None:
        return float(THRESHOLD_OVERRIDE)
    return COSINE_THRESHOLDS[MODEL_NAME]


# --------------------------------------------------------------------------- #
# Visualizacao (cores em BGR, padrao do OpenCV)
# --------------------------------------------------------------------------- #
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

TEXT_GRANTED = "acesso liberado"
TEXT_DENIED = "acesso negado"

BOX_THICKNESS = 3
FONT_SCALE = 0.7
FONT_THICKNESS = 2

# --------------------------------------------------------------------------- #
# Camera
# --------------------------------------------------------------------------- #
CAMERA_INDEX = 0
PROCESS_EVERY_N_FRAMES = 5   # roda o reconhecimento a cada N frames
PROCESS_SCALE = 0.5          # reduz o frame antes de processar (1.0 = original)
MIRROR_CAMERA = True         # espelha a imagem (efeito "espelho")
WINDOW_NAME = "Reconhecimento facial - 'q' ou ESC para sair"
