"""Reconhecimento em tempo real pela webcam. Pressione 'q' ou ESC para sair."""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from facial_recognition import config  # noqa: E402
from facial_recognition.modeling.predict import predict_image  # noqa: E402
from facial_recognition.recognition import load_database  # noqa: E402
from facial_recognition.visualization import draw_predictions  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    parser.add_argument("--every", type=int, default=config.PROCESS_EVERY_N_FRAMES,
                        help="processa a cada N frames")
    parser.add_argument("--scale", type=float, default=config.PROCESS_SCALE,
                        help="fator de reducao do frame antes de processar")
    args = parser.parse_args()

    database = load_database()
    print(f"Autorizados: {', '.join(database.people)} | limiar {config.get_threshold():.3f}")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Nao foi possivel abrir a camera {args.camera}. Tente --camera 1.")
        return 1

    predictions = []
    frame_count = 0
    last_time = time.time()
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Falha ao ler frame da camera.")
                break
            if config.MIRROR_CAMERA:
                frame = cv2.flip(frame, 1)

            # Reconhecimento e pesado: roda a cada N frames e reaproveita o resultado.
            if frame_count % max(1, args.every) == 0:
                predictions = predict_image(frame, database, scale=args.scale)
                now = time.time()
                fps = max(1, args.every) / max(now - last_time, 1e-6)
                last_time = now
            frame_count += 1

            display = draw_predictions(frame, predictions, copy=False)
            status = f"{fps:.1f} analises/s" if predictions else "nenhum rosto detectado"
            cv2.putText(display, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, config.WHITE, 2, cv2.LINE_AA)

            cv2.imshow(config.WINDOW_NAME, display)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
