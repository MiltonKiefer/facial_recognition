"""Testa todas as imagens de data/raw/test/ e salva o resultado anotado."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from facial_recognition import config  # noqa: E402
from facial_recognition.dataset import list_test_images, load_image  # noqa: E402
from facial_recognition.modeling.predict import predict_image  # noqa: E402
from facial_recognition.recognition import load_database  # noqa: E402
from facial_recognition.visualization import draw_predictions, save_image, show_image  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true", help="exibe cada resultado na tela")
    args = parser.parse_args()

    database = load_database()
    print(f"Limiar: {config.get_threshold():.3f} | Autorizados: {', '.join(database.people)}\n")

    granted = denied = no_face = 0
    for path in list_test_images():
        image = load_image(path)
        predictions = predict_image(image, database)

        if not predictions:
            no_face += 1
            print(f"{path.name:35s} -> nenhum rosto detectado")
        for p in predictions:
            if p.authorized:
                granted += 1
                print(f"{path.name:35s} -> LIBERADO ({p.name}, dist={p.distance:.3f})")
            else:
                denied += 1
                print(f"{path.name:35s} -> NEGADO   (mais proximo: {p.name}, dist={p.distance:.3f})")

        annotated = draw_predictions(image, predictions)
        save_image(annotated, config.RESULTS_DIR / f"{path.stem}_result.jpg")
        if args.show:
            show_image(annotated, path.name)

    print(f"\nResumo: {granted} liberado(s), {denied} negado(s), {no_face} sem rosto.")
    print(f"Imagens salvas em: {config.RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
