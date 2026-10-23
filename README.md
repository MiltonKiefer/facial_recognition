# facial_recognition

Controle de acesso por reconhecimento facial com **OpenCV** + **DeepFace**.
Caixa **verde** ("acesso liberado") para pessoas autorizadas e **vermelha** ("acesso negado") para as demais.

## Instalação
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Uso
1. Coloque 3–5 fotos de cada pessoa autorizada em `data/raw/authorized/<nome>/` (varie luz e ângulo).
2. Gere o banco de embeddings:
   ```bash
   python scripts/build_database.py
   ```
3. Teste com fotos (coloque-as em `data/raw/test/`); os resultados vão para `data/processed/results/`:
   ```bash
   python scripts/run_test_images.py
   ```
4. Teste ao vivo pela webcam (`q` ou ESC para sair):
   ```bash
   python scripts/run_webcam.py
   ```
5. Testes automatizados: `pytest`

## Calibração
Edite `facial_recognition/config.py`:
- `THRESHOLD_OVERRIDE`: menor = mais rigoroso (menos falsos positivos).
- `COSINE_THRESHOLDS`: configura o rigor específico de cada modelo de reconhecimento.
- `DETECTOR_BACKEND`: `"opencv"` é mais rápido; `"retinaface"` é mais preciso.
- `MODEL_NAME`: `"Facenet512"` (padrão) ou `"ArcFace"`.

Ao trocar modelo ou detector, o banco é reconstruído automaticamente.

## Privacidade
Fotos e embeddings são dados biométricos e estão no `.gitignore`.
