import numpy as np

from facial_recognition import config
from facial_recognition.modeling.predict import Prediction
from facial_recognition.visualization import draw_prediction


def _border_pixel(authorized):
    frame = np.zeros((300, 300, 3), np.uint8)
    pred = Prediction((100, 100, 80, 80), "maria", 0.2, authorized, 0.99)
    draw_prediction(frame, pred)
    return tuple(int(v) for v in frame[140, 100])   # meio da borda esquerda


def test_authorized_box_is_green():
    assert _border_pixel(True) == config.GREEN


def test_denied_box_is_red():
    assert _border_pixel(False) == config.RED
