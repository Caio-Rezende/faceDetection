import numpy as np
import cv2

FaceLocation = tuple[int, int, int, int]
FaceEncoding = list[np.array]

max_output_size = 720


class RecognitionModel:
    def __init__(obj, encoding: FaceEncoding, location: FaceLocation, name: str, path: str | None = None):
        obj.encoding = encoding
        obj.location = location
        obj.name = name
        obj.path = path


def standardize_frame(frame: cv2.Mat, max_size: int = max_output_size) -> tuple[int, cv2.Mat]:
    cols, rows, _ = frame.shape
    largest = max(cols, rows)
    resize_factor = max_size / largest

    frame = cv2.resize(frame, [
        int(resize_factor * rows),
        int(resize_factor * cols)
    ])

    del cols, rows, largest

    return (resize_factor, frame)
