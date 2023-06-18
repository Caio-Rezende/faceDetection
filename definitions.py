import numpy as np
import cv2

FaceLocation = tuple[int, int, int, int]
FaceEncoding = list[np.array]

max_output_size = 720


class RecognitionModel:
    def __init__(obj, encoding, name: str, file: str):
        obj.encoding = encoding
        obj.name = name
        obj.file = file


def standardize_frame(frame: cv2.Mat):
    cols, rows, _ = frame.shape
    largest = max(cols, rows)

    frame = cv2.resize(frame, [
        int(max_output_size / largest * rows),
        int(max_output_size / largest * cols)
    ])

    del cols, rows, largest

    return frame
