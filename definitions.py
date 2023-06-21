import cv2
import numpy as np
import os

from pathlib import Path

box_thickness = 1
box_color = (0, 0, 255)
font = cv2.FONT_HERSHEY_DUPLEX

max_processes = 4
samples_dir = "./assets/models"
outputs_dir = "./assets/outputs"
pkl_file = Path("./assets/outputs/models.pkl")

default_match_name = None

thumbnail_size = 240
max_output_size = 720


FaceLocation = tuple[int, int, int, int]
FaceEncoding = list[np.array]


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


class RecognitionModel:
    def __init__(self, id: int, encoding: FaceEncoding, location: FaceLocation, name: str, accuracy: float, path: str | None = None):
        self.id = id
        self.encoding = encoding
        self.location = location
        self.name = name
        self.accuracy = accuracy
        self.path = path

    def save_path(self) -> str:
        dir = os.path.join(outputs_dir, self.name)
        path = os.path.join(dir, f"{self.id:05}.jpg")
        return dir, path

    def __str__(self) -> str:
        return f'\n#{self.id} {self.name}\n\taccuracy:{int(self.accuracy * 100):02}%\n\tpath:{self.path}\n'


class SingletonMeta(type):
    """
    The Singleton class can be implemented in different ways in Python. Some
    possible methods include: base class, decorator, metaclass. We will use the
    metaclass because it is best suited for this purpose.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]
