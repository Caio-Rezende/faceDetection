import cv2
import mediapipe as mp
import numpy as np
import typing

import face_recognition
import process_models

drawing = mp.solutions.drawing_utils
_normalized_to_pixel_coordinates = drawing._normalized_to_pixel_coordinates

default_name = "unkown"
font = cv2.FONT_HERSHEY_DUPLEX


def call(frame: cv2.Mat, face: typing.NamedTuple):
    if face.location_data is None or not face.location_data.HasField('relative_bounding_box'):
        return

    cols, rows, _ = frame.shape
    largest = max(cols, rows)

    relative = face.location_data.relative_bounding_box
    try:
        x, y = _normalized_to_pixel_coordinates(
            relative.xmin,
            relative.ymin,
            cols, rows
        )
        cv2.putText(frame, get_match_name(frame, x, y), (x,
                                                      y+12), font, largest/1000, (255, 255, 255), 3)
        del x, y
    except:
        pass

    del cols, rows, largest, relative


def get_match_name(frame: cv2.Mat, x: int, y: int):
    models = process_models.load_models()

    name = default_name

    cols, rows, _ = frame.shape
    face_locations = face_recognition.api._trim_css_to_bounds(
        [x, y + cols, x + rows, y], frame.shape)
    face_encodings = face_recognition.face_encodings(frame, [face_locations], 1, "large")

    known_models = [model.encoding for model in models]
    matches = face_recognition.compare_faces(
        known_models, face_encodings[0], tolerance=0.45)
    face_distances = face_recognition.face_distance(
        known_models, face_encodings[0])

    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        name = models[best_match_index].name

    del models, cols, rows, face_locations, face_encodings, known_models, matches, face_distances, best_match_index

    return name
