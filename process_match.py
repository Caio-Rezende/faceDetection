import cv2
import numpy as np

import face_recognition
import process_models
from definitions import FaceLocation, FaceEncoding

default_name = None
font = cv2.FONT_HERSHEY_DUPLEX

box_thickness = 2
box_color = (0, 0, 255)

models = process_models.load_models()
known_models = [model.encoding for model in models]


def call(frame: cv2.Mat):
    face_locations = face_recognition.face_locations(frame, 1, "cnn")
    face_encodings = face_recognition.face_encodings(
        frame, face_locations)

    for (face_location, face_encoding) in zip(face_locations, face_encodings):
        name = get_match_name(face_encoding)

        draw_box_face(frame, face_location, name)

        del name

    del face_locations, face_encodings


def get_match_name(face_encoding: FaceEncoding):
    name = default_name

    matches = face_recognition.compare_faces(
        known_models, face_encoding, tolerance=0.45)
    face_distances = face_recognition.face_distance(
        known_models, face_encoding)

    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        name = models[best_match_index].name

    del matches, face_distances, best_match_index

    return name


def draw_box_face(frame: cv2.Mat, face_location: FaceLocation, name: str | None):
    (top, right, bottom, left) = face_location

    cv2.rectangle(frame, (left, top), (right, bottom),
                  box_color, box_thickness)

    if not name is None:
        cv2.putText(frame, name, (left + 6, bottom - 6),
                    font, 1.0, (255, 255, 255), 1)

    del top, right, bottom, left
