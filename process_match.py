import cv2
import numpy as np

import face_recognition
import process_models
from definitions import FaceEncoding, RecognitionModel
from process_models import append_model, update_models
from process_parser import get_args

args = get_args()

default_name = None

models = process_models.load_models()
known_models = [model.encoding for model in models]


def call(frame: cv2.Mat, path: str) -> list[RecognitionModel]:
    global models, known_models
    matches: list[RecognitionModel] = []

    face_locations = face_recognition.face_locations(frame, 1, "cnn")
    face_encodings = face_recognition.face_encodings(
        frame, face_locations)

    for (face_location, face_encoding) in zip(face_locations, face_encodings):
        name = get_match_name(face_encoding)
        model = RecognitionModel(face_encoding, face_location, name, path)
        matches.append(model)

        if (not name is None) and args.save:
            append_model(models, model, frame[:, :, ::-1])
        else:
            if args.unknown:
                unkowns = set([a.name for a in
                               filter(lambda a: a.name.count('unknown') > 0, models)])
                if len(unkowns) == 0:
                    unknownIndex = 0
                else:
                    unknownIndex = 1 + max(
                        [int(unknown.replace('unknown-', '')) for unknown in unkowns])
                name = f'unknown-{unknownIndex:03}'
                model.name = name
                append_model(models, model, frame[:, :, ::-1])
                del unkowns, unknownIndex

        del name, model

    del face_locations, face_encodings

    if len(known_models) != len(models):
        update_models(models)
        known_models = [model.encoding for model in models]

    return matches


def get_match_name(face_encoding: FaceEncoding):
    global models, known_models

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
