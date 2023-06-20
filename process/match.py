import cv2
import numpy as np

import face_recognition

from definitions import FaceLocation, FaceEncoding, RecognitionModel, default_match_name
from models_loader import loader
from process.args import get_args
from process.models import append_model,  create_model

args = get_args()


def call(frame: cv2.Mat, path: str) -> list[RecognitionModel]:
    models = loader.load()
    modelsSize = len(models)

    matches: list[RecognitionModel] = []

    face_locations, face_encodings = frame_match(frame)

    for (face_location, face_encoding) in zip(face_locations, face_encodings):
        accuracy, name = get_match_name(face_encoding)
        model = create_model(encoding=face_encoding, location=face_location,
                             name=name, accuracy=1-accuracy, path=path)
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
                name = f'unknown-{unknownIndex:05}'
                model.name = name
                append_model(models, model, frame[:, :, ::-1])
                del unkowns, unknownIndex

        del name, model

    del face_locations, face_encodings

    if modelsSize != len(models):
        loader.save(models)

    del models, modelsSize

    return matches


def get_match_name(face_encoding: FaceEncoding) -> tuple[float, str | None]:
    models = loader.load()
    known_models = [model.encoding for model in models]

    accuracy = 0
    name = default_match_name

    matches = face_recognition.compare_faces(
        known_models, face_encoding, tolerance=args.tolerance)
    face_distances = face_recognition.face_distance(
        known_models, face_encoding)

    best_match_index = np.argmin(face_distances)

    if matches[best_match_index]:
        accuracy = face_distances[best_match_index]
        name = models[best_match_index].name

    del matches, face_distances, best_match_index, models, known_models

    return accuracy, name


def frame_match(frame: cv2.Mat) -> tuple[FaceLocation, FaceEncoding]:
    face_locations = face_recognition.face_locations(frame, 1, args.model)
    face_encodings = face_recognition.face_encodings(
        frame, face_locations)

    return face_locations, face_encodings
