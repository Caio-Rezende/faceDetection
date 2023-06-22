# import multiprocessing
import cv2
import os
import time

from actions.print import verbose
from definitions import RecognitionModel, samples_dir
from models_loader import loader
from process.args import get_args
from process.match import frame_match
from process.models import append_model, create_model

args = get_args()


def call() -> list[RecognitionModel]:
    start = time.time()

    models = loader.load() if not args.remake else []
    process_files_dir(samples_dir, models)
    loader.save(models)

    end = time.time()

    verbose("total load models time elapsed {} ms".format(int(end-start)*1000))

    del start, end

    return models


def process_files_dir(dir: str, models: list[RecognitionModel], parent: str = None):
    list_dir = os.listdir(dir)

    # if (not parent is None):
    #    pool = multiprocessing.Pool(processes=min(
    #        max_processes, len(list_dir)))
    #    pool.starmap(process_dir, [(dir, models, name,
    #                                parent) for name in list_dir])
    #    del pool
    # else:
    [process_dir_entry(dir, models, name,
                       parent) for name in list_dir]

    del list_dir


def process_dir_entry(dir: str, models: list[RecognitionModel], name: str, parent: str = None):
    path = os.path.join(dir, name)

    if os.path.isfile(path):
        if not path.count('.jpg'):
            del path
            return

        match_first(path, models, name, parent)
        del path
        return

    if os.path.isdir(path):
        process_files_dir(path, models, name)
        del path
        return


def match_first(path: str, models: list[RecognitionModel], name: str, parent: str = None):
    start = time.time()

    img = cv2.imread(path)
    if img is None:
        return

    frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    del img

    face_locations, face_encodings = frame_match(frame)

    end = time.time()

    verbose("load model {} time elapsed {} ms".format(
        path, int(end-start)*1000))

    if len(face_encodings) > 0:
        append_model(
            models,
            create_model(
                id=len(models),
                encoding=face_encodings[0],
                location=face_locations[0],
                name=name.replace(
                    ".jpg", "") if parent is None else parent,
                path=path,
                accuracy=1
            ),
            frame[:, :, ::-1]
        )

    del start, frame, face_locations, face_encodings, end
