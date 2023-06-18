from pathlib import Path
import definitions
import dlib
import face_recognition
#import multiprocessing
import os
import pickle
import time

Model = definitions.RecognitionModel

max_processes = 4
samples_dir = "./assets/models"
pkl_file = Path("./assets/models/models.pkl")


def load_models(remake: bool = False) -> list[Model]:
    if not remake and os.path.isfile(pkl_file):
        return pickle.load(pkl_file.open(mode='rb'))

    print("devices: {}, cuda: {}".format(
        dlib.cuda.get_num_devices(), dlib.DLIB_USE_CUDA))
    start = time.time()

    models = []
    get_files(samples_dir, models)
    pickle.dump(models, pkl_file.open(mode='wb'))

    end = time.time()

    print("total load models time elapsed {} ms".format((end-start)*1000))

    del start, end

    return models


def get_files(dir: str, models: list[Model], parent: str = None):
    list_dir = os.listdir(dir)

    # if (not parent is None):
    #    pool = multiprocessing.Pool(processes=min(
    #        max_processes, len(list_dir)))
    #    pool.starmap(process_dir, [(dir, models, name,
    #                                parent) for name in list_dir])
    #    del pool
    # else:
    [process_dir(dir, models, name,
                 parent) for name in list_dir]

    del list_dir


def process_dir(dir: str, models: list[Model], name: str, parent: str = None):
    path = os.path.join(dir, name)

    if path.count(".pkl") > 0:
        del path
        return

    if os.path.isfile(path):
        encoding = get_encoding(path)
        if not encoding is None:
            models.append(
                Model(
                    encoding=encoding,
                    name=name.replace(
                        ".jpg", "") if parent is None else parent,
                    file=path,
                )
            )
        del path, encoding
        return

    if os.path.isdir(path):
        get_files(path, models, name)
        del path
        return


def get_encoding(file: str):
    start = time.time()

    img = face_recognition.load_image_file(file)
    locations = face_recognition.face_locations(img, 1, "cnn")
    encodings = face_recognition.face_encodings(img, locations, 1, "large")

    end = time.time()

    print("load model {} time elapsed {} ms".format(file, (end-start)*1000))

    del start, img, locations, end

    return encodings[0] if len(encodings) > 0 else None


def remake():
    load_models(True)
