from pathlib import Path
import dlib
import face_recognition
# import multiprocessing
import os
import pickle
import time
import cv2

from definitions import RecognitionModel, standardize_frame

max_processes = 4
samples_dir = "./assets/models"
outputs_dir = "./assets/outputs"
pkl_file = Path("./assets/outputs/models.pkl")
thumbnail_size = 240


def load_models(remake: bool = False) -> list[RecognitionModel]:
    if not remake and os.path.isfile(pkl_file):
        return pickle.load(pkl_file.open(mode='rb'))

    print("devices: {}, cuda: {}".format(
        dlib.cuda.get_num_devices(), dlib.DLIB_USE_CUDA))
    start = time.time()

    models = []
    get_files(samples_dir, models)
    update_models(models)

    end = time.time()

    print("total load models time elapsed {} ms".format(int(end-start)*1000))

    del start, end

    return models


def update_models(models: list[RecognitionModel]):
    pickle.dump(models, pkl_file.open(mode='wb'))


def get_files(dir: str, models: list[RecognitionModel], parent: str = None):
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


def process_dir(dir: str, models: list[RecognitionModel], name: str, parent: str = None):
    path = os.path.join(dir, name)

    if path.count(".pkl") > 0:
        del path
        return

    if os.path.isfile(path):
        match_first(path, models, name, parent)
        del path
        return

    if os.path.isdir(path):
        get_files(path, models, name)
        del path
        return


def append_model(models: list[RecognitionModel], model: RecognitionModel, frame: cv2.Mat):
    if any([a.name == model.name and a.path == model.path for a in models]):
        return

    (top, right, bottom, left) = model.location
    dir, path = save_path(len(models), model.name)

    cropped = frame[top:bottom, left:right]
    (resize_factor, resized) = standardize_frame(cropped, thumbnail_size)

    if not os.path.exists(dir):
        os.mkdir(dir)
    cv2.imwrite(path, resized)

    models.append(model)
    del top, right, bottom, left, dir, path, cropped, resize_factor, resized


def match_first(path: str, models: list[RecognitionModel], name: str, parent: str = None):
    start = time.time()

    img = cv2.imread(path)
    if img is None:
        return

    frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    del img

    locations = face_recognition.face_locations(frame, 1, "cnn")
    encodings = face_recognition.face_encodings(frame, locations)

    end = time.time()

    print("load model {} time elapsed {} ms".format(path, int(end-start)*1000))

    if len(encodings) > 0:
        append_model(
            models,
            RecognitionModel(
                encoding=encodings[0],
                location=locations[0],
                name=name.replace(
                    ".jpg", "") if parent is None else parent,
                path=path,
            ),
            frame[:, :, ::-1]
        )

    del start, frame, locations, encodings, end


def models_fix(index: int, name: str):
    models = load_models()
    old_dir, old_path = save_path(index, models[index].name)
    models[index].name = name
    new_dir, new_path = save_path(index, name)

    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    os.rename(old_path, new_path)

    update_models(models)

    if len(os.listdir(old_dir)) == 0:
        try:
            os.remove(old_dir)
        except:
            print(f'!! failed to delete empty folder ({old_dir}) !!')
            pass

    del old_path, old_dir, new_path, new_dir, models


def save_path(index: int, name: str):
    dir = os.path.join(outputs_dir, name)
    path = os.path.join(dir, f"{index:03}.jpg")
    return dir, path


def remake():
    load_models(True)
