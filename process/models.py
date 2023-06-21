# import multiprocessing
import cv2
import os

from actions.print import verbose
from definitions import RecognitionModel, standardize_frame, FaceEncoding, FaceLocation, outputs_dir, thumbnail_size, default_match_name
from models_loader import loader
from process.args import get_args

args = get_args()


def append_model(models: list[RecognitionModel], model: RecognitionModel, frame: cv2.Mat):
    if any([a.name == model.name and a.path == model.path for a in models]):
        return

    if model.name == default_match_name:
        prepare_unknown(models, model)

    (top, right, bottom, left) = model.location
    dir, path = model.save_path()

    cropped = frame[top:bottom, left:right]
    (resize_factor, resized) = standardize_frame(cropped, thumbnail_size)

    if not os.path.exists(dir):
        os.mkdir(dir)
    cv2.imwrite(path, resized)

    models.append(model)
    verbose(model)
    del top, right, bottom, left, dir, path, cropped, resize_factor, resized


def prepare_unknown(models: list[RecognitionModel], model: RecognitionModel):
    unkowns = set([a.name for a in
                   filter(lambda a: a.name.count('unknown') > 0, models)])
    if len(unkowns) == 0:
        unknownIndex = 0
    else:
        unknownIndex = 1 + max(
            [int(unknown.replace('unknown-', '')) for unknown in unkowns])
    model.name = f'unknown-{unknownIndex:05}'
    del unkowns, unknownIndex


def models_fix(index: int, name: str):
    models = loader.load()
    old_dir, old_path = models[index].save_path()
    models[index].name = name
    models[index].accuracy = 1
    new_dir, new_path = models[index].save_path()

    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    os.rename(old_path, new_path)

    loader.save(models)

    if len(os.listdir(old_dir)) == 0:
        try:
            os.remove(old_dir)
        except:
            print(f'!! failed to delete empty folder ({old_dir}) !!')
            pass

    del old_path, old_dir, new_path, new_dir, models


def clear(name: str, filtered: list[RecognitionModel], models: list[RecognitionModel]):
    dir = os.path.join(outputs_dir, name)
    for file in os.listdir(dir):
        path = os.path.join(dir, file)
        try:
            os.remove(path)
        except:
            print(f'!! failed to delete file ({path}) !!')
            pass
        del path

    if len(os.listdir(dir)) == 0:
        try:
            os.remove(dir)
        except:
            print(f'!! failed to delete empty folder ({dir}) !!')
            pass
    del dir

    loader.save([m for m in models if m not in filtered])


def create_model(encoding: FaceEncoding, location: FaceLocation, name: str, accuracy: float, path: str | None = None, id: int | None = None) -> RecognitionModel:
    models = loader.load()

    if id is None:
        id = 0

        if len(models) > 0:
            id = max([m.id for m in models]) + 1

    return RecognitionModel(id=id, encoding=encoding, location=location, name=name, path=path, accuracy=accuracy)
