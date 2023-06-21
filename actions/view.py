import cv2

from models_loader import loader
from process.args import get_args
from process.draw import draw_face_image

args = get_args()


def call(indices: list[int]):
    models = loader.load()

    maxIndex = len(indices) - 1

    for index, modelIndex in enumerate(indices):
        if len(models) >= modelIndex:
            draw_face_image(models[modelIndex])

            if index == maxIndex or (args.slow and (index + 1) % 10 == 0):
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    del models
