import cv2

from definitions import standardize_frame
from models_loader import loader
from process.args import get_args
from process.image import draw_box_face

args = get_args()


def call(indices: int):
    global models
    maxIndex = len(indices) - 1

    for index, modelIndex in enumerate(indices):
        if len(models) >= modelIndex:
            read_image(modelIndex)

            if index == maxIndex or (args.slow and (index + 1) % 10 == 0):
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    del models


def read_image(index: int) -> None:
    models = loader.load()
    model = models[index]
    del models

    img = cv2.imread(model.path)
    if img is None:
        return

    (resize_factor, frame) = standardize_frame(img)
    del img

    draw_box_face(frame, [int(pos * resize_factor)
                  for pos in model.location], model.name)

    cv2.imshow(f"FD #{index} ({model.name}) src: {model.path}", frame)

    del resize_factor, frame, model
