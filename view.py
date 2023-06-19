import cv2

import process_models
from process_image import draw_box_face
from definitions import standardize_frame
from process_parser import get_args

args = get_args()
models = process_models.load_models()


def read_image(index: int) -> None:
    global models

    model = models[index]
    img = cv2.imread(model.path)
    if img is None:
        return

    (resize_factor, frame) = standardize_frame(img)
    del img

    draw_box_face(frame, [int(pos * resize_factor)
                  for pos in model.location], model.name)

    cv2.imshow(f"FD #{index} ({model.name}) src: {model.path}", frame)

    del resize_factor, frame, model


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
