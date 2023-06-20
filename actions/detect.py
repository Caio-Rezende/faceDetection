import cv2
import os
import time

import process.image as image

from actions.print import verbose
from process.args import get_args

args = get_args()


def read_image(path: str, name: str) -> None:
    img = cv2.imread(path)
    if img is None:
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    output = image.call(img, path)
    del img

    if args.view:
        # from face_recognition format to cv2 format
        output = output[:, :, ::-1]

        cv2.imshow(f"FD ({name})", output)
    del output


def call():
    start = time.time()
    for dir in args.path:
        process_dir(dir)

    verbose(f'total time: {int(time.time()-start)*1000}ms')
    del start


def process_dir(dir: str):
    maxIndex = len(os.listdir(dir)) - 1

    for index, name in enumerate(os.listdir(dir)):
        path = os.path.join(dir, name)

        if os.path.isfile(path):
            start = time.time()
            read_image(path, name)
            verbose(
                f'\t reading file #{index + 1}/{maxIndex + 1} - {path} - {int(time.time()-start)*1000}ms')

        if os.path.isdir(path):
            process_dir(path)

        del path

        if index == maxIndex or (args.slow and (index + 1) % 10 == 0):
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    del maxIndex
