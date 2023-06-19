import cv2
import os
import time

import process_image
from process_parser import get_args

args = get_args()


def read_image(path: str, name: str) -> None:
    img = cv2.imread(path)
    if img is None:
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    output = process_image.call(img, path)
    del img

    if args.view:
        # from face_recognition format to cv2 format
        output = output[:, :, ::-1]

        cv2.imshow(f"FD ({name})", output)
    del output


def call():
    start = time.time()
    for dir in args.path:
        maxIndex = len(os.listdir(dir)) - 1

        for index, name in enumerate(os.listdir(dir)):
            path = os.path.join(dir, name)

            print(
                f'\t reading file #{index}/{maxIndex + 1} - {path} - {int(time.time()-start)*1000}ms')

            if os.path.isfile(path):
                read_image(path, name)

            del path

            if index == maxIndex or (args.slow and (index + 1) % 10 == 0):
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        del maxIndex

    print(f'total time: {int(time.time()-start)*1000}ms')
    del start
