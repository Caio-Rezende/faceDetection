import os
import cv2

import process_image

samples_dir = "./assets/samples"


def read_image(path: str, name: str) -> None:
    img = cv2.imread(path)
    if img is None:
        return

    output = process_image.call(img)
    cv2.imshow(f"FD ({name})", output)


def call():
    for name in os.listdir(samples_dir):
        path = os.path.join(samples_dir, name)
        if os.path.isfile(path):
            read_image(path, name)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
