import os
import cv2

import process_image

samples_dir = "./assets/samples"


def read_image(path: str, name: str) -> None:
    img = cv2.imread(path)
    if img is None:
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    output = process_image.call(img)

    # from face_recognition format to cv2 format
    output = output[:, :, ::-1]

    cv2.imshow(f"FD ({name})", output)


def call():
    for name in os.listdir(samples_dir):
        path = os.path.join(samples_dir, name)
        if os.path.isfile(path):
            read_image(path, name)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
