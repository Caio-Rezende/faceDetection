import cv2
import numpy as np
from PIL import ImageGrab

import process.image as image

from process.args import get_args

args = get_args()


def call():
    while True:
        # input from camera
        desktop = ImageGrab.grab(all_screens=True)

        frame = np.array(desktop)
        del desktop

        # Convert RGB to BGR
        frame = frame[:, :, ::-1].copy()

        output = image.call(frame, 'desktop')
        del frame

        cv2.imshow(f"FD (Desktop)", output)

        del output

        if cv2.waitKey(100) > -1:
            break

    cv2.destroyAllWindows()
