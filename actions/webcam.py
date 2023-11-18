import cv2

import process.image as image

from definitions import standardize_frame
from process.args import get_args

args = get_args()


def call():
    webcam = cv2.VideoCapture(0)

    while True:
        # input from camera
        check, frame = webcam.read()

        if not check:
            break

        del check

        factor, standardFrame = standardize_frame(frame, 512)

        del frame

        output = image.call(standardFrame, 'webcam')
        del standardFrame

        cv2.imshow(f"FD (Webcam)", output)

        del output

        if cv2.waitKey(100) > -1:
            break

    webcam.release()
    del webcam

    cv2.destroyAllWindows()
