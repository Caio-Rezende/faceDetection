import cv2

import process_image
from process_parser import get_args

args = get_args()


def call():
    webcam = cv2.VideoCapture(0)

    while True:
        # input from camera
        check, frame = webcam.read()

        if not check:
            break

        del check

        output = process_image.call(frame)
        del frame

        cv2.imshow(f"FD (Webcam)", output)

        del output

        if cv2.waitKey(100) > -1:
            break

    webcam.release()
    del webcam

    cv2.destroyAllWindows()
