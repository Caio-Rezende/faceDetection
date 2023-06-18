import cv2

import process_match

from definitions import standardize_frame


def call(frame: cv2.Mat) -> cv2.Mat:
    frame = standardize_frame(frame)

    process_match.call(frame)

    return frame
