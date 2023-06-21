import cv2

import process.match as process_match

from definitions import standardize_frame
from process.args import get_args
from process.draw import draw_box_face

args = get_args()


def call(frame: cv2.Mat, path: str) -> cv2.Mat:
    matches = process_match.call(frame, path)

    if not args.view:
        del matches
        return frame

    (resize_factor, frame) = standardize_frame(frame)

    [draw_box_face(frame, [int(pos * resize_factor) for pos in match.location], match.name)
     for match in matches]

    del matches, resize_factor

    return frame
