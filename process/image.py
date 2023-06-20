import cv2

import process.match as process_match

from definitions import standardize_frame, FaceLocation, box_thickness, box_color, font
from process.args import get_args

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


def draw_box_face(frame: cv2.Mat, face_location: FaceLocation, name: str | None):
    (top, right, bottom, left) = face_location

    cv2.rectangle(frame, (left, top), (right, bottom),
                  box_color, box_thickness)

    if not name is None and name.count('unknown') == 0:
        cv2.putText(frame, name, (left + 6, bottom - 6),
                    font, 1.0, (255, 255, 255), 1)

    del top, right, bottom, left
