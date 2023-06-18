import cv2
import mediapipe as mp

import process_match

fd = mp.solutions.face_detection
drawing = mp.solutions.drawing_utils
DrawingSpec = drawing.DrawingSpec

max_output_size = 720
fdInstance = fd.FaceDetection()
keys = DrawingSpec(
    thickness=0
)


def call(frame: cv2.Mat) -> cv2.Mat:
    cols, rows, _ = frame.shape
    largest = max(cols, rows)

    face_list = fdInstance.process(frame)

    if face_list.detections:
        for face in face_list.detections:
            process_match.call(frame, face)
            draw_box(frame, face)

    frame = cv2.resize(frame, [
        int(max_output_size / largest * rows),
        int(max_output_size / largest * cols)
    ])

    del cols, rows, largest, face_list

    return frame


def draw_box(frame, face):
    cols, rows, _ = frame.shape
    largest = max(cols, rows)

    box = DrawingSpec(
        thickness=int(largest * 0.005)
    )

    drawing.draw_detection(frame, face, keys, box)

    del cols, rows, largest, box
