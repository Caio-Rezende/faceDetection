import cv2

from definitions import RecognitionModel, standardize_frame, FaceLocation, box_thickness, box_color, font


def draw_face_image(model: RecognitionModel) -> str:
    img = cv2.imread(model.path)
    if img is None:
        return

    (resize_factor, frame) = standardize_frame(img)
    del img

    draw_box_face(frame, [int(pos * resize_factor)
                  for pos in model.location], model.name)

    window_name = f"FD #{model.id} ({model.name}) src: {model.path}"
    cv2.imshow(window_name, frame)

    del resize_factor, frame
    return window_name


def draw_box_face(frame: cv2.Mat, face_location: FaceLocation, name: str | None):
    (top, right, bottom, left) = face_location

    cv2.rectangle(frame, (left, top), (right, bottom),
                  box_color, box_thickness)

    if not name is None and name.count('unknown') == 0:
        cv2.putText(frame, name, (left + 6, bottom - 6),
                    font, 1.0, (255, 255, 255), 1)

    del top, right, bottom, left
