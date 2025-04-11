import os
import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_bounding_box(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    annotated_image = np.copy(rgb_image)
    box = []

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        top_x = int(min(x_coordinates) * width) - MARGIN * 3
        top_y = int(min(y_coordinates) * height) - MARGIN * 3
        bot_x = int(max(x_coordinates) * width) + MARGIN * 3
        bot_y = int(max(y_coordinates) * height) + MARGIN * 3

        box.append((top_x, top_y, bot_x, bot_y))

        cv.rectangle(annotated_image, (top_x,top_y), (bot_x,bot_y), HANDEDNESS_TEXT_COLOR, FONT_THICKNESS*2)

    return (annotated_image, box)

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

video = cv.VideoCapture(0)
video.set(cv.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
li = 0
ri = 0
gesname = input("Enter gesture name: ")
os.chdir("test")

while True:
    success, frame = video.read()
    if not success:
        break

    y,x,_ = frame.shape
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv.cvtColor(frame, cv.COLOR_BGR2RGB))

    detection_result = detector.detect(image)

    (annotated_image, box) = draw_bounding_box(image.numpy_view(), detection_result)
    annotated_image = cv.cvtColor(annotated_image, cv.COLOR_RGB2BGR)
    cv.imshow("landmarks", annotated_image)

    k = cv.waitKey(1) & 0xff
    if k == 27: # esc
        break

    elif k == 115: # s pressed capture data point
        hand_landmarks_list = detection_result.hand_landmarks
        handedness_list = detection_result.handedness

        # Assuming ek hi haath hoga at a time for capturing data
        # for idx in range(len(hand_landmarks_list)):
        for (tx,ty,bx,by) in box:
            hand_landmarks = hand_landmarks_list[0]
            handedness = handedness_list[0]
            handedness = handedness[0].category_name
            folder = f"{gesname}_{handedness}"
            os.chdir(folder)
            filename = ""
            if handedness == "Left":
                filename = f"{li}"
                li += 1
            elif handedness == "Right":
                filename = f"{ri}"
                ri += 1
            filename += ".png"
            cv.imwrite(filename, frame[ty:by,tx:bx])
            print(f"Saved to {filename}")

detector.close()
cv.destroyAllWindows()
video.release()
