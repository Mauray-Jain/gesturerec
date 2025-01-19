import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import pandas as pd

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style()
    )

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv.LINE_AA)

  return annotated_image

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

video = cv.VideoCapture(0)
video.set(cv.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
data = []
gesname = input("Enter gesture name: ")

while True:
    success, frame = video.read()
    if not success:
        break

    y,x,_ = frame.shape
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv.cvtColor(frame, cv.COLOR_BGR2RGB))

    detection_result = detector.detect(image)

    annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
    cv.imshow("landmarks", cv.cvtColor(annotated_image, cv.COLOR_RGB2BGR))

    k = cv.waitKey(1) & 0xff
    if k == 27: # esc
        break
    elif k == 115: # s pressed capture data point
        hand_landmarks_list = detection_result.hand_landmarks
        handedness_list = detection_result.handedness
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            handedness = handedness_list[idx]
            x_coordinates = [landmark.x for landmark in hand_landmarks]
            y_coordinates = [landmark.y for landmark in hand_landmarks]

            datapt = {
                "gesture": gesname,
                "handedness": handedness[0].category_name,
            }
            for j in range(len(hand_landmarks)):
                datapt[f"x{j}"] = hand_landmarks[j].x * x
                datapt[f"y{j}"] = hand_landmarks[j].y * y

            data.append(datapt)

new = pd.DataFrame(data)
print(new)

yn = input("y/n? ")
if yn == "y":
    old = pd.read_csv("gestures.csv")
    naya = pd.concat([old, new], ignore_index=True)
    naya.to_csv("gestures.csv", index=False)

detector.close()
cv.destroyAllWindows()
video.release()
