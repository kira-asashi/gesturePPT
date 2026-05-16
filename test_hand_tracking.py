import cv2
import mediapipe as mp
import time
import pyautogui
import math

def distance_between_points(point1, point2, width, height):
    x1 = int(point1.x * width)
    y1 = int(point1.y * height)

    x2 = int(point2.x * width)
    y2 = int(point2.y * height)

    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "hand_landmarker.task"

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera could not be opened.")
    exit()

last_action_time = 0
cooldown = 1.0

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        success, frame = cap.read()

        if not success:
            print("Failed to read from camera.")
            break

        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp_ms = int(time.time() * 1000)

        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                height, width, _ = frame.shape

                thumb_tip = hand_landmarks[4]
                index_tip = hand_landmarks[8]
                middle_tip = hand_landmarks[12]

                index_pinch_distance = distance_between_points(
                    thumb_tip, index_tip, width, height
                )

                index_pinchOut_distance = distance_between_points(
                    thumb_tip, index_tip, width, height
                )

                current_time = time.time()

                if current_time - last_action_time > cooldown:
                    if index_pinch_distance < 40:
                        pyautogui.hotkey("ctrl", "+")
                        print("Zoom in")
                        last_action_time = current_time

                    elif index_pinchOut_distance > 40:
                        pyautogui.hotkey("ctrl", "-")
                        print("Zoom out")
                        last_action_time = current_time

                # Draw landmark points
                for landmark in hand_landmarks:
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                # Draw simple hand connections manually
                connections = [
                    (0, 1), (1, 2), (2, 3), (3, 4),
                    (0, 5), (5, 6), (6, 7), (7, 8),
                    (0, 9), (9, 10), (10, 11), (11, 12),
                    (0, 13), (13, 14), (14, 15), (15, 16),
                    (0, 17), (17, 18), (18, 19), (19, 20),
                    (5, 9), (9, 13), (13, 17)
                ]

                for start, end in connections:
                    x1 = int(hand_landmarks[start].x * width)
                    y1 = int(hand_landmarks[start].y * height)
                    x2 = int(hand_landmarks[end].x * width)
                    y2 = int(hand_landmarks[end].y * height)
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        cv2.imshow("New MediaPipe Hand Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()