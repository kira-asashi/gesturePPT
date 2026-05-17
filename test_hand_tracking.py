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
    min_hand_detection_confidence=0.8,
    min_hand_presence_confidence=0.8,
    min_tracking_confidence=0.8,
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Camera could not be opened.")
    exit()

last_action_time = 0
cooldown = 1.0

screen_width, screen_height = pyautogui.size()

prev_mouse_x, prev_mouse_y = 0, 0
smoothening = 7

pyautogui.FAILSAFE = False

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
                ring_tip = hand_landmarks[16]
                pinky_tip = hand_landmarks[20]

                index_x = int(index_tip.x * width)
                index_y = int(index_tip.y * height)

                index_ThumbToIndex_distance = distance_between_points( #pinching
                    thumb_tip, index_tip, width, height
                )

                index_IndexToMiddle_distance = distance_between_points(
                    index_tip, middle_tip, width, height
                )

                index_ThumbToMiddle_distance = distance_between_points(
                    hand_landmarks[9], thumb_tip, width, height
                )

                index_ThumbToWrist_distance = distance_between_points(
                    hand_landmarks[0],thumb_tip, width, height
                )

                index_IndexToWrist_distance = distance_between_points(
                    hand_landmarks[0],index_tip, width, height
                )

                index_MiddleToWrist_distance = distance_between_points(
                    hand_landmarks[0], index_tip, width, height
                )

                index_RingToWrist_distance = distance_between_points(
                    hand_landmarks[0],ring_tip, width, height
                )
                
                index_PinkyToWrist_distance = distance_between_points(
                    hand_landmarks[0],pinky_tip, width, height
                )

                index_ThumbToKnuckle_distance = distance_between_points(
                    hand_landmarks[5], thumb_tip, width, height
                )

                index_IndexToKnuckle_distance = distance_between_points(
                    hand_landmarks[5], index_tip, width, height
                )

                index_MiddleToKnuckle_distance = distance_between_points(
                    hand_landmarks[9], middle_tip, width, height
                )

                index_RingToKnuckle_distance = distance_between_points(
                    hand_landmarks[13], ring_tip, width, height
                )

                index_PinkyToKnuckle_distance = distance_between_points(
                    hand_landmarks[17], pinky_tip, width, height
                )

                current_time = time.time()

                # Gesture 1: Thumb + index pinch, other fingers up = zoom in
                if current_time - last_action_time > cooldown:
                    if (index_ThumbToIndex_distance < 40 and 
                        index_MiddleToKnuckle_distance > 40 and 
                        index_RingToKnuckle_distance > 40 and
                        index_PinkyToKnuckle_distance > 40):

                        pyautogui.hotkey("ctrl", "+")
                        print("Zoom in")
                        last_action_time = current_time

                    # Gesture 2: Index open, other fingers folded = zoom out
                    elif (index_ThumbToIndex_distance > 40 and index_ThumbToMiddle_distance > 40 and
                          index_IndexToKnuckle_distance > 40 and 
                          index_ThumbToKnuckle_distance > 40 and
                          index_MiddleToKnuckle_distance < 40 and 
                          index_RingToKnuckle_distance < 40 and
                          index_PinkyToKnuckle_distance < 40):

                        pyautogui.hotkey("ctrl", "-")
                        print("Zoom out")
                        last_action_time = current_time

                # Gesture 3: Only index and middle up placed together = move cursor
                if (index_IndexToKnuckle_distance > 40 and index_MiddleToKnuckle_distance > 40 and 
                    index_IndexToWrist_distance > 40 and index_MiddleToWrist_distance > 40 and 
                    index_IndexToMiddle_distance < 40 and
                    (index_ThumbToKnuckle_distance < 40 or index_ThumbToWrist_distance < 40) and 
                    (index_RingToKnuckle_distance < 40 or index_RingToWrist_distance < 40) and
                    (index_PinkyToKnuckle_distance < 40 or index_PinkyToWrist_distance < 40)):

                    # Correct mapping from camera coordinates to screen coordinates
                    screen_x = int(index_tip.x * screen_width)
                    screen_y = int(index_tip.y * screen_height)

                    curr_mouse_x = prev_mouse_x + (screen_x - prev_mouse_x) / smoothening
                    curr_mouse_y = prev_mouse_y + (screen_y - prev_mouse_y) / smoothening

                    pyautogui.moveTo(curr_mouse_x, curr_mouse_y)

                    prev_mouse_x, prev_mouse_y = curr_mouse_x, curr_mouse_y

                    cv2.circle(frame, (index_x, index_y), 12, (0, 0, 255), -1)
                    cv2.putText(
                        frame,
                        "Cursor Move Mode",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )

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
