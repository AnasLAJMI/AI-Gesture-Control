
import cv2
import mediapipe as mp
import pyautogui
import pygetwindow as gw
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =====================================
# CONFIGURATION
# =====================================

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.02

CAMERA_WINDOW = "AI Gesture Slide Controller"

MODEL_PATH = "hand_landmarker.task"

COOLDOWN = 1.0
last_action_time = 0

timestamp = 0


# =====================================
# MEDIAPIPE
# =====================================

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)

detector = vision.HandLandmarker.create_from_options(options)


# =====================================
# CAMERA
# =====================================

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Camera not detected!")
    detector.close()
    raise SystemExit


# =====================================
# GESTURE DETECTION
# =====================================

def detect_gesture(landmarks):

    index_up = landmarks[8].y < landmarks[6].y
    middle_up = landmarks[12].y < landmarks[10].y
    ring_up = landmarks[16].y < landmarks[14].y
    pinky_up = landmarks[20].y < landmarks[18].y

    fingers = [
        index_up,
        middle_up,
        ring_up,
        pinky_up
    ]

    # Peace: index + middle
    if index_up and middle_up and not ring_up and not pinky_up:
        return "NEXT"

    # Index only
    if index_up and not middle_up and not ring_up and not pinky_up:
        return "PREVIOUS"

    # Open hand
    if all(fingers):
        return "FULLSCREEN"

    # Fist
    if not any(fingers):
        return "EXIT_FULLSCREEN"

    return "UNKNOWN"


# =====================================
# ACTIVE WINDOW CHECK
# =====================================

def get_active_title():

    try:
        window = gw.getActiveWindow()

        if window:
            return window.title

    except Exception:
        pass

    return ""


def send_action(action):

    title = get_active_title()

    # Never send commands to our camera window
    if not title or CAMERA_WINDOW.lower() in title.lower():
        return "FOCUS PDF WINDOW!"

    if action == "NEXT":
        pyautogui.press("right")
        return "NEXT PAGE"

    elif action == "PREVIOUS":
        pyautogui.press("left")
        return "PREVIOUS PAGE"

    elif action == "FULLSCREEN":
        pyautogui.press("f11")
        return "FULLSCREEN"

    elif action == "EXIT_FULLSCREEN":
        pyautogui.press("esc")
        return "EXIT FULLSCREEN"

    return "UNKNOWN"


# =====================================
# MAIN PROGRAM
# =====================================

print("================================")
print("AI GESTURE SLIDE CONTROLLER")
print("================================")
print("Open your PDF first.")
print("Click the PDF window to make it active.")
print("Show gestures to control pages.")
print("Press Q to quit.")
print("================================")

cv2.namedWindow(CAMERA_WINDOW)

try:

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        results = detector.detect_for_video(
            mp_image,
            timestamp
        )

        timestamp += 33

        status = "SHOW YOUR HAND"
        gesture = "UNKNOWN"

        if results.hand_landmarks:

            landmarks = results.hand_landmarks[0]

            gesture = detect_gesture(landmarks)

            current_time = time.time()

            if current_time - last_action_time > COOLDOWN:

                if gesture != "UNKNOWN":

                    status = send_action(gesture)

                    # Cooldown prevents repeated commands
                    last_action_time = current_time

            else:
                status = "WAIT..."

            # Draw landmarks
            h, w, _ = frame.shape

            for point in landmarks:

                x = int(point.x * w)
                y = int(point.y * h)

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )

            # Draw connections
            for connection in vision.HandLandmarksConnections.HAND_CONNECTIONS:

                start = landmarks[connection.start]
                end = landmarks[connection.end]

                x1 = int(start.x * w)
                y1 = int(start.y * h)

                x2 = int(end.x * w)
                y2 = int(end.y * h)

                cv2.line(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    2
                )

        # Display information
        cv2.putText(
            frame,
            "Gesture: " + gesture,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Status: " + status,
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.imshow(CAMERA_WINDOW, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:

    cap.release()
    detector.close()
    cv2.destroyAllWindows()

    print("Gesture Slide Controller stopped.")
