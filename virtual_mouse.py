
import cv2
import mediapipe as mp
import pyautogui
import math
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ==========================================
# 1. MOUSE CONFIGURATION
# ==========================================

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0

SCREEN_W, SCREEN_H = pyautogui.size()

# Cursor initial position
smooth_x = SCREEN_W // 2
smooth_y = SCREEN_H // 2

# Adaptive smoothing factors
SLOW_SMOOTH = 0.20
MEDIUM_SMOOTH = 0.30
FAST_SMOOTH = 0.45

previous_pinch = False
timestamp = 0


# ==========================================
# 2. MEDIAPIPE CONFIGURATION
# ==========================================

base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
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


# ==========================================
# 3. CAMERA CONFIGURATION
# ==========================================

cap = cv2.VideoCapture(0)

# Request a reasonable camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("ERROR: Camera not detected!")
    detector.close()
    exit()

print("================================")
print("AI VIRTUAL MOUSE STARTED")
print("================================")
print("Move INDEX finger to control mouse")
print("Pinch THUMB + INDEX to click")
print("Press Q to quit")
print("================================")


# ==========================================
# 4. MAIN LOOP
# ==========================================

try:

    while True:

        success, frame = cap.read()

        if not success:
            print("Camera frame error!")
            break

        # Mirror camera
        frame = cv2.flip(frame, 1)

        # Convert image to RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Detect hand
        results = detector.detect_for_video(
            mp_image,
            timestamp
        )

        timestamp += 33

        status = "SHOW YOUR HAND"

        # ==================================
        # 5. HAND DETECTION
        # ==================================

        if results.hand_landmarks:

            landmarks = results.hand_landmarks[0]

            h, w, _ = frame.shape

            # Finger detection
            index_up = (
                landmarks[8].y < landmarks[6].y
            )

            middle_up = (
                landmarks[12].y < landmarks[10].y
            )

            # Thumb and index fingertips
            thumb = landmarks[4]
            index = landmarks[8]

            # Pinch distance
            distance = math.sqrt(
                (thumb.x - index.x) ** 2 +
                (thumb.y - index.y) ** 2
            )

            pinch = distance < 0.045

            # ==================================
            # 6. VIRTUAL MOUSE MOVEMENT
            # ==================================

            if True:

                status = "MOUSE ACTIVE"

                # Convert camera coordinates to screen
                target_x = int(
                    (index.x - 0.1) / 0.8 * SCREEN_W
                )

                target_y = int(
                    (index.y - 0.1) / 0.8 * SCREEN_H
                )

                # Keep cursor inside screen
                target_x = max(
                    0,
                    min(SCREEN_W - 1, target_x)
                )

                target_y = max(
                    0,
                    min(SCREEN_H - 1, target_y)
                )

                # ==================================
                # 7. ADAPTIVE SMOOTHING
                # ==================================

                movement = math.sqrt(
                    (target_x - smooth_x) ** 2 +
                    (target_y - smooth_y) ** 2
                )

                if movement > 100:
                    smooth_factor = FAST_SMOOTH

                elif movement > 40:
                    smooth_factor = MEDIUM_SMOOTH

                else:
                    smooth_factor = SLOW_SMOOTH

                smooth_x += (
                    target_x - smooth_x
                ) * smooth_factor

                smooth_y += (
                    target_y - smooth_y
                ) * smooth_factor

                # Move cursor
                pyautogui.moveTo(
                    int(smooth_x),
                    int(smooth_y)
                )

                # ==================================
                # 8. PINCH CLICK
                # ==================================

                if pinch and not previous_pinch:

                    pyautogui.click()
                    status = "CLICK!"

                previous_pinch = pinch

            else:

                previous_pinch = False
                status = "INDEX FINGER TO MOVE"

            # ==================================
            # 9. DRAW HAND LANDMARKS
            # ==================================

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

            # Highlight fingertips
            cv2.circle(
                frame,
                (int(thumb.x * w), int(thumb.y * h)),
                10,
                (0, 0, 255),
                -1
            )

            cv2.circle(
                frame,
                (int(index.x * w), int(index.y * h)),
                10,
                (0, 0, 255),
                -1
            )

            # Show pinch distance
            cv2.putText(
                frame,
                f"Pinch: {distance:.3f}",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )

        else:

            previous_pinch = False

        # ==================================
        # 10. DISPLAY STATUS
        # ==================================

        cv2.putText(
            frame,
            status,
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Press Q to quit",
            (20, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "AI Virtual Mouse",
            frame
        )

        # Exit with Q
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


# ==========================================
# 11. CLEANUP
# ==========================================

finally:

    cap.release()
    detector.close()
    cv2.destroyAllWindows()

    print("AI Virtual Mouse stopped.")
