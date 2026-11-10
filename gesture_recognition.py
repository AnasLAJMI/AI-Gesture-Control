
import cv2
import mediapipe as mp

from collections import deque, Counter

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "hand_landmarker.task"

# Configure MediaPipe
base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)

detector = vision.HandLandmarker.create_from_options(options)


def recognize_gesture(landmarks):

    # Detect four fingers
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]

    fingers = []

    for tip, pip in zip(tips, pips):
        if landmarks[tip].y < landmarks[pip].y:
            fingers.append(1)
        else:
            fingers.append(0)

    # Detect thumb
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]

    if thumb_tip.x < thumb_ip.x:
        thumb = 1
    else:
        thumb = 0

    # Count all five fingers
    count = sum(fingers) + thumb

    if count == 5:
        return "OPEN HAND"

    elif count == 0:
        return "FIST"

    elif fingers == [1, 0, 0, 0] and thumb == 0:
        return "INDEX FINGER"

    elif fingers == [1, 1, 0, 0]:
        return "PEACE"

    else:
        return "UNKNOWN"
# Gesture stabilization
gesture_histories = {}

def stabilize_gesture(gesture, hand_label):

    history = gesture_histories.setdefault(hand_label, deque(maxlen=5))
    history.append(gesture)

    counts = Counter(history)

    stable_gesture, frequency = counts.most_common(1)[0]

    if frequency >= 4:
        return stable_gesture

    return "DETECTING..."


# Open camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not detected!")
    detector.close()
    exit()

timestamp = 0

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

    if timestamp % 1000 < 33:
       print("Number of hands:", len(results.hand_landmarks), flush=True)

    timestamp += 33

    if results.hand_landmarks:
        detected_hands = set()

        for hand_index, hand_landmarks in enumerate(results.hand_landmarks):
            hand_label = results.handedness[hand_index][0].category_name
            detected_hands.add(hand_label)
            gesture = recognize_gesture(hand_landmarks)

            stable_gesture = stabilize_gesture(gesture, hand_label)

            height, width, _ = frame.shape

            # Draw landmarks
            for landmark in hand_landmarks:

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )

            # Draw connections
            for connection in vision.HandLandmarksConnections.HAND_CONNECTIONS:

                start = hand_landmarks[connection.start]
                end = hand_landmarks[connection.end]

                x1 = int(start.x * width)
                y1 = int(start.y * height)

                x2 = int(end.x * width)
                y2 = int(end.y * height)

                cv2.line(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    2
                )

            # Display each hand's label and stable gesture on its own line
            cv2.putText(
                frame,
                f"{hand_label}: {stable_gesture}",
                (20, 60 + hand_index * 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

        for hand_label in set(gesture_histories) - detected_hands:
            gesture_histories[hand_label].clear()

    else:
        for history in gesture_histories.values():
            history.clear()

        cv2.putText(
            frame,
            "NO HAND DETECTED",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    cv2.imshow("AI Gesture Recognition - Improved", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
detector.close()
cv2.destroyAllWindows()
