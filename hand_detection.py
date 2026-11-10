
import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Model path
MODEL_PATH = "hand_landmarker.task"

# Configure Hand Landmarker
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

# Create detector
detector = vision.HandLandmarker.create_from_options(options)

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

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Create MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Detect hands
    results = detector.detect_for_video(
        mp_image,
        timestamp
    )

    timestamp += 33

    # Draw landmarks
    if results.hand_landmarks:

        for hand_landmarks in results.hand_landmarks:

            height, width, _ = frame.shape

            # Draw 21 points
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

            # Draw connections between fingers
            connections = vision.HandLandmarksConnections.HAND_CONNECTIONS

            for connection in connections:
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

        cv2.putText(
            frame,
            "HAND DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    cv2.imshow("AI Gesture Control", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
detector.close()
cv2.destroyAllWindows()
