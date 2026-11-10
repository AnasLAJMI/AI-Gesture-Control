import mediapipe as mp

print("MediaPipe version:", mp.__version__)
print("MediaPipe location:", mp.__file__)
print("Has solutions:", hasattr(mp, "solutions"))
print("Available attributes:")

print(dir(mp))