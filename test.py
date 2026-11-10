import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(
        # preparation de model ia
        model_asset_path="hand_landmarker.task"
    ),#recuperation des resultats de la detection de camera 
    running_mode=vision.RunningMode.VIDEO
)
#creation de l'objet detector pour la detection de la main
detector = vision.HandLandmarker.create_from_options(options)