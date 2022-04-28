import cv2
import mediapipe as mp
import numpy as np
import os
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def generate_truth_data(directory):
  """Generates coordinates for training data (pictures we take)

  Args:
      directory (string): path to the directory holding the training images

  Returns:
      np.array: nx21x2 np array, with n being number of images
  """
  with mp_hands.Hands(
      static_image_mode=True,
      max_num_hands=2,
      min_detection_confidence=0.5) as hands:
    hand_data = []
    for file in os.scandir(directory):
      # Read an image, flip it around y-axis for correct handedness output (see
      # above).
      image = cv2.flip(cv2.imread(file.path), 1)
      # Convert the BGR image to RGB before processing.
      results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

      image_height, image_width, _ = image.shape
      
      for hand_landmarks in results.multi_hand_landmarks:
        landmark_data = []
        for lm in hand_landmarks.landmark:
          lmx = int(lm.x * image_width)
          lmy = int(lm.y * image_height)
          landmark_data.append([lmx, lmy])
        hand_data.append(landmark_data)
  return np.array(hand_data)

