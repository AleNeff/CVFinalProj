import numpy as np
import random

def fake_dists():
  fake_dists = []
  for i in range(21):
    x = random.randint(0,20)
    y = random.randint(0,20)
    fake_dists.append((x,y))
  return fake_dists

def classify(landmarks, target):
  if len(landmarks) < 21:
    return []
  base_x, base_y = landmarks[0]
  for i in range(20):
    lm_x, lm_y = landmarks[i+1]
    # √[(x2 – x1)2 + (y2 – y1)2]
    lm_dist = np.sqrt(np.square(lm_x - base_x) + np.square(lm_y - base_y))
    print(lm_dist)

fake_dists_1 = fake_dists()
fake_dists_2 = fake_dists()
classify(fake_dists_1, fake_dists_2)