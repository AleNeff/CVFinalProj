import numpy as np
import random

from sklearn.metrics import euclidean_distances

"""
params:
landmarks = 21 (x,y) values in range [0,1] from landmark detector
frame_shape = x, y, _ representing frame from landmark detector

output:
20-length array of normalized distances to base point of palm
returns an empty array if not all landmarks are present (may need to change later)

This function re-spreads the x y coordinates, then normalizes based on distance
from base of hand, and returns a descriptor of these normalized distances to base point.
The furthest point from the base will be = to 1, all others will be a fraction of this based on relative size.
"""
def create_descriptor(landmarks, frame_shape):
  if len(landmarks) < 21:
    return []
  x, y, _ = frame_shape
  base_x, base_y = landmarks[0]
  base_x *= x
  base_y *= y
  lm_dists_normed = []
  for i in range(20):
    lm_x, lm_y = landmarks[i+1]
    lm_x *= x
    lm_y *= y
    # √[(x2 – x1)2 + (y2 – y1)2]
    lm_dist = np.sqrt(np.square(lm_x - base_x) + np.square(lm_y - base_y))
    lm_dists_normed.append(lm_dist)

  max_dist = max(lm_dists_normed)
  lm_dists_normed = np.array(lm_dists_normed)
  lm_dists_normed /= max_dist
  return lm_dists_normed

"""
params:
descriptor: 20-d descriptor of normalized dists to base point
target_descriptor: 20-d descriptor for the target class, should be 1 of 5

output:
pseudo-euclidean distance of input and target descriptor

Use this function to get a distance metric back for the new hand to each target class
Call 5 times, and then perform ratio test on those 5 distances to see if there is a
clear winner -- if so, that's the predicted handsign. (ratio test params TBD)
"""
def dist_to_target(descriptor, target_descriptor):
  euclidean_dist = 0
  if len(descriptor) != len(target_descriptor):
    return -1
  for i in range(len(descriptor)):
    euclidean_dist += np.square(descriptor[i] - target_descriptor[i])
  euclidean_dist = np.sqrt(euclidean_dist)
  return euclidean_dist


## TESTING FUNCTIONS

def fake_dists():
  fake_dists = []
  for i in range(21):
    x = random.randint(0,100)
    y = random.randint(0,100)
    x /= 100
    y /= 100
    fake_dists.append((x,y))
  return fake_dists

def test_funcs():
  fake_non_target = fake_dists() # a new hand
  frame = 600, 400, 0
  descriptor = create_descriptor(fake_non_target, frame)

  fake_targets = []
  for i in range(5):
    fake_targets.append(create_descriptor(fake_dists(), frame))

  for fake_targ in fake_targets:  # existing hands
    print(dist_to_target(descriptor, fake_targ))