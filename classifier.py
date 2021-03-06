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
def create_descriptor(landmarks):
  if len(landmarks) < 21:
    return []
  base_x, base_y = landmarks[0] # the base here will be considered as the WRIST point, baseXY are acutal pixel coordinates
  lm_dists_normed = []
  for i in range(20):  # compute distance between WRIST base and every other point, square, and add up, then sqrt
    lm_x, lm_y = landmarks[i+1]
    # √[(x2 – x1)2 + (y2 – y1)2]
    lm_dist = np.sqrt(np.square(lm_x - base_x) + np.square(lm_y - base_y))
    lm_dists_normed.append(lm_dist)

  max_dist = max(lm_dists_normed)  # normalization done by dividing each distance by the largest distance measured
  lm_dists_normed = np.array(lm_dists_normed)
  lm_dists_normed /= max_dist
  return lm_dists_normed


"""
Plural version of create_descriptor
"""
def create_descriptors(hands):
  descriptors = []
  for hand_lm in hands:
    descriptor = create_descriptor(hand_lm)
    descriptors.append(descriptor)
  return np.array(descriptors)


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
  PENALTY_FACTOR = 5
  euclidean_dist = 0
  if len(descriptor) != len(target_descriptor):
    return -1  # error code
  for i in range(len(descriptor)):
    euclidean_dist += np.square(descriptor[i] - target_descriptor[i])
    if i in [3, 7, 11, 19]:
      euclidean_dist * PENALTY_FACTOR
  euclidean_dist = np.sqrt(euclidean_dist)
  return euclidean_dist


## TESTING FUNCTIONS
# gets all of the points of a hand, but fake
def get_fake_hand_points():
  fake_hand_points = []
  for i in range(21):
    x = random.randint(0,100)
    y = random.randint(0,100)
    x /= 100
    y /= 100
    fake_hand_points.append((x,y))
  return fake_hand_points

def test_funcs():
  fake_non_target = get_fake_hand_points() # a new hand
  descriptor = create_descriptor(fake_non_target)

  fake_targets = []
  for i in range(5):
    fake_targets.append(create_descriptor(get_fake_hand_points()))

  for fake_targ in fake_targets:  # existing hands
    print(dist_to_target(descriptor, fake_targ))