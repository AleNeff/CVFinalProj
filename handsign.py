import cv2
import numpy as np
import mediapipe as mp
import classifier
import static_handsign
import cluster

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load class names
f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()
print(classNames)

# Initialize the webcam for Hand Gesture Recognition Python project
cap = cv2.VideoCapture(0)

"""
Get truth data
"""
truth_data = []
truth_data.append(static_handsign.generate_truth_data("./images/claws")[0])
truth_data.append(static_handsign.generate_truth_data("./images/frogs")[0])
truth_data.append(static_handsign.generate_truth_data("./images/gigem")[0])
truth_data.append(static_handsign.generate_truth_data("./images/gunsup")[0])
truth_data.append(static_handsign.generate_truth_data("./images/horns")[0])
truth_descriptors = [classifier.create_descriptor(truth_data[i]) for i in range(len(truth_data))]

# perform clustering
# kmodel = cluster.build_kmeans(truth_descriptors)

# relabel_data = []
# relabel_data.append(static_handsign.generate_truth_data("./images/claws")[1])
# relabel_data.append(static_handsign.generate_truth_data("./images/frogs")[1])
# relabel_data.append(static_handsign.generate_truth_data("./images/gigem")[1])
# relabel_data.append(static_handsign.generate_truth_data("./images/gunsup")[1])
# relabel_data.append(static_handsign.generate_truth_data("./images/horns")[1])

# labels = kmodel.predict([classifier.create_descriptor(relabel_data[i]) for i in range(len(relabel_data))])
# print(labels)
# label_alignment = {
#   labels[0]:"claws",
#   labels[1]:"frogs",
#   labels[2]:"gigem",
#   labels[3]:"gunsup",
#   labels[4]:"horns"
# }

while True:
  # Read each frame from the webcam
  _, frame = cap.read()
  x , y, c = frame.shape

  # Flip the frame vertically
  frame = cv2.flip(frame, 1)
  # Show the final output
 

  framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Get hand landmark prediction
  result = hands.process(framergb)

  className = ''

  # post process the result
  if result.multi_hand_landmarks:
    landmarks = []
    # signs = np.zeros(5)
    for handslms in result.multi_hand_landmarks:
      for lm in handslms.landmark:
          # print(id, lm)
          lmx = int(lm.x * x)
          lmy = int(lm.y * y)

          landmarks.append([lmx, lmy])

      # Drawing landmarks on frames
      mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

      # REPLACE WITH NEAREST NEIGHBOR CHECKS BASED ON POSITIONS
      
      """
      Steps:
      Have a set with an image for each class's hand
      Normalize their x and y positions for their landmarks, store those permanently
      Normalize x and y positions of handlandmarks
      Do Nearest Neighbors check on the new normed landmarks with each stored set of normed landmarks
      Closest match (use ratio test) is the predicted handsign.
      If failing ratio test, don't assign a handsign label
      """
      new_descriptor = classifier.create_descriptor(landmarks)
      dists = [classifier.dist_to_target(new_descriptor, truth_descriptors[x]) for x in range(5)]
      # className = classNames[np.argmin(dists)]
      sorted_dists = [dists[x] for x in range(5)]
      sorted_dists.sort()
      ratio = sorted_dists[0] / sorted_dists[1]
      if ratio < 0.9:
        className = classNames[np.argmin(dists)]
      else:
        className = ""
      # signs[np.argmin(dists)] += 1
    # className = label_alignment[int(cluster.predict_centroids(new_descriptor, kmodel))]
    # className = ("claws: " + str(signs[0]) + " frogs: " + str(signs[1]) + " gigem: " + str(signs[2]) + " gunsup: " + str(signs[3]) + " horns: " + str(signs[4]))
  # show the prediction on the frame
  cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
  cv2.imshow("Output", frame)
  if cv2.waitKey(1) == ord('q'):
    break
  # release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()