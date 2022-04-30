import static_handsign
import classifier
import numpy as np

truth_data = []
truth_data.extend(static_handsign.generate_truth_data("./images/claws"))
truth_data.extend(static_handsign.generate_truth_data("./images/frogs"))
truth_data.extend(static_handsign.generate_truth_data("./images/gigem"))
truth_data.extend(static_handsign.generate_truth_data("./images/gunsup"))
truth_data.extend(static_handsign.generate_truth_data("./images/horns"))
# n x 21 x 2

# print(np.array(truth_data).shape)
truth_descriptors = [classifier.create_descriptor(truth_data[i]) for i in range(5)]
fake_new_hand = classifier.create_descriptor(truth_data[0])
# print(truth_descriptors)
dists = [classifier.dist_to_target(fake_new_hand, truth_descriptors[x]) for x in range(5)]
print(dists)
