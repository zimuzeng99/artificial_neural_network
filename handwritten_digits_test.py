from mnist import MNIST
import random
import neural_network
import numpy as np

# Converts a digit label to a probability vector. E.g. 5 would convert to
# [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
def label_to_vector(digit_label):
    prob_vector = [0] * 10
    prob_vector[digit_label] = 1
    return prob_vector

# Brings each pixel value closer to 0 to increase effectiveness of learning
def normalize(image):
    normalized_image = []
    for i in range(0, len(image)):
        normalized_image.append(image[i] / 255)
    return normalized_image

# Loads the training and test data from the MNIST files in the root directory
mndata = MNIST('')
training_images, training_labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()

# Converts all training labels from single digit into probability vector form
temp = []
for label in training_labels:
    temp.append(label_to_vector(label))
training_labels = temp

# Converts all test labels from single digit into probability vector form
temp = []
for label in test_labels:
    temp.append(label_to_vector(label))
test_labels = temp

image_width = 28
image_height = 28

"""
index = random.randrange(0, len(training_images))
print(mndata.display(training_images[index]))
print(training_labels[index])
"""

nn = neural_network.NeuralNetwork([image_width * image_height, 200, 80, 10], 0.2)

training_size = 5000

# Normalizes the training set to improve learning effectiveness
for i in range(0, training_size):
    training_images[i] = normalize(training_images[i])

test_size = 10000

# Normalizes the entire test set to improve learning effectiveness
for i in range(0, test_size):
    test_images[i] = normalize(test_images[i])

num_iterations = 0
cost = nn.calculate_cost(training_images[ : training_size], training_images[ : training_size])
while num_iterations < 10:
    for i in range(0, training_size):
        nn.train(training_images[i], training_labels[i])
    num_iterations += 1

num_correct = 0

for i in range(0, training_size):
    # The output will be a vector of probabilities. We use the largest value
    # in this vector to determine the digit that the NN recognized.
    output = nn.compute(training_images[i])
    digit = output.index(max(output))

    # If the NN predicted the correct digit
    if digit == training_labels[i].index(1):
        num_correct += 1

accuracy = num_correct / training_size

print("Training acurracy: " + str(accuracy))

num_correct = 0

for i in range(0, test_size):
    # The output will be a vector of probabilities. We use the largest value
    # in this vector to determine the digit that the NN recognized.
    output = nn.compute(test_images[i])
    digit = output.index(max(output))

    # If the NN predicted the correct digit
    if digit == test_labels[i].index(1):
        num_correct += 1

accuracy = num_correct / test_size
print("Test accuracy: " + str(accuracy))

np.save("weights0", nn.weights[0])
np.save("weights1", nn.weights[1])
np.save("weights2", nn.weights[2])

np.save("biases0", nn.biases[0])
np.save("biases1", nn.biases[1])
np.save("biases2", nn.biases[2])
