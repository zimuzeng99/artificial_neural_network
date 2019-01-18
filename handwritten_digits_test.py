from mnist import MNIST
import random
import neural_network

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

nn = neural_network.NeuralNetwork([image_width * image_height, 16, 16, 10], 0.2)

training_size = 1000

# Normalizes the training set to improve learning effectiveness
for i in range(0, training_size):
    training_images[i] = normalize(training_images[i])



for i in range(0, 10):
    nn.train(training_images[ : training_size], training_labels[ : training_size])
    print(nn.calculate_cost(training_images[ : training_size], training_images[ : training_size]))


test_size = 2000

# Normalizes the entire test set to improve learning effectiveness
for i in range(0, test_size):
    test_images[i] = normalize(test_images[i])
"""
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
print("Accuracy: " + str(accuracy))
"""