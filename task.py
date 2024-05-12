import numpy as np

iris_data = np.loadtxt('iris.data', delimiter="\t").reshape(-1, 5)

# Split features and labels
features = iris_data[:, :4]  # first four columns are features
labels = iris_data[:, 4]     # last column is the class

# Function to train perceptron
def train_perceptron(features, labels, epochs, learning_rate):
    n_samples, n_features = features.shape
    weights = np.zeros(n_features)
    bias = 0
    
    for _ in range(epochs):
        for idx in range(n_samples):
            x_i = features[idx]
            y_i = labels[idx]
            prediction = np.dot(x_i, weights) + bias
            update = learning_rate * (y_i - prediction)
            weights += update * x_i
            bias += update
    return weights, bias

# Encode labels for one-vs-rest binary classification
def encode_labels(labels, class_label):
    return np.where(labels == class_label, 1, -1)

# Train one perceptron for each class
epochs = 100
learning_rate = 0.01
unique_classes = np.unique(labels)
classifiers = []

for class_label in unique_classes:
    binary_labels = encode_labels(labels, class_label)
    w, b = train_perceptron(features, binary_labels, epochs, learning_rate)
    classifiers.append((w, b))

# Function to make predictions using the trained classifiers
def predict(features, classifiers):
    outputs = [np.dot(features, w) + b for w, b in classifiers]
    return np.argmax(outputs) + 1  # return the class index with the highest output

# Example: predict the class of the first sample
predicted_class = predict(features[0], classifiers)
print("Predicted class:", predicted_class)
print("Actual class:", labels[0])
