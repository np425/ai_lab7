import numpy as np

NUM_FEATURES = 4
NUM_CLASSES = 3
ITERATIONS = 100
LEARNING_RATE = 0.01

def initialize_parameters(num_features, num_classes):
    weights = {i: np.zeros(num_features) for i in range(1, num_classes + 1)}
    biases = {i: 0 for i in range(1, num_classes + 1)}
    return weights, biases

def load_and_prepare_data(filepath, delimiter="\t"):
    data = np.loadtxt(filepath, delimiter=delimiter).reshape(-1, 5)
    np.random.seed(42)  # Seed for reproducibility
    np.random.shuffle(data)  # Shuffle data
    split_index = int(0.8 * len(data))  # 80% training, 20% testing
    train_data = data[:split_index]
    test_data = data[split_index:]
    return train_data, test_data

def train_model(train_data, weights, biases, iterations, learning_rate):
    train_features = train_data[:, :NUM_FEATURES]
    train_labels = train_data[:, NUM_FEATURES].astype(int)
    for _ in range(iterations):
        for feature, label in zip(train_features, train_labels):
            scores = {cls: np.dot(feature, weights[cls]) + biases[cls] for cls in range(1, NUM_CLASSES + 1)}
            predicted_class = max(scores, key=scores.get)
            if predicted_class != label:
                weights[label] += learning_rate * feature
                biases[label] += learning_rate
                weights[predicted_class] -= learning_rate * feature
                biases[predicted_class] -= learning_rate

def predict(features, weights, biases):
    predictions = []
    for feature in features:
        scores = [np.dot(feature, weights[cls]) + biases[cls] for cls in range(1, NUM_CLASSES + 1)]
        predicted_class = np.argmax(scores) + 1
        predictions.append(predicted_class)
    return predictions

def calculate_accuracy(predictions, labels):
    return np.mean(predictions == labels) * 100

# Main execution
weights, biases = initialize_parameters(NUM_FEATURES, NUM_CLASSES)
train_data, test_data = load_and_prepare_data('iris.data')
train_model(train_data, weights, biases, ITERATIONS, LEARNING_RATE)
test_features = test_data[:, :NUM_FEATURES]
test_labels = test_data[:, NUM_FEATURES].astype(int)
test_predictions = predict(test_features, weights, biases)
accuracy = calculate_accuracy(test_predictions, test_labels)
print(f"Test Accuracy: {accuracy:.2f}%")
