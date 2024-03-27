import numpy as np
import matplotlib.pyplot as plt

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ReLU function
def relu(x):
    return np.maximum(0, x)

# Leaky ReLU function
def leaky_relu(x, alpha=0.01):
    return np.where(x >= 0, x, alpha * x)

# Tanh function
def tanh(x):
    return np.tanh(x)

# Generate an array of values from -10 to 10
x = np.linspace(-10, 10, 1000)

# Create a dictionary of the activation functions
activations = {
    "Sigmoid": sigmoid(x),
    "ReLU": relu(x),
    "Leaky ReLU": leaky_relu(x),
    "Tanh": tanh(x)
}

# Plot each activation function
for name, y in activations.items():
    plt.figure()
    plt.plot(x, y)
    plt.title(name)

plt.show()
