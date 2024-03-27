import numpy as np
import matplotlib.pyplot as plt
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.1):
    return np.maximum(alpha * x, x)

def tanh(x):
    return np.tanh(x)   

x = np.linspace(-5, 5, 1000) 

y_sigmoid = sigmoid(x)
y_relu = relu(x)
y_leaky_relu = leaky_relu(x)
y_tanh = tanh(x)
plt.figure(figsize=(10, 6))

plt.plot(x, y_sigmoid, label='Sigmoid')
plt.plot(x, y_relu, label='ReLU')
plt.plot(x, y_leaky_relu, label='Leaky ReLU')
plt.plot(x, y_tanh, label='Tanh')

plt.xlabel('x')
plt.ylabel('Activation Function Output')
plt.title('Activation Functions')
plt.legend()

plt.grid(True)
plt.show()
