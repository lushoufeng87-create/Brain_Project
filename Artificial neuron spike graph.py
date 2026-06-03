import numpy as np

def relu(x):
    #Activation function: Rectified Linear Unit
    return np.maximum(0, x)

def artificial_neuron(inputs, weights, bias):
    #Calculates the output of a single artificial neuron.
    # Dot product of inputs and weights, plus bias
    z = np.dot(inputs, weights) + bias
    return relu(z)

# usage
inputs = np.array([0.5, -0.2, 1.0])
weights = np.array([0.8, -0.5, 1.2])
bias = 0.1

output = artificial_neuron(inputs, weights, bias)
print(f"Artificial Neuron Output: {output:.4f}")

