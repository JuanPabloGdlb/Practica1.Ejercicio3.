import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Función de activación (en este caso, sigmoide)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivada de la función de activación
def sigmoid_derivative(x):
    return x * (1 - x)

# Clase para la implementación del perceptrón multicapa
class MultiLayerPerceptron:
    def __init__(self, layers):
        self.layers = layers
        self.weights = []
        for i in range(1, len(layers)):
            self.weights.append(np.random.uniform(-1, 1, (layers[i - 1] + 1, layers[i])))

    def forward_propagation(self, X):
        activations = [X]
        for i in range(len(self.weights)):
            activations[-1] = np.append(activations[-1], np.ones((len(activations[-1]), 1)), axis=1)
            activations.append(sigmoid(np.dot(activations[-1], self.weights[i])))
        return activations

    def backward_propagation(self, X, y, activations, learning_rate):
        deltas = [activations[-1] - y]
        for i in range(len(self.weights) - 1, 0, -1):
            deltas.append(np.dot(deltas[-1], self.weights[i][:-1].T) * sigmoid_derivative(activations[i]))
        deltas.reverse()
        
        for i in range(len(self.weights)):
            activation_with_bias = np.append(activations[i], np.ones((len(activations[i]), 1)), axis=1)
            weight_update = np.dot(activation_with_bias.T, deltas[i])
            self.weights[i] -= learning_rate * weight_update

    def fit(self, X, y, epochs=1000, learning_rate=0.1):
        for _ in range(epochs):
            activations = self.forward_propagation(X)
            self.backward_propagation(X, y, activations, learning_rate)

    def predict(self, X):
        activations = self.forward_propagation(X)
        return np.round(activations[-1])

# Cargar el conjunto de datos
data = pd.read_csv("concentlite.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values.reshape(-1, 1)

# Visualizar el conjunto de datos
plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='coolwarm')
plt.title("Conjunto de Datos Concentlite")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Entrenar el perceptrón multicapa
mlp = MultiLayerPerceptron(layers=[2, 5, 1])
mlp.fit(X, y, epochs=10000, learning_rate=0.1)

# Visualizar la clasificación realizada por el perceptrón multicapa
h = 0.01
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = mlp.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)


plt.contourf(xx, yy, Z, cmap='coolwarm', alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='coolwarm', edgecolors='k')
plt.title("Clasificación por Perceptrón Multicapa")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

