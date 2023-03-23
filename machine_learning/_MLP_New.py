import numpy as np
from sklearn.metrics import accuracy_score

def threshold(X):
    return np.where(np.array(X) > 0.5, 1.0, 0.0)

class Sigmoid:
    def forward(self, input_data):
        self.input = input_data
        self.output = 1 / (1 + np.exp(-input_data))
        return self.output
    def backward(self, output_error, learning_rate):
        return self.output * (1.0 - self.output) * output_error

class Softmax:
    def forward(self, input_data):
        self.input = input_data
        self.output = np.exp(input_data) / np.sum(np.exp(input_data), axis=1, keepdims=True)
        return self.output
    def backward(self, output_error, learning_rate):
        return self.output * (1.0 - self.output) * output_error

class Layer:
    def __init__(self, n_in, n_out):
        self.input = []
        self.W = np.random.rand(n_in,n_out)-0.5
        self.b = np.random.rand(1,n_out)-0.5
        self.outputs = []
        self.g = 0
            
    def forward(self, X):
        self.input = X
        self.outputs = np.dot(self.input, self.W) + self.b
        return self.outputs

    def backward(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.W.T)
        weights_error = np.dot(self.input.T, output_error)

        # update parameters
        self.W -= learning_rate * weights_error
        self.b -= learning_rate * output_error
        return input_error

class NeuralNetwork:
    def __init__(self, n_dims = [10], epochs=100, learning_rate = 0.001):
        self.n_dims = n_dims
        self.outputs = []
        self.layers = []
        self.epochs = epochs
        self.learning_rate = learning_rate

    def cross_entropy(self, y_pred, y_true):
        return np.sum(y_true * np.log(y_pred))

    def cross_entropy_derivative(self, y_true, y_pred):
        return (y_pred-y_true)/(y_pred*(1-y_pred)*y_true.size)

    def fit(self, X_train, y_train):
        for idx, elem in enumerate(self.n_dims[:-1]):
            self.layers.append(Layer(self.n_dims[idx], self.n_dims[idx+1]))
            if elem != self.n_dims[-2]:
                self.layers.append(Sigmoid())
            else:
                self.layers.append(Softmax())

    # sample dimension first
        samples = len(X_train)

    # training loop
        for i in range(self.epochs):
            err = 0
            for j in range(samples):
                # forward propagation
                output = np.array(X_train[j]).reshape(1,-1)
                for layer in self.layers:
                    output = layer.forward(output)

                # compute loss (for display purpose only)
                output = np.squeeze(output)
                
                err += self.cross_entropy(output, y_train[j])

                # backward propagation
                error = self.cross_entropy_derivative(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward(error, self.learning_rate)

            # calculate average error on all samples
            err /= samples
            print('epoch %d/%d   error=%f' % (i+1, self.epochs, err))


    def predict(self, X_test):
        # sample dimension first
        samples = len(X_test)
        result = []
        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = np.array(X_test[i]).reshape(1,-1)
            for layer in self.layers:
                output = layer.forward(output)
            result.append(output)
        result = np.squeeze(threshold(result))
        return result

    
    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return accuracy_score(y_pred, y_test)
