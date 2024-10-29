import pandas as pd
import numpy as np

np.random.seed(0)

def get_data():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[1], [0], [1], [0]])

    return X, y

class Model():
    def __init__(self, input_size=2, output_size=1) -> None:
        self.hidden_size = 3
        self.input_size = input_size
        self.output_size = output_size
        self.W1 = np.random.randn(self.hidden_size, self.input_size)
        self.b1 = np.zeros((self.hidden_size, 1))
        self.W2 = np.random.randn(self.hidden_size, self.hidden_size)
        self.b2 = np.zeros((self.hidden_size, 1))
        self.W3 = np.random.randn(self.output_size, self.hidden_size)
        self.b3 = np.zeros((self.output_size, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def foward(self, x):
        self.Y1 = np.dot(self.W1, x.T) + self.b1
        self.A1 = self.sigmoid(self.Y1)

        self.Y2 = np.dot(self.W2, self.A1) + self.b2
        self.A2 = self.sigmoid(self.Y2)

        self.Y3 = np.dot(self.W3, self.A2) + self.b3
        self.A3 = self.sigmoid(self.Y3)

        return self.A3.T
    
    def binary_cross_entropy_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        loss = -(1/m) * np.sum(y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred))
        
        return loss
    
    def backward(self, X: np.array, y_true: np.array):
        
        """layer 3"""
        # find derivative on node value
        dA3 = (1 / y_true.shape[0]) * (- (y_true.T / self.A3) + ((1-y_true.T)/(1-self.A3)))

        # find derivative on activatoin functino
        dY3 = dA3 * self.A3 * (1 - self.A3)
        print(f"dy3 shape {dY3.shape}")


        # find derivative on param
        self.dW3 = np.dot(dY3, self.A2.T)
        print(f"A2 T shape {self.A2.T.shape}")
        print("dw3 shape", self.dW3.shape)
        self.db3 = np.sum(dY3, axis=1, keepdims=True)

        """layer 2"""
        # find derivative on node value of node value
        dA2 = np.dot(self.W3.T, dY3)
        
        # find derivative on activation function
        dY2 = dA2 * self.A2 * (1 - self.A2)
        print(f"\ndy2 shape {dY2.shape}")
        # find the gradient of the weight
        self.dW2 = np.dot(dY2, self.A1.T)
        print(f"dw2 shape {self.dW2.shape}")
        self.db2 = np.sum(dY2, axis=1, keepdims=True)

        """layer 1"""
        dA1 = np.dot(self.W2.T, dY2)
        dY1 = dA1 * self.A1 * (1 - self.A1)
        print(f"\ndy1 shape {dY1.shape}")
        self.dW1 = np.dot(dY1, X)
        print(f"dw1 shape {self.dW1.shape}")
        self.db1 = np.sum(dY1, axis=1, keepdims=True)

    def step(self, learning_rate=0.5):
        # update the weights and biases
        self.W1 = self.W1 - learning_rate*self.dW1
        self.b1 = self.b1 - learning_rate*self.db1
        self.W2 = self.W2 - learning_rate*self.dW2
        self.b2 = self.b2 - learning_rate*self.db2
        self.W3 = self.W3 - learning_rate*self.dW3
        self.b3 = self.b3 - learning_rate*self.db3
        
        # print(f"step | W1 {self.W1}, b1 {self.b1}, W2 {self.W2}, b2 {self.b2}, W3 {self.W3}, b3 {self.b3}")

if __name__ == '__main__':
    model = Model()
    X, y = get_data()

    for epoch in range(10):
        y_pred = model.foward(X)
        loss = model.binary_cross_entropy_loss(y_pred, y)
        
        model.backward(X, y)
        model.step()

        print(f"Epoch {epoch}, loss {loss}")


