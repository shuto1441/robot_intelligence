# coding: utf-8
import numpy as np
from functions import *

class Affine:

    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]

    def forward(self, x):
        W, b = self.params
        self.x = x
        out = np.dot(x, W) + b
        return out

    def backward(self, dout):
        W, _ = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        db = np.sum(dout, axis=0)
        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx


class Sigmoid:

    def __init__(self):
        self.params = []
        self.grads  = []

    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * self.out * (1.0-self.out)
        return dx


class SoftmaxWithLoss:

    def __init__(self):
        self.params = []
        self.grads  = []

    def forward(self, x, t):
        self.y = softmax(x)
        self.t = acurate_label(self.y, t)

        # loss = cross_entropy_error(self.y, self.t)
        # return loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx  = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx *= dout
        dx  = dx / batch_size
        return dx


class SGD:

    def __init__(self):
        pass

    def update(self, params, grads, lr):
        for i in range(len(params)):
            params[i] -= lr * grads[i]
