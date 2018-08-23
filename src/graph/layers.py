from __future__ import absolute_import

from src import activations
import numpy as np

class Layer:

    def __init__(self, shape):
        self.shape = shape
        self.graph = None

    def get_preceding_shape(self):
        if self.graph is None:
            raise TypeError('Layer was not added to any graph. Preceding shape is not known.')
        
        index = self.graph.get_index(self)
        return self.graph.input_shape if index == 0 else self.graph.get_layer(index-1).shape

    def is_output(self):
        if self.graph is None:
            raise TypeError('Layer was not added to any graph. Preceding shape is not known.')
        
        index = self.graph.get_index(self)
        return index+1 == len(self.graph.layers)

    def forward(self, inputs):
        pass

    def gradient(self, gradient):
        pass

class Dense(Layer):

    def __init__(self, num_nodes, activation=None):
        super().__init__((num_nodes,))
        self.activation = activations.get(activation)
        self.graph = None
        self.W = None
        self.DW = None

    def initialize_weights(self):
        shape = self.get_weights_shape()
        self.W = np.random.randn(shape) / np.sqrt(shape[1])

    def get_weights_shape(self, include_bias=True):
        preceding_shape = self.get_preceding_shape()
        return (self.shape[0], preceding_shape[0] + (1 if include_bias else 0))

    def compute(self, A):
        Z = np.matmul(A, self.W.T)
        A = self.activation(Z)
        return A

    def gradient(self, upstream_gradient):
        pass