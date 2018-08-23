import numpy as np

class Node:
    """ Represents a node in a computational graph. """

    def __init__(self, input_nodes):
        self.input_nodes = input_nodes
        self.output_nodes = list()

        # Track this node as an output node of all the input nodes
        for node in input_nodes:
            node.output_nodes.append(self)

    def forward(self):
        """ Makes a forward pass and computes the output of the node. """
        pass

    def backward(self):
        """ Makes a backward pass and computes the gradient of the current node. """
        pass

class Add(Node):
    """ Represents the element-wise addition operation as a node. """

    def __init__(self, a, b):
        super().__init__([a, b])

    def forward(self, a, b):
        return a + b

class MatMul(Node):
    """ Represents the matrix multiplication operation as a node. """

    def __init__(self, a, b):
        super().__init__([a, b])

    def forward(self, a, b):
        return np.matmul(a, b)

class Placeholder:
    """ Represents a placeholder node that has to be provided with a value in order for output of the graph to be computed. """

    def __init__(self):
        self.output_nodes = list()

class Variable:
    """ Represents a variable in the graph. """