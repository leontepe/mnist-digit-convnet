
class LayerGraph:
    """ 
    Represents a linear computational graph of layers. Different kinds of neural network variations can be easily put together using this module.
    
    Example:
    --------
    ```
    # Define model
    model = LayerGraph(input_shape=(784,))
    model.add(Dense(30, activation='sigmoid'))
    model.add(Dense(10, activation='softmax'))

    # Train model
    model.compile(optimizer='sgd', loss='categorical_cross_entropy')

    # Train model
    model.fit(X_train, y_train, epochs=10)
    model.evaluate(X_test, y_test)
    ```

    Abc
    """

    def __init__(self, input_shape=None, optimizer=''):
        self.input_shape = input_shape
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)
        layer.graph = self

    def get_layer(self, index):
        if index < 0 or index >= len(self.layers):
            raise IndexError('Index is out of bounds:', index)
        else:
            return self.layers

    def get_index(self, layer):
        if layer in self.layers:
            return self.layers.index(layer)
        else:
            raise ValueError('Layer not found in layer graph.')

    def initialize_weights(self):
        for layer in self.layers:
            layer.initialize_weights()

    def forward(self):

    def backward(self):
