import numpy

class ML_core:
    def __init__(self,input_shape=None, isbias=False, activation=None):
        # Init parameters for all layers
        self.input_shape = input_shape
        self.isbias = isbias

        # Inert parameters for all layers
        self.delta = None
        self.delta_biases = 0
        self.delta_weights = 0
        self.input = None
        self.out = None
        self.output_shape = None
        self.parameters = 0

        #activations
        self.activations = [None,'linear','relu','sigmoid','softmax','tanh']
        if activation not in self.activations:
            raise ValueError(f'Activation function not recognized. Use one of {self.activations} instead.')
        else:
            self.activation = activation

    def activation_fn(self, r):
        if self.activation == None or self.activation == "linear":
            return r
        if self.activation == 'tanh':
            return numpy.tanh(r)
        if self.activation == 'sigmoid':
            return 1 / (1 + numpy.exp(-r))
        if self.activation == 'softmax':
            r = r - numpy.max(r)
            s = numpy.exp(r)
            return s / numpy.sum(s)
        if self.activation == 'relu':
            r[r < 0] = 0
            return r

    def activation_dfn(self, r):
        if self.activation is None or self.activation == "linear":
            return r
        if self.activation == 'tanh':
            return 1 - r ** 2
        if self.activation == 'sigmoid':
            r = self.activation_fn(r)
            return r * (1 - r)
        if self.activation == "softmax":
            soft = self.activation_fn(r)
            diag_soft = soft * (1 - soft)
            return diag_soft
        if self.activation == 'relu':
            r[r < 0] = 0
            return r

    def apply_activation(self):
        raise ValueError('Reoder method in your class')

    def backpropagate(self):
        raise ValueError('Reoder method in your class')

    def set_output_shape(self):
        raise ValueError('Reoder method in your class')

