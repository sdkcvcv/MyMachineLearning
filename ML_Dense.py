import numpy
from ML_base import ML_core

class Dense(ML_core):

    def __init__(self,input_shape=None, isbias=True,neurons=1,
                 activation = 'softmax',weights=None, bias=None):
        super().__init__(input_shape=input_shape,isbias=isbias,activation=activation)
        self.neurons = neurons

        # other all variables
        self.bias = bias
        self.w = weights

        if input_shape != None:
            self.set_output_shape()

    def set_variables(self):
        self.weights = self.w if self.w != None else (
            numpy.random.normal(size=(self.input_shape, self.neurons)))
        if self.isbias:
            self.biases = self.bias if self.bias != None else numpy.random.normal(size=(self.neurons))
        else:
            self.biases = 0
        self.parameters =  self.input_shape * self.neurons + (self.neurons if self.isbias else 0)
        self.delta_weights = numpy.zeros(self.weights.shape)
        self.delta_biases = numpy.zeros(self.biases.shape)

    def set_output_shape(self):
        self.set_variables()
        self.output_shape = self.neurons

    def apply_activation(self,x):
        buff = numpy.dot(x, self.weights) + self.biases
        self.out = self.activation_fn(buff)
        return self.out

    def backpropagate(self,nx_layer):
        if hasattr(nx_layer, 'delta'):
            self.error = nx_layer.delta
        else:
            self.error = nx_layer
        self.delta = numpy.dot(self.weights, self.error)
        self.delta_weights += self.error * self.activation_dfn(self.out) * numpy.atleast_2d(self.input).T
        self.delta_biases += self.error * self.activation_dfn(self.out)


if __name__ == "__main__":
    x = numpy.random.normal(size=(100))
    #print(x.shape)

    layer1 = Dense(input_shape=100, neurons=20)
    layer1.input = x
    layer1.set_output_shape()
    layer1.apply_activation(x)

    layer2 = Dense(neurons=10)
    layer2.input = layer1.out
    layer2.input_shape = layer1.output_shape
    layer2.set_output_shape()
    layer2.apply_activation(layer1.out)

    layer3 = Dense()
    error = numpy.zeros((10)) + 0.01
    error[1] = 0.98
    font = [0,0.99,0,0,0,0,0,0,0,0]
    layer3.delta = error

    layer2.backpropagate(error)
    layer1.backpropagate(layer2)





