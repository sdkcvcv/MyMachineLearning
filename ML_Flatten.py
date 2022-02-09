import numpy
from ML_base import ML_core

class Flatten(ML_core):
    def __init__(self):
        super().__init__()

    def set_output_shape(self):
        self.output_shape = numpy.prod(self.input_shape)

    def apply_activation(self,x):
        self.out = numpy.array(x).flatten()
        return self.out

    def backpropagate(self,nx_layer):
        self.delta = nx_layer.delta.reshape(self.input_shape)

if __name__ == '__main__':
    x = numpy.array([[1, 1, 1], [1, 0, 1], [0, 1, 1], [0, 0, 1]])
    f = Flatten()
    f.input_shape = x.shape
    f.apply_activation(x)
    f2 = Flatten()
    f2.delta = numpy.ones((12))
    f.backpropagate(f2)

    print(f.delta)