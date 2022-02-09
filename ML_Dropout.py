import numpy
from ML_base import ML_core

class Dropout(ML_core):
    def __init__(self,prob=0.5):
        super().__init__()
        self.prob = prob

    def set_output_shape(self):
        self.output_shape = self.input_shape

    def apply_activation(self, x, train=True):
        if train:
            flat = numpy.array(x).flatten()
            random_indices = numpy.random.randint(0, len(flat),int(self.prob * len(flat)))
            flat[random_indices] = 0
            self.out = flat.reshape(x.shape)
        else:
            self.out = x/self.prob
        return self.out

    def backpropagate(self, nx_layer):
        self.delta = nx_layer.delta
        self.delta[self.out == 0] = 0

if __name__ == '__main__':
    x = numpy.arange(100).reshape(10,10)
    dp = Dropout()
    dp.apply_activation(x)
    print(dp.out)
    dp2 = Dropout()
    #dp2.weights = numpy.random.normal(size=(10,10))
    dp2.delta = numpy.random.normal(size=(10,10))
    dp.backpropagate(dp2)
    print(dp.delta)