import numpy
from ML_base import ML_core

class Conv2d(ML_core):
    def __init__(self,input_shape=None, isbias=True,
                 filters=1, kernel_size=(3,3), stride=(1,1),
                 weights=None, bias=None, activation='relu',
                 padding=None):
        super().__init__(input_shape=input_shape, isbias=isbias, activation=activation)

        # parameters for this class
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride

        #test param
        self.padding = padding
        self.p = 1 if padding != None else 0

        # other all variables
        self.bias = bias
        self.w = weights

        if input_shape != None:
            self.set_output_shape()

    def init_param(self, size):
        stddev = 1/numpy.sqrt((numpy.prod(size)))
        return numpy.random.normal(loc=0,scale=stddev, size=size)

    def set_variables(self):
        self.weights = self.w if self.w != None else self.init_param(self.kernel_size)
        if self.isbias:
            self.biases = self.bias if self.bias != None else self.init_param((self.filters, 1))
        else:
            self.biases = numpy.zeros((self.filters,1))
        self.parameters = numpy.multiply.reduce(self.kernel_size) + (self.filters if self.isbias else 1)
        self.delta_weights = numpy.zeros(self.kernel_size)
        self.delta_biases = numpy.zeros(self.biases.shape)

    def set_output_shape(self):
        self.kernel_size = (self.kernel_size[0], self.kernel_size[1],
                            self.input_shape[2], self.filters)
        self.set_variables()
        row = int((self.input_shape[0] - self.kernel_size[0] + 2 * self.p) / self.stride[0]) + 1
        col = int((self.input_shape[1] - self.kernel_size[1] + 2 * self.p) / self.stride[1]) + 1
        self.output_shape = (row, col, self.filters)

    def apply_activation(self, image):
        ksize = self.kernel_size
        stride = self.stride

        self.out = numpy.zeros(self.output_shape)

        if self.padding == 'zeros':
            image = numpy.pad(image,((1,1),(1,1),(0,0)))
        if self.padding == 'same':
            image = numpy.pad(image,((1,1),(1,1),(0,0)),'edge')

        for f in range(self.filters):
            for r in range(self.output_shape[0]):
                rr = r * stride[0]
                for c in range(self.output_shape[1]):
                    cc = c * stride[1]
                    buff = image[rr:rr+ksize[0], cc:cc+ksize[1]]
                    self.out[r,c,f] = numpy.sum(buff * self.weights[:,:,:,f]) + self.biases[f]
        self.out = self.activation_fn(self.out)
        return self.out

    def backpropagate(self, nx_layer):
        ksize = self.kernel_size
        stride = self.stride
        image = self.input

        self.delta = numpy.zeros((self.input_shape[0:3]))

        for f in range(self.filters):
            for r in range(self.output_shape[0]- (ksize[0]-1) if self.padding != None else 0):
                rr = r * stride[0]
                for c in range(self.output_shape[1]- (ksize[1]-1) if self.padding != None else 0):
                    cc = c * stride[1]
                    buff = image[rr:rr + ksize[0], cc:cc + ksize[1]]
                    self.delta_weights[:,:,:,f] += buff * nx_layer.delta[r,c,f]
                    buff = nx_layer.delta[r,c,f] * self.weights[:,:,:,f]
                    self.delta[rr:rr+ksize[0], cc:cc+ksize[1]] += buff
            self.delta_biases[f] = numpy.sum(nx_layer.delta[:,:,f])
        self.delta = self.activation_dfn(self.delta)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    img = numpy.random.normal(0,3, size=(28,28,1))
    conv = Conv2d(input_shape=(28,28,1),padding='zeros')
    conv.input = img
    conv.weights = numpy.array([[1, 0, -1],
                                [1, 0, -1],
                                [1, 0, -1]]).reshape(3, 3, 1, 1)
    conv.out = numpy.zeros((28, 28, 1))
    cout = conv.apply_activation(img)

    conv2 = Conv2d()
    conv2.delta = conv.init_param((26, 26, 1))
    cout2 = conv.backpropagate(conv2)


    plt.imshow(conv2.delta)
    plt.show()