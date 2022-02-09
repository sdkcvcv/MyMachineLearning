import numpy
from ML_base import ML_core

class Pool2d(ML_core):
    def __init__(self,kernel_size=(2,2), stride=None, kind='max'):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

        if self.stride == None:
            self.stride = self.kernel_size

        self.pools = ['max', "average", 'min']
        if kind not in self.pools:
            raise ValueError(f"Pool kind not understoood. Use one of {self.pools} instead.")
        self.kind = kind

    def set_output_shape(self):
        row = int((self.input_shape[0] - self.kernel_size[0]) / self.stride[0]) + 1
        col = int((self.input_shape[1] - self.kernel_size[1]) / self.stride[1]) + 1
        self.output_shape = (row,col,self.input_shape[2])

    def apply_activation(self, image):
        ksize = self.kernel_size
        stride = self.stride
        self.out = numpy.zeros(self.output_shape)

        for f in range(self.output_shape[2]):
            for r in range(self.output_shape[0]):
                rr = r * stride[0]
                for c in range(self.output_shape[1]):
                    cc = c * stride[1]
                    buff = image[rr:rr+ksize[0],cc:cc+ksize[1],f]
                    if len(buff) > 0:
                        if self.kind == 'max':
                            buff = numpy.max(buff)
                        if self.kind == 'min':
                            buff = numpy.min(buff)
                        if self.kind == 'average':
                            buff = numpy.mean(buff)
                    self.out[r,c,f] = buff
        return self.out

    def backpropagate(self, nx_layer):
        ksize = self.kernel_size
        stride = self.stride
        image = self.input

        self.delta = numpy.zeros((image.shape))

        for f in range(self.output_shape[2]):
            for r in range(self.output_shape[0]):
                rr = r * stride[0]
                for c in range(self.output_shape[1]):
                    cc = c * stride[1]
                    buff = image[rr:rr + ksize[0], cc:cc + ksize[1], f]
                    dout = nx_layer.delta[r,c,f]
                    if self.kind == 'max':
                        p = numpy.max(buff)
                        index = numpy.argwhere(buff == p)[0]
                        self.delta[rr+index[0], cc + index[1],f] = dout
                    if self.kind == 'min':
                        p = numpy.min(buff)
                        index = numpy.argwhere(buff == p)[0]
                        self.delta[rr + index[0], cc + index[1], f] = dout
                    if self.kind == 'average':
                        #p = numpy.mean(buff)
                        self.delta[rr:rr + ksize[0], cc:cc + ksize[1], f] = dout
        return self.delta


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    x = numpy.random.randint(1, 100, (32, 32, 3))

    pool = Pool2d(kernel_size=(4, 4), kind="average")
    pool.input = x
    pool.input_shape = x.shape
    pool.set_output_shape()
    out = pool.apply_activation(x)

    pool2 = Pool2d(kernel_size=(4, 4), kind="average")
    pool2.delta = out
    pool.backpropagate(pool2)

    fig, ax = plt.subplots(nrows=2, ncols=1)
    ax[0].imshow(pool.delta/99, cmap='Accent')
    ax[1].imshow(pool.delta1/99, cmap='Accent')
    plt.show()