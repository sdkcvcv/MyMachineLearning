import json
import time

import numpy
import pandas as pd

from ML_Conv2d import Conv2d
from ML_Dropout import Dropout
from ML_Dense import Dense
from ML_Flatten import Flatten
from ML_Optimizer import Optimizer
from ML_Pool2d import Pool2d

class CNN:
    def __init__(self):
        self.layers = []
        #self.info_df = {} # Зачем??
        #self.column = ['Lname','Input Shape', 'Output Shape', 'Activation', 'Bias'] # Зачем??
        self.parameters = []
        #self.optimizer = '' # задается в другом месте
        #self.loss = 'mse' # задается в другом месте
        #self.lr = 0.01 # задается в другом месте
        #self.mr = 0.001 # задается в другом месте
        #self.metrics = [] # задается в другом месте
        self.av_optimizers = ["sgd", "iterative", "momentum", "rmsprop", "adagrad", "adam", "adamax", "adadelta"]
        self.av_metrics = ['mse','accuracy','cse']
        self.av_loss = ['mse','cse']
        self.iscompiled = False
        #self.model_dict = None # Зачем?
        #self.out = [] # Зачем??
        self.eps = 1e-15
        self.train_loss = {}
        self.val_loss = {}
        self.train_acc = {}
        self.val_acc = {}

    def add(self,layer):
        if len(self.layers) > 0:
            prev_layer = self.layers[-1]
            prev_layer.name = f'{type(prev_layer).__name__} {len(self.layers)-1}'
            if layer.input_shape == None:
                layer.input_shape = prev_layer.output_shape
                layer.set_output_shape()
            layer.name = f'Out Layer({type(layer).__name__})'
        else:
            layer.name = 'Input Layer'
        if type(layer).__name__ == 'Conv2d':
            if(layer.output_shape[0] <= 0 or layer.output_shape[1] <= 0):
                raise ValueError(f'The output shape became invalid [i.e. {layer.output_shape}]. Reduce filter size or increase image size')
        self.layers.append(layer)
        self.parameters.append(layer.parameters)

    def summary(self):
        lname = []
        linput = []
        loutput = []
        lactivation = []
        lisbias = []
        lparam = []
        for layer in self.layers:
            lname.append(layer.name)
            linput.append(layer.input_shape)
            loutput.append(layer.output_shape)
            lactivation.append(layer.activation)
            lisbias.append(layer.isbias)
            lparam.append(layer.parameters)
        model_dict = {'Layer Name': lname, 'Input':linput, 'Output Shape':loutput,
                      'Activation': lactivation,'bias': lisbias, 'Parameters':lparam}
        model_df = pd.DataFrame(model_dict).set_index("Layer Name")
        print(model_df)
        print(f"Total Parameters: {sum(lparam)}")

    def train(self, X, Y, epochs, show_every=1, batch_size = 32,
              shuffle = True, val_split=0.1, val_x=None,val_y=None):
        self.check_trainable(X,Y)
        self.batch_size = batch_size
        t1 = time.time()
        curr_ind = numpy.arange(0, len(X), dtype=numpy.int32)
        if shuffle:
            numpy.random.shuffle(curr_ind)
        if type(val_x) != type(None) and type(val_y) != type(None):
            self.check_trainable(val_x, val_y)
            print('\nValidation data found.\n')
        else:
            val_ex = int(len(X) * val_split)
            val_exs = []
            while len(val_exs) != val_ex:
                rand_ind = numpy.random.randint(0, len(X))
                if rand_ind not in val_exs:
                    val_exs.append(rand_ind)
            val_ex = numpy.array(val_exs)
            val_x, val_y = X[val_ex], Y[val_ex]
            curr_ind = numpy.array([v for v in curr_ind if v not in val_ex])
        print(f'\nTotal {len(X)} samples.\nTraining samples: {len(curr_ind)} Validation samples: {len(val_x)}.')
        out_activation = self.layers[-1].activation
        #batches = []
        len_batch = int (len(curr_ind)/batch_size)
        if len(curr_ind)%batch_size != 0:
            len_batch +=1
        batches = numpy.array_split(curr_ind, len_batch)
        print(f'Total {len_batch} batches, most batch has {batch_size} samples.\n')
        for e in range(epochs):
            err = []
            for batch in batches:
                #a = []
                curr_x, curr_y = X[batch], Y[batch]
                b = 0
                batch_loss = 0
                for x,y in zip(curr_x,curr_y):
                    out = self.feedforward(x)
                    loss, error = self.apply_loss(y,out)
                    batch_loss += loss
                    err.append(error)
                    update = False
                    if b == batch_size-1:
                        update = True
                        loss = batch_loss/batch_size
                    self.backpropagate(loss,update)
                    b += 1
            if e % show_every == 0:

                train_out = self.predict(X[curr_ind])
                train_loss, train_error = self.apply_loss(Y[curr_ind],train_out)
                val_out = self.predict(val_x)
                val_loss,val_error = self.apply_loss(val_y,val_out)
                if out_activation == "softmax":
                    train_acc = train_out.argmax(axis=1) == Y[curr_ind].argmax(axis=1)
                    val_acc = val_out.argmax(axis=1) == val_y.argmax(axis=1)
                elif out_activation == "sigmoid":
                    train_acc = train_out > 0.7
                    val_acc = val_out > 0.7
                elif out_activation == None:
                    train_acc = abs(Y[curr_ind] - train_out) < 0.000001
                    val_acc = abs(Y[val_ex] - val_out) < 0.000001
                self.train_loss[e] = round(train_error.mean(),4)
                self.train_acc[e] = round(train_acc.mean(),4)
                self.val_loss[e] = round(val_error.mean(),4)
                self.val_acc[e] = round(val_acc.mean(),4)
                print(f'Epochs: {e}:')
                print(f'Time: {round(time.time()-t1,3)}sec')
                print(f'Train Loss: {round(train_error.mean(),4)} Train Accuracy: {round(train_acc.mean() * 100,4)}%')
                print(f'Val Loss: {round(val_error.mean(),4)} Val Accuracy: {round(val_acc.mean() * 100,4)}%\n')
                t1 = time.time()

    def check_trainable(self, X,Y):
        if self.iscompiled == False:
            raise ValueError('Model is not compiled.')
        if len(X) != len(Y):
            raise ValueError('Length of training input and label is not equal.')
        if X[0].shape != self.layers[0].input_shape:
            layer = self.layers[0]
            raise ValueError(f'"{layer.name}" expects input of {layer.input_shape} while {X[0].shape} is given.')
        if Y.shape[-1] != self.layers[-1].neurons:
            op_layer = self.layers[-1]
            raise ValueError(f'"{op_layer.name}" expects input of {op_layer.neurons} while {Y.shape[-1]} is given.')

    def compile_model(self, lr=0.01, mr = 0.001, opt = 'sgd', loss = 'mse', metrics=['mse']):
        if opt not in self.av_optimizers:
            raise ValueError(f'Optimizer is not understood, use one of {self.av_optimizers}.')
        for m in metrics:
            if m not in self.av_metrics:
                raise ValueError(f'Metrics is not understood, use one of {self.av_metrics}.')
        if loss not in self.av_loss:
            raise ValueError(f'Loss function is not understood, use one of {self.av_loss}.')
        self.loss = loss
        self.lr = lr
        self.mr = mr
        self.metrics = metrics
        self.iscompiled = True
        # Инициализация нужных настроек для "Оптимизатора"
        self.optimizer = Optimizer(layers=self.layers, name = opt, learning_rate=lr, mr=mr)
        self.optimizer = self.optimizer.opt_dict[opt]

    def feedforward(self,x,train=True):
        if train:
            for l in self.layers:
                l.input = x
                x = numpy.nan_to_num(l.apply_activation(l.input))
                l.out = x
            return x
        else:
            for l in self.layers:
                l.input = x
                if type(l).__name__ == "Dropout":
                    x = numpy.nan_to_num(l.apply_activation(x, train=train))
                else:
                    x = numpy.nan_to_num(l.apply_activation(x))
                l.out = x
            return x

    def apply_loss(self,y,out):
        if self.loss == 'mse':
            loss = y - out
            mse = numpy.mean(numpy.square(loss))
            return loss, mse
        if self.loss == 'cse':
            if len(out) == len(y) == 1:#print('Using Binary CSE.')
                cse = -(y * numpy.log(out) + (1-y) * numpy.log(1-out))
                loss = -(y / out - (1-y)/(1-out))
            else: #print("using Categorical CSE.")
                if self.layers[-1].activation == 'softmax':
                    loss = y - out
                    loss = loss / self.layers[-1].activation_dfn(out)
                else:
                    y = numpy.float64(y)
                    out += self.eps
                    loss = -(numpy.nan_to_num(y/out) - numpy.nan_to_num((1-y)/(1-out)))
                cse = - numpy.sum((y * numpy.nan_to_num(numpy.log(out)) + (1-y) * numpy.nan_to_num(numpy.log(1 - out))))
            return loss, cse

    def backpropagate(self,loss, update):
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            if layer == self.layers[-1]:
                if type(layer).__name__ == 'Dense':
                    layer.backpropagate(loss)
            else:
                nx_layer = self.layers[i+1]
                layer.backpropagate(nx_layer)
            if update: # Нужно, чтобы учесть пропорционное влияние количества "картинок" в пакете на всю "СЕТЬ"
                layer.delta_weights /= self.batch_size
                layer.delta_biases /= self.batch_size
        if update:
            self.optimizer(self.layers) # собственно обновляет веса, "обучает"
            self.zerograd() # Обнуляет предыдущие дельты

    def zerograd(self):
        for l in self.layers:
            try:
                l.delta_weights = numpy.zeros(l.delta_weights.shape)
                l.delta_biases = numpy.zeros(l.delta_biases.shape)
            except:
                pass

    def predict(self,X):
        out = []
        if X.shape != self.layers[0].input_shape:
            for x in X:
                out.append(self.feedforward(x, train=False))
        else:
            out.append(self.feedforward(X, train=False))
        return numpy.array(out)

    def save_model(self, path='model.json'):
        dict_model = {}
        to_save = ['activation', 'biases', 'filters', 'input_shape', 'isbias',
                   'kernel_size', 'kind', 'name', 'neurons', 'output_shape',
                   'padding', 'parameters', 'prob', 'stride', 'weights']
        for l in self.layers:
            current_layer = vars(l)
            values = {'type': str(type(l).__name__)}
            for key, value in current_layer.items():
                if key in to_save:
                    if key in ['weights', 'biases']:
                        try:
                            value = value.tolist()
                        except:
                            value = float(value)
                    if type(value) == numpy.int32:
                        value = float(value)
                    if key == 'input_shape' or key == 'output_shape':
                        try:
                            value = tuple(value)
                        except:
                            pass
                    values[key] = value
            dict_model[l.name] = values
        json_dict = json.dumps(dict_model)
        with open(path, mode='w') as f:
            f.write(json_dict)
        print('\nModel Saved.')

    def load_model(self, path='model.json'):
        layers = {"Dense": Dense, "Conv2d": Conv2d, "Dropout": Dropout,
                  "Flatten": Flatten, "Pool2d": Pool2d}
        with open(path, 'r') as f:
            dict_model = json.load(f)
            for value in dict_model.values():
                if isinstance(value, dict):
                    layer_type = value.pop('type')
                    layer = layers[layer_type]()
                    for key, val in value.items():
                        val = numpy.array(val) if key == 'weights' else val
                        val = numpy.array(val) if key == 'biases' else val
                        val = int(val) if key == 'filters' else val
                        val = int(val) if key == 'parameters' else val
                        try:
                            val = tuple(val) if key == 'input_shape' else val
                            val = tuple(val) if key == 'output_shape' else val
                        except:
                            val = int(val)
                        else:
                            pass
                        layer.__dict__[key] = val
                    if type(layer).__name__ in ["Conv2d","Dense"]:
                        layer.delta_weights = numpy.zeros(layer.weights.shape)
                        layer.delta_biases = numpy.zeros(layer.biases.shape)
                    self.layers.append(layer)
            print('Model loaded...')


if __name__ == '__main__':
    from keras.datasets import mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x = x_train.reshape(-1, 28 * 28)
    x = (x - x.mean(axis=1).reshape(-1, 1)) / x.std(axis=1).reshape(-1, 1)
    x = x.reshape(-1, 28, 28, 1)
    y = pd.get_dummies(y_train).to_numpy()
    xt = x_test.reshape(-1, 28 * 28)
    xt = (xt - xt.mean(axis=1).reshape(-1, 1)) / xt.std(axis=1).reshape(-1, 1)
    xt = xt.reshape(-1, 28, 28, 1)
    yt = pd.get_dummies(y_test).to_numpy()

    m = CNN()
    m.add(Conv2d(input_shape=(28, 28, 1), filters=4, kernel_size=(3,3)))
    m.add(Pool2d(kernel_size=(2, 2)))
    m.add(Conv2d(filters=8, kernel_size=(3, 3)))
    m.add(Dropout(0.2))
    m.add(Flatten())
    m.add(Dense(neurons=10, activation='softmax'))
    m.compile_model(lr=0.001, opt="adam", loss="mse")
    m.summary()
    #m.train(x[:100], y[:100], epochs=100, batch_size=10, val_x=xt[:10], val_y=yt[:10])
    #m.save_model()

    n = CNN()
    n.load_model()
    m.summary()
    n.compile_model(lr=0.001, opt="adam", loss="mse")
    n.train(x[:100], y[:100], epochs=1, batch_size=10, val_x=xt[:10], val_y=yt[:10])