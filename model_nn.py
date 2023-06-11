import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class Layer_Dense():

    def __init__(self, n_features, n_neurons):
        self.weight= 0.01 * np.random.randn(n_features, n_neurons)
        self.bias= np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output= np.dot(inputs, self.weight) + self.bias

class Activation_ReLU:

    def forward(self, inputs):
        self.output= np.maximum(0, inputs)

class Activation_Softmax:

    def forward(self, inputs):
        exp= np.exp(inputs - np.max(inputs, axis= 1, keepdims= True))
        probas= exp / np.sum(exp, axis= 1, keepdims=True)
        self.output= probas

class Loss:

    def calculate(self, y, output):
        sample_losses= self.forward(y, output)
        data_loss= np.mean(sample_losses)
        return data_loss
    
class Loss_CategoricalCrossentropy(Loss):

    def forward(self, y_true, y_pred):
        n_samples= y_true.shape[0]
        # check the shape of y_true
        n_dims= len(y_true.shape)
        y_pred_clipped= np.clip(y_pred, 1e-7, 1-1e-7)
        if n_dims == 1:
            confidences= y_pred_clipped[range(n_samples), y_true]
        else: 
            confidences= np.sum(y_true * y_pred_clipped, 
                                axis= 1, 
                                keepdims= True)
        neg_log= -np.log(confidences)
        return neg_log


if __name__ == "__main__":

    # create dataset
    n_samples= 100
    n_classes= 3
    X, y= spiral_data(samples= n_samples, classes= n_classes)
    n_features= X.shape[1]

    # layer 1 dense, 2 inputs (because 2 features), 3 neurons, relu
    layer1= Layer_Dense(n_features= n_features, n_neurons= 3)
    activ1= Activation_ReLU()

    # layer 2 dense, 3 inputs (because layer 1= 3 neurons), softmax
    # 3 outputs (because there are 3 classes)
    layer2= Layer_Dense(n_features= 3, n_neurons= n_classes)
    activ2= Activation_Softmax()

    # loss function
    loss_function= Loss_CategoricalCrossentropy()

    # perform forward pass
    layer1.forward(X)
    activ1.forward(layer1.output)
    layer2.forward(activ1.output)
    activ2.forward(layer2.output)

    # see the output
    print(activ2.output[:5])

    # calculate loss
    loss= loss_function.calculate(y, activ2.output)
    print(f'loss = {loss}')

    print(activ2.output)
