import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class Layer_Dense():

    def __init__(self, n_features, n_neurons):
        # initialize weights and biases
        self.weights= 0.01 * np.random.randn(n_features, n_neurons)
        self.biases= np.zeros((1, n_neurons))

    def forward(self, inputs):
        # remember input to be used in backprop
        self.inputs= inputs 
        # calculate output values
        self.output= np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        # gradient on parameters
        self.dweights= np.dot(self.inputs.T, dvalues)
        self.dbiases= np.sum(dvalues, axis= 0, keepdims= True)
        # gradient on values
        self.dinputs= np.dot(dvalues, self.weights.T)

class Activation_ReLU:

    def forward(self, inputs):
        # remember input to be used in backprop
        self.inputs= inputs 
        self.output= np.maximum(0, inputs)

    def backward(self, dvalues):
        # create a copy of original var
        # because we need to modify it
        self.dinputs= dvalues.copy()
        # 0 gradient where input values where negative
        self.dinputs[self.inputs <= 0] = 0

class Activation_Softmax:

    def forward(self, inputs):
        # remember input values
        self.inputs= inputs
        # get unnormalized probas
        exp= np.exp(inputs - np.max(inputs, axis= 1, 
                                    keepdims= True))
        # normalize probas
        probas= exp / np.sum(exp, axis= 1, keepdims=True)
        self.output= probas

    def backward(self, dvalues):
        # create uninitialized array
        self.dinputs= np.empty_like(dvalues)

        # enumerate outputs and gradients
        for index, (single_output, single_dvalues) in \
            enumerate(zip(self.output, dvalues)):
            # flatten output array
            single_output= single_output.reshape(-1, 1)
            # calculate jacobian matrix of the output
            jacobian_matrix= np.diagflat(single_output) - \
                np.dot(single_output, single_output.T)
            # calculate sample-wise gradient
            # add it to the aray of sample gradient
            self.dinputs[index]= np.dot(jacobian_matrix, 
                                        single_dvalues)

# common loss class
class Loss:

    def calculate(self, y, output):
        # loss for all samples
        sample_losses= self.forward(y, output)
        # mean loss 
        data_loss= np.mean(sample_losses)
        return data_loss
    
class Loss_CategoricalCrossentropy(Loss):

    def forward(self, y_true, y_pred):
        # calculate number of samples in a batch
        n_samples= y_true.shape[0]
        # check the shape of y_true
        n_dims= len(y_true.shape)
        # clip the values to prevent division by 0
        y_pred_clipped= np.clip(y_pred, 1e-7, 1-1e-7)
        if n_dims == 1:
            # for categorical labels
            confidences= y_pred_clipped[range(n_samples), y_true]
        else: 
            # for one-hot labels
            confidences= np.sum(y_true * y_pred_clipped, 
                                axis= 1, 
                                keepdims= True)
        neg_log= -np.log(confidences)
        return neg_log
    
    def backward(self, dvalues, y_true):
        # calculate number of samples
        n_samples= len(dvalues)
        # number of labels in every sample
        labels= len(dvalues[0])
        # if labels are sparse, turn them into one-hot
        if len(y_true.shape) == 1:
            y_true= np.eye(labels)[y_true]
        # calculate gradient
        self.dinputs= -y_true / dvalues
        # normalize gradient
        self.dinputs= self.dinputs / n_samples


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
