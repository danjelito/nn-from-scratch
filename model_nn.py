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

# combined softmax activation and crossentropy loss 
# for faster backward steps
class Activation_Softmax_Loss_CategoricalCrossentropy():

    # creates activation and loss function objects
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    def forward(self, inputs, y_true):
        # output layer's activation function
        self.activation.forward(inputs)
        # set the output
        self.output = self.activation.output
        # calculate and return loss value
        return self.loss.calculate(y_true, self.output)

    def backward(self, dvalues, y_true):
        # calculate number of samples
        n_samples = len(dvalues)
        # if labels are one-hot encoded,
        # turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        # copy so we can safely modify
        self.dinputs = dvalues.copy()
        # calculate gradient
        self.dinputs[range(n_samples), y_true] -= 1
        # normalize gradient
        self.dinputs = self.dinputs / n_samples


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

    # combine softmax activation with loss function
    loss_activation= Activation_Softmax_Loss_CategoricalCrossentropy()

    # perform forward pass
    layer1.forward(X)
    activ1.forward(layer1.output)
    layer2.forward(activ1.output)
    loss= loss_activation.forward(layer2.output, y)

    # see the output
    print(loss_activation.output[:5])

    # calculate loss
    print(f'loss = {loss}')

    # calculate accuracy from output of activation2 and targets
    # calculate values along first axis
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions==y)

    # Print accuracy
    print('acc:', accuracy)

    # Backward pass
    loss_activation.backward(loss_activation.output, y)
    layer2.backward(loss_activation.dinputs)
    activ1.backward(layer2.dinputs)
    layer1.backward(activ1.dinputs)

    # Print gradients
    print(layer1.dweights)
    print(layer1.dbiases)
    print(layer2.dweights)
    print(layer2.dbiases)
