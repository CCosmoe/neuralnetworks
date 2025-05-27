import numpy as np

# def main():
#     input =  np.array([[1,    2,   3,  2.5],
#                        [2,     5,  -1,  2.0],
#                        [-1.5, 2.7, 3.3, 0.8]])
    
#     layer_weights = np.array([[0.2, 0.5, -0.26],
#                               [0.8,-0.91, -0.27],
#                               [-0.5, 0.26, 0.17],
#                               [1.0, -0.5, 0.87 ]])
    
#     matrix_multi = matrix_multiplication(input, layer_weights)




class Layer_Creation: 
    def __init__(self, inputs, neurons):
        self.weights = 0.10*(np.random.randn(inputs, neurons))
        self.biases = np.zeros((1, neurons))

    def forward_pass(self, input, weights):
        self.output = np.dot(input, weights) + self.biases

class RELU_Activation:
    # RELU either returns 0 if input is less than or equal 0. Otherwise return input itself.
    def activate(self, inputs):
        self.output = np.maximum(0, inputs)
 
class SoftMax_Activation:
    def activate(self, inputs): 
        get_max_each_row = np.max(inputs, axis = 1, keepdims=True)
        set_to_zero = np.subtract(inputs, get_max_each_row)
        x = np.exp(set_to_zero)
        y = np.sum(x, axis=1, keepdims=True)
        normval = x / y 
        self.output = normval
        # exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # prob = exp_values/np.sum(exp_values, axis=1, keepdims=True)
        # self.output = prob

class Categorical_Loss:
    def calculate(self, y_pred, y_true):
        y = np.sum(y_pred * y_true, axis=1)
        natural_log = -np.log(y)
        takemean = np.mean(natural_log)
        self.meanloss = takemean


class LossOf_L_Over_YPredicted:
    def calculate(self, y_predicted, y):
        # Here y is one hot encoded values.
        Y_Predicted_Minus_Y =  np.subtract(y_predicted, y)
        self.output = Y_Predicted_Minus_Y


def main():

    input =  np.array([[1,    15,   3,     30],
                       [63,   65,  -1,    500],
                       [-15, -15,   3.3, -100]])
    
    one_hot = np.array([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]])
    
    
    # Forward pass intialization
    hiddenlayer1 = Layer_Creation(4, 3)
    hiddenlayer2 = Layer_Creation(3, 4)
    output_layer = Layer_Creation(4, 3)
    hiddenlayer_activation = RELU_Activation()
    hiddenlayer_activation2 = RELU_Activation()
    outputlayer_activation = SoftMax_Activation()
    loss = Categorical_Loss()

    #Backward pass initialization
    derivative_l_over_ypredicted = LossOf_L_Over_YPredicted()

    
    #Forward passing values

    #First layer compute
    hiddenlayer1.forward_pass(input, hiddenlayer1.weights)
    hiddenlayer_activation.activate(hiddenlayer1.output)

    #Second layer compute
    hiddenlayer2.forward_pass(hiddenlayer_activation.output, hiddenlayer2.weights)
    hiddenlayer_activation2.activate(hiddenlayer2.output)
    
    #Output layer compute
    output_layer.forward_pass(hiddenlayer_activation2.output, output_layer.weights)
    outputlayer_activation.activate(output_layer.output)
    print("Y_Predicted_Values: \n", outputlayer_activation.output)
    
    loss.calculate(outputlayer_activation.output, one_hot)
    print("Loss of this pass: \n", loss.meanloss)



    #Backward passing values

    derivative_l_over_ypredicted.calculate(outputlayer_activation.output, one_hot)
    print("Derivative of L respect to Y Predicted: \n", derivative_l_over_ypredicted.output)

    
    #saving 


if __name__ == "__main__":
    main()