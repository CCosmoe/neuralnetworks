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

#     print(matrix_multi)



class Layer_Creation: 
    def __init__(self, inputs, neurons):
        self.weights = 0.10*(np.random.randn(inputs, neurons))
        self.biases = np.zeros((1, neurons))
        self.forward_output = None
        self.updated_params = None

    def forward_pass(self, input, weights):
        self.forward_output = np.dot(input, weights) + self.biases
    
    def updating_weights_biases(self, layerWeights, layerBiases):
        self.weights = layerWeights
        self.biases = layerBiases
        self.updated_params =  self.weights, self.biases



class RELU_Activation:
    # RELU either returns 0 if input is less than or equal 0. Otherwise return input itself.
    def activate(self, inputs):
        # Maybe have an self.input to save the input values. They are needed for back propgations.
        self.output = np.maximum(0, inputs)
 
    # The derivative of RELU can be added here.
    # The derivative of RELU returns 1 if the value is greater than 0. Otherwise it returns 0. This indicates which neuron was active.

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


class Derivative_L_Over_Derivative_YPredicted:
    def calculate(self, y_predicted, y):
        # YPredicted - y
        # Here y is one hot encoded values.
        Y_Predicted_Minus_Y =  np.subtract(y_predicted, y)
        self.output = Y_Predicted_Minus_Y

class Derivative_YPredicted_Over_Derivative_Z:
    def calculate(self, y_predicted):
        # YPredicted(1 - YPredicted)
        One_Minus_Y_Predicted =  y_predicted * (np.subtract(1, y_predicted))
        self.output = One_Minus_Y_Predicted

class Derivative_Z_Over_Derivative_W:
    def calculate(self, output_input):
        # transposed X 
        transposed =  np.transpose(output_input)
        self.output = transposed

class Calculate_Delta:
    def calculate(self, l_over_ypredicted, ypredicted_over_z):
        delta =  np.dot(l_over_ypredicted, ypredicted_over_z)
        self.output = delta


class Derivative_L_over_Derivative_w:
    def calculate(self, xj, delta):
        multiplying_with_delta = np.dot(xj, delta)
        self.output = multiplying_with_delta

class NewWeights: 
    def calculate(self, derivative_l_over_derivative_w, learningrate, layerweights):
        multiply =  np.dot(learningrate, derivative_l_over_derivative_w)
        updateweight = np.subtract(layerweights, multiply)
        self.output = updateweight

class NewBiases: 
    def calculate(self, derivative_l_over_derivative_b, learningrate, layerbiases):
        # The only difference between calculating weights and biases is the sum we calculate in bias. This is done because original biases have a different shape than the calculated bias. 
        multiply =  np.dot(learningrate, derivative_l_over_derivative_b)
        summing = np.sum(multiply, axis=0, keepdims=True)
        newBiases = np.subtract(layerbiases, summing)
        self.output = newBiases


def main():

    input =  np.array([[1,    15,   3,     30],
                       [63,   65,  -1,    500],
                       [-15, -15,   3.3, -100]])
    
    one_hot = np.array([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]])
    
    
    # Forward pass intialization
    # 3 layers in total
    # Remember input layer exists but it does not have any weights. Its pupose is to serve the input to the first hidden layer.
    hiddenlayer1 = Layer_Creation(4, 3)
    hiddenlayer2 = Layer_Creation(3, 4)
    output_layer = Layer_Creation(4, 3)
    number_of_layers = 3  # number depends on how many layers we have. In this case 2 hidden layer and one output layer.

    hiddenlayer_activation = RELU_Activation()
    hiddenlayer_activation2 = RELU_Activation()
    outputlayer_activation = SoftMax_Activation()

    loss = Categorical_Loss()

    #Backward pass initialization
    derivative_l_over_derivative_ypredicted = Derivative_L_Over_Derivative_YPredicted()
    derivative_ypredicted_over_derivative_z = Derivative_YPredicted_Over_Derivative_Z()
    derivative_z_over_derivative_w = Derivative_Z_Over_Derivative_W()
    calculate_delta = Calculate_Delta()
    derivative_l_over_derivative_w = Derivative_L_over_Derivative_w()
    newWeights = NewWeights()
    newBiases = NewBiases()
    learningrate = 0.01

    #Forward passing values

    #First layer compute
    hiddenlayer1.forward_pass(input, hiddenlayer1.weights)
    hiddenlayer_activation.activate(hiddenlayer1.forward_output)

    #Second layer compute
    hiddenlayer2.forward_pass(hiddenlayer_activation.output, hiddenlayer2.weights)
    hiddenlayer_activation2.activate(hiddenlayer2.forward_output)
    
    #Output layer compute
    print("Output's input: \n", hiddenlayer_activation2.output)

    output_layer.forward_pass(hiddenlayer_activation2.output, output_layer.weights)
    print("Output's weights: \n", output_layer.weights)
    print("Output's biases: \n", output_layer.biases)

    outputlayer_activation.activate(output_layer.forward_output)
    print("Y_Predicted_Values: \n", outputlayer_activation.output)
    
    loss.calculate(outputlayer_activation.output, one_hot)
    print("Loss of this pass: \n", loss.meanloss)


    #Backward passing values

    derivative_l_over_derivative_ypredicted.calculate(outputlayer_activation.output, one_hot)
    l_over_ypredicted = derivative_l_over_derivative_ypredicted.output
    print("Derivative of L respect to Y Predicted: \n", l_over_ypredicted)

    derivative_ypredicted_over_derivative_z.calculate(outputlayer_activation.output)
    ypredicted_over_z = derivative_ypredicted_over_derivative_z.output
    print("Derivative of YPredicted respect to Z: \n", ypredicted_over_z)
    
    derivative_z_over_derivative_w.calculate(hiddenlayer_activation2.output)
    z_over_w = derivative_z_over_derivative_w.output
    print("Derivative of Z respect to W: \n", z_over_w)

    calculate_delta.calculate(l_over_ypredicted, ypredicted_over_z)
    delta_value = calculate_delta.output

    derivative_l_over_derivative_w.calculate(z_over_w, delta_value)
    l_over_w = derivative_l_over_derivative_w.output
    print("Derivative of L respect to W: \n", l_over_w)

    newWeights.calculate(l_over_w, learningrate, output_layer.weights)
    layer_new_weights = newWeights.output
    print("New Weights for layer: \n", layer_new_weights)


    newBiases.calculate(delta_value, learningrate, output_layer.biases)
    layer_new_biases = newBiases.output
    print("New biases for layer: \n", layer_new_biases)


    output_layer.updating_weights_biases(layer_new_weights, layer_new_biases)
    outputlayer_weights, outputlayer_biases = output_layer.updated_params
    print('Outputlayer_New_Weights: \n', outputlayer_weights)
    print('Outputlayer_New_Biases: \n', outputlayer_biases)


    # Got the math for updating hidden layers.
    # Got math for updating bias for hidden layers.
    # Next try the equations with actual numbers.
    # Once done with that figure out how to do this with a for loop or recursion and not manually.

if __name__ == "__main__":
    main()