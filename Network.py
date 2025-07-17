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
        self.old_weights = None                                      # We need the old weights and biases to update earlier layers.
        self.old_biases = None
        self.forward_output = None
        self.updated_params = None

    def forward_pass(self, input, weights):
        self.forward_output = np.dot(input, weights) + self.biases
    
    def updating_weights_biases(self, layerWeights, layerBiases):
        self.old_weights = self.weights
        self.old_biases = self.biases
        self.weights = layerWeights
        self.biases = layerBiases
        self.updated_params =  self.weights, self.biases, self.old_weights, self.old_biases



class RELU_Activation:
    # RELU either returns 0 if input is less than or equal 0. Otherwise return input itself.
    def activate(self, inputs):
        # Maybe have an self.input to save the input values. They are needed for back propgations.
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
 
    # The derivative of RELU can be added here.
    # The derivative of RELU returns 1 if the value is greater than 0. Otherwise it returns 0. This indicates which neuron was active.
    def backward(self, inputs):
        self.derivative  = np.where(inputs > 0, 1.0, 0.0)

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
        return Y_Predicted_Minus_Y

class Derivative_YPredicted_Over_Derivative_Z:
    def calculate(self, y_predicted):
        # YPredicted(1 - YPredicted)
        One_Minus_Y_Predicted =  y_predicted * (np.subtract(1, y_predicted))
        return One_Minus_Y_Predicted

class Derivative_Z_Over_Derivative_W:
    def calculate(self, output_input):
        # transposed X 
        return np.transpose(output_input)

class Calculate_Delta:
    def calculate(self, l_over_ypredicted, ypredicted_over_z):
        delta =  np.dot(l_over_ypredicted, ypredicted_over_z)
        return delta


class Derivative_L_over_Derivative_w:
    def calculate(self, xj, delta):
        multiplying_with_delta = np.dot(xj, delta)
        return multiplying_with_delta

class NewWeights: 
    def calculate(self, derivative_l_over_derivative_w, learningrate, layerweights):
        multiply =  np.dot(learningrate, derivative_l_over_derivative_w)
        updateweight = np.subtract(layerweights, multiply)
        return updateweight

class NewBiases: 
    def calculate(self, derivative_l_over_derivative_b, learningrate, layerbiases):
        # The only difference between calculating weights and biases is the sum we calculate in bias. This is done because original biases have a different shape than the calculated bias. 
        multiply =  np.dot(learningrate, derivative_l_over_derivative_b)
        summing = np.sum(multiply, axis=0, keepdims=True)
        newBiases = np.subtract(layerbiases, summing)
        return newBiases


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
    hiddenlayer1_zvalue = hiddenlayer1.forward_output
    hiddenlayer_activation.activate(hiddenlayer1_zvalue)
    hiddenlayer1_activate_output = hiddenlayer_activation.output
    print("Hidden Layer's 1 after activation function: \n", hiddenlayer1_activate_output)

    #Second layer compute
    hiddenlayer2.forward_pass(hiddenlayer_activation.output, hiddenlayer2.weights)
    hiddenlayer2_zvalue = hiddenlayer2.forward_output
    hiddenlayer_activation2.activate(hiddenlayer2_zvalue)
    hiddenlayer2_activate_output = hiddenlayer_activation2.output
    print("Hidden's Layer's 2 after activation: \n", hiddenlayer2_activate_output)

    #Output layer compute
    print("Output's input: \n", hiddenlayer_activation2.output)

    output_layer.forward_pass(hiddenlayer_activation2.output, output_layer.weights)
    print("Output's weights: \n", output_layer.weights)
    print("Output's biases: \n", output_layer.biases)

    outputlayer_activation.activate(output_layer.forward_output)
    print("Y_Predicted_Values: \n", outputlayer_activation.output)
    
    loss.calculate(outputlayer_activation.output, one_hot)
    print("Loss of this pass: \n", loss.meanloss)


    #Backward passing values for output

    # Equation for output layer weight update.
    # dL/dWo = dL/dYpredicted * dYpredicted/dZo * dZo/dWo
    # dL/dWo = dZo/dWo * deltaO                            Here deltaO is created by matrix multiplying dL/dYpredicted * dYpredicted/dZo
    # dL/dWo = deltaOutput                                 Here deltaOutput is achieved after transposing dZo/dWo and then matrix multiplying with deltaO

    
    l_over_ypredicted = derivative_l_over_derivative_ypredicted.calculate(outputlayer_activation.output, one_hot)
    print("Derivative of L respect to Y Predicted: \n", l_over_ypredicted)

    ypredicted_over_z = derivative_ypredicted_over_derivative_z.calculate(outputlayer_activation.output)
    print("Derivative of YPredicted respect to Z: \n", ypredicted_over_z)
    
    z_over_w = derivative_z_over_derivative_w.calculate(hiddenlayer_activation2.output)
    print("Derivative of Z respect to W: \n", z_over_w)

    delta_value_l_over_ypredicted_times_ypredicted_over_zo = calculate_delta.calculate(l_over_ypredicted, ypredicted_over_z) # Derivative of L over Derivative of ypredicted * derivative of ypredicted over derivative of zo
    # print("Delta value for Output Layer: \n", delta_value_l_over_ypredicted_times_ypredicted_over_zo)

    l_over_w = derivative_l_over_derivative_w.calculate(z_over_w, delta_value_l_over_ypredicted_times_ypredicted_over_zo)
    print("Derivative of L respect to W: \n", l_over_w)

    layer_new_weights = newWeights.calculate(l_over_w, learningrate, output_layer.weights)
    print("New Weights for output layer: \n", layer_new_weights)


    layer_new_biases = newBiases.calculate(delta_value_l_over_ypredicted_times_ypredicted_over_zo, learningrate, output_layer.biases)
    print("New biases for output layer: \n", layer_new_biases)

    output_layer.updating_weights_biases(layer_new_weights, layer_new_biases)
    outputlayer_new_weights, outputlayer_new_biases, outputlayer_old_weights, outputlayer_old_biases = output_layer.updated_params
    print('Outputlayer_New_Weights: \n', outputlayer_new_weights)
    print('Outputlayer_New_Biases: \n', outputlayer_new_biases)
    print('Outputlayer_old_Weights: \n', outputlayer_old_weights)
    print('Outputlayer_old_Biases: \n', outputlayer_old_biases)

    # Backward passing values for hidden layer2
    print("Hidden Layer two's weights: \n", hiddenlayer2.weights)
    print("Hidden Layer two's biases: \n", hiddenlayer2.biases)

    # Equation for hidden layer 2 weight update.
    # dL/dWh2 = dL/dYpredicted * dYpredicted/dZo * dZo/dAh2 * dAh2/ dZh2 * dZh2/dWh2
    # dL/dWh2 = deltaO * dZo/dAh2 * dAh2/ dZh2 * dZh2/dWh2                            Here deltaO is achieved by matrix multiplying dL/dYpredicted * dYpredicted/dZo
    # dL/dWh2 = deltaH * dAh2/ dZh2 * dZh2/dWh2                                       Here deltaH is achieved by matrix multiplying deltaO with dZo/dAh2(transposed value)
    # dL/dWh2 = dZh2/dWh2 * DeltaF                                                    Here deltaF is achieved by applying elemntwise operation between deltaH * dAh2/ dZh2
    # dL/dWh2 = deltaHidden2                                                          Here deltaHidden2 is achieved by transposing dZh2/dWh2 and matrix multiplying by DeltaF
   
    derivative_zo_derivative_ah = derivative_z_over_derivative_w.calculate(output_layer.old_weights)
    hiddenlayer_activation2.backward(hiddenlayer2_zvalue)
    print("Derv RELU: \n", hiddenlayer_activation2.derivative)
    
    derivative_ah_over_zh = hiddenlayer_activation2.derivative
    print("Deltavalue_output: \n", delta_value_l_over_ypredicted_times_ypredicted_over_zo)
    print("Derivative_zo_over_ah: \n", derivative_zo_derivative_ah)
    print("Original Derivative_ah_over_zh: \n", hiddenlayer_activation2.derivative)
    print("Derivative_ah_over_zh: \n", derivative_ah_over_zh)
    # need to calculater z over w but for hidden2.
    derivative_zh_over_wh = derivative_z_over_derivative_w.calculate(hiddenlayer1_activate_output)
    print("Derivative_zh_over_wh: \n", derivative_zh_over_wh)
    delta_value_times_zo_over_ah = calculate_delta.calculate(delta_value_l_over_ypredicted_times_ypredicted_over_zo, 
                                                                        derivative_zo_derivative_ah)
    print("Delta_value_times_zo_over_ah: \n", delta_value_times_zo_over_ah)
    delta_value_times_zo_over_ah_times_ah_over_zh = delta_value_times_zo_over_ah * derivative_ah_over_zh
    print("Delta_value_times_zo_over_ah_times_ah_over_zh: \n", delta_value_times_zo_over_ah_times_ah_over_zh)

    delta_value_times_zo_over_ah_times_ah_over_zh_times_zh_over_wh = derivative_l_over_derivative_w.calculate(
        derivative_zh_over_wh, delta_value_times_zo_over_ah_times_ah_over_zh)
    
    print("Delta_value_times_zo_over_ah_times_ah_over_zh_times_zh_over_wh: \n", delta_value_times_zo_over_ah_times_ah_over_zh_times_zh_over_wh)

    hidden_layer2_new_weights = newWeights.calculate(delta_value_times_zo_over_ah_times_ah_over_zh_times_zh_over_wh
                                                    , learningrate, hiddenlayer2.weights)
    print("Hidden_layer_new_weights: \n", hidden_layer2_new_weights)


    # Equation for hidden layer 2 biases update.
    # dL/dBh2 = dL/dYpredicted * dYpredicted/dZo * dZo/dAh2 * dAh2/ dZh2 * dZh2/dBh2
    # dL/dBh2 = deltaO * dZo/dAh2 * dAh2/ dZh2 * dZh2/dBh2                            Here deltaO is achieved by matrix multiplying dL/dYpredicted * dYpredicted/dZo
    # dL/dBh2 = deltaH * dAh2/ dZh2 * dZh2/dBh2                                       Here deltaH is achieved by matrix multiplying deltaO with dZo/dAh2(transposed value)
    # dL/dBh2 = DeltaF * dZh2/dBh2                                                    Here deltaF is achieved by applying elemntwise operation between deltaH * dAh2/ dZh2
    # dL/dBh2 = DeltaF                                                                Here the answer is just DeltaF because dZh2/dBh2 is just 1.

    # Need these values to calculate bias.
    delta_value_times_zo_over_ah
    delta_value_times_zo_over_ah_times_ah_over_zh

    # bias equation
    # b = b - 0.01(calculated bias avg per column)


    # TO DO:
    # Rename the variables on the hidden layer 2 to match the equation
    # Once done work on the calculating the biases for hidden layer 2.


if __name__ == "__main__":
    main()