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
        self.updated_params = None

        # Initializing forward and backward calculated variables.
        # Example: dL/dWh2 = dL/dYpredicted * dYpredicted/dZo * dZo/dAh2 * dAh2/ dZh2 * dZh2/dWh2

        self.z = None
        self.secondLastProduct = None


    def forward_pass(self, input, weights):
        self.z = np.dot(input, weights) + self.biases

    def updating_weights_biases(self, layerWeights, layerBiases):
        self.old_weights = self.weights
        self.old_biases = self.biases
        self.weights = layerWeights
        self.biases = layerBiases
        self.updated_params =  self.weights, self.biases, self.old_weights, self.old_biases



class RELU_Activation:
    # RELU either returns 0 if input is less than or equal 0. Otherwise return input itself.
    def activate(self, inputs):
        # We have an self.input to save the input values. They are needed for back propagations.
        self.inputs = inputs
        self.a = np.maximum(0, inputs)
 
    # The derivative of RELU returns 1 if the value is greater than 0. Otherwise it returns 0. This indicates which neuron was active.
    def backward(self, inputs):
        self.da = np.where(inputs > 0, 1.0, 0.0)

class SoftMax_Activation:
    def activate(self, inputs): 
        get_max_each_row = np.max(inputs, axis = 1, keepdims=True)
        set_to_zero = np.subtract(inputs, get_max_each_row)
        x = np.exp(set_to_zero)
        y = np.sum(x, axis=1, keepdims=True)
        normval = x / y 
        self.a = normval
        # exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # prob = exp_values/np.sum(exp_values, axis=1, keepdims=True)
        # self.output = prob
    def backward(self, y_predicted):
        # YPredicted(1 - YPredicted)
        One_Minus_Y_Predicted =  y_predicted * (np.subtract(1, y_predicted))
        return One_Minus_Y_Predicted
    
class Categorical_Loss:
    def calculate(self, y_pred, y_true):
        y = np.sum(y_pred * y_true, axis=1)
        natural_log = -np.log(y)
        takemean = np.mean(natural_log)
        self.meanloss = takemean

    def backward(self, y_predicted, y):
        # YPredicted - y
        # Here y is one hot encoded values.
        Y_Predicted_Minus_Y =  np.subtract(y_predicted, y)
        return Y_Predicted_Minus_Y

class Transposed:
    def calculate(self, output_input):
        # transposed X 
        return np.transpose(output_input)

class DotProduct:
    def calculate(self, l_over_ypredicted, ypredicted_over_z):
        delta =  np.dot(l_over_ypredicted, ypredicted_over_z)
        return delta


class DotProductFlipped:
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

    # Remember input layer exists but it does not have any weights. Its pupose is to serve the input to the first hidden layer.
    hiddenlayer1 = Layer_Creation(4, 3)
    hiddenlayer2 = Layer_Creation(3, 4)
    output_layer = Layer_Creation(4, 3)

    hiddenlayer_activation = RELU_Activation()
    hiddenlayer_activation2 = RELU_Activation()
    outputlayer_activation = SoftMax_Activation()

    loss = Categorical_Loss()

    #Backward pass initialization
    # derivative_l_over_derivative_ypredicted = Derivative_L_Over_Derivative_YPredicted()
    # derivative_ypredicted_over_derivative_z = Derivative_YPredicted_Over_Derivative_Z()
    transposing = Transposed()
    dotproduct = DotProduct()
    dotproductflipped = DotProductFlipped()
    newWeights = NewWeights()
    newBiases = NewBiases()
    learningrate = 0.01

    #Forward passing values

    #First layer compute
    hiddenlayer1.forward_pass(input, hiddenlayer1.weights)
    hiddenlayer_activation.activate(hiddenlayer1.z)
    print("Hidden Layer's 1 after activation function: \n", hiddenlayer_activation.a)

    #Second layer compute
    hiddenlayer2.forward_pass(hiddenlayer_activation.a, hiddenlayer2.weights)
    hiddenlayer_activation2.activate(hiddenlayer2.z)
    print("Hidden's Layer's 2 after activation: \n", hiddenlayer_activation2.a)

    #Output layer compute
    output_layer.forward_pass(hiddenlayer_activation2.a, output_layer.weights)
    outputlayer_activation.activate(output_layer.z)
    print("Y_Predicted_Values: \n", outputlayer_activation.a)
    
    loss.calculate(outputlayer_activation.a, one_hot)
    print("Loss of this pass: \n", loss.meanloss)


    #Backward passing values for output

    # Equation for output layer weight update.
    # dL/dWo = dL/dYpredicted * dYpredicted/dZo * dZo/dWo
    # dL/dWo = dZo/dWo * deltaO                            Here deltaO is created by matrix multiplying dL/dYpredicted * dYpredicted/dZo
    # dL/dWo = deltaOutput                                 Here deltaOutput is achieved after transposing dZo/dWo and then matrix multiplying with deltaO

    # Weight update equation
    # w = w - 0.01(dL/dWo)

    l_over_ypredicted = loss.backward(outputlayer_activation.a, one_hot)
    print("Derivative of L respect to Y Predicted: \n", l_over_ypredicted)

    ypredicted_over_z = outputlayer_activation.backward(outputlayer_activation.a)
    print("Derivative of YPredicted respect to Z: \n", ypredicted_over_z)
    
    z_over_w = transposing.calculate(hiddenlayer_activation2.a)
    print("Derivative of Z respect to W: \n", z_over_w)

    delta_value_l_over_ypredicted_times_ypredicted_over_zo = dotproduct.calculate(l_over_ypredicted, ypredicted_over_z) # Derivative of L over Derivative of ypredicted * derivative of ypredicted over derivative of zo
    # print("Delta value for Output Layer: \n", delta_value_l_over_ypredicted_times_ypredicted_over_zo)

    l_over_w = dotproductflipped.calculate(z_over_w, delta_value_l_over_ypredicted_times_ypredicted_over_zo)
    print("Derivative of L respect to W: \n", l_over_w)

    layer_new_weights = newWeights.calculate(l_over_w, learningrate, output_layer.weights)
    layer_new_biases = newBiases.calculate(delta_value_l_over_ypredicted_times_ypredicted_over_zo, learningrate, output_layer.biases)

    output_layer.updating_weights_biases(layer_new_weights, layer_new_biases)
    outputlayer_new_weights, outputlayer_new_biases, outputlayer_old_weights, outputlayer_old_biases = output_layer.updated_params
    print('Outputlayer_New_Weights: \n', outputlayer_new_weights)
    print('Outputlayer_New_Biases: \n', outputlayer_new_biases)
    print('Outputlayer_old_Weights: \n', outputlayer_old_weights)
    print('Outputlayer_old_Biases: \n', outputlayer_old_biases)

    # Backward passing values for hidden layer2

    # Equation for hidden layer 2 weight gradient.
    # dL/dWh2 = dL/dYpredicted * dYpredicted/dZo * dZo/dAh2 * dAh2/ dZh2 * dZh2/dWh2
    # dL/dWh2 = deltaO * dZo/dAh2 * dAh2/ dZh2 * dZh2/dWh2                            Here deltaO is achieved by matrix multiplying dL/dYpredicted * dYpredicted/dZo
    # dL/dWh2 = deltaH * dAh2/ dZh2 * dZh2/dWh2                                       Here deltaH is achieved by matrix multiplying deltaO with dZo/dAh2(transposed value)
    # dL/dWh2 = dZh2/dWh2 * DeltaF                                                    Here deltaF is achieved by applying elemntwise operation between deltaH * dAh2/ dZh2
    # dL/dWh2 = deltaHidden2                                                          Here deltaHidden2 is achieved by transposing dZh2/dWh2 and matrix multiplying by DeltaF
   
    # Weight update equation
    # w = w - 0.01(dL/dWh2)

    derivative_zo_derivative_ah2 = transposing.calculate(output_layer.old_weights)
    hiddenlayer_activation2.backward(hiddenlayer2.z)
    print("Derv RELU: \n", hiddenlayer_activation2.da)
    
    derivative_ah2_over_zh2 = hiddenlayer_activation2.da
    print("Deltavalue_output: \n", delta_value_l_over_ypredicted_times_ypredicted_over_zo)
    print("Derivative_zo_over_ah: \n", derivative_zo_derivative_ah2)
    print("Original Derivative_ah_over_zh: \n", hiddenlayer_activation2.da)
    print("Derivative_ah_over_zh: \n", derivative_ah2_over_zh2)
    # need to calculater z over w but for hidden2.
    derivative_zh2_over_wh2 = transposing.calculate(hiddenlayer_activation.a)
    print("Derivative_zh_over_wh: \n", derivative_zh2_over_wh2)
    delta_value_times_zo_over_ah2 = dotproduct.calculate(delta_value_l_over_ypredicted_times_ypredicted_over_zo, 
                                                                        derivative_zo_derivative_ah2)
    print("Delta_value_times_zo_over_ah: \n", delta_value_times_zo_over_ah2)
    delta_value_times_zo_over_ah2_times_ah2_over_zh2 = delta_value_times_zo_over_ah2 * derivative_ah2_over_zh2
    print("Delta_value_times_zo_over_ah_times_ah_over_zh: \n", delta_value_times_zo_over_ah2_times_ah2_over_zh2)

    delta_value_times_zo_over_ah2_times_ah2_over_zh2_times_zh2_over_wh2 = dotproductflipped.calculate(
        derivative_zh2_over_wh2, delta_value_times_zo_over_ah2_times_ah2_over_zh2)
    
    print("Delta_value_times_zo_over_ah_times_ah_over_zh_times_zh_over_wh: \n", delta_value_times_zo_over_ah2_times_ah2_over_zh2_times_zh2_over_wh2)

    hidden_layer2_new_weights = newWeights.calculate(delta_value_times_zo_over_ah2_times_ah2_over_zh2_times_zh2_over_wh2
                                                    ,learningrate, hiddenlayer2.weights)
    
    # Equation for hidden layer 2 bias gradient.
    # dL/dBh2 = dL/dYpredicted * dYpredicted/dZo * dZo/dAh2 * dAh2/ dZh2 * dZh2/dBh2
    # dL/dBh2 = deltaO * dZo/dAh2 * dAh2/ dZh2 * dZh2/dBh2                            Here deltaO is achieved by matrix multiplying dL/dYpredicted * dYpredicted/dZo
    # dL/dBh2 = deltaH * dAh2/ dZh2 * dZh2/dBh2                                       Here deltaH is achieved by matrix multiplying deltaO with dZo/dAh2(transposed value)
    # dL/dBh2 = DeltaF * dZh2/dBh2                                                    Here deltaF is achieved by applying elemntwise operation between deltaH * dAh2/ dZh2
    # dL/dBh2 = DeltaF                                                                Here the answer is just DeltaF because dZh2/dBh2 is just 1.

    # Bias update equation
    # b = b - 0.01(calculated bias avg per column. This is done over dL/dBh2.)
    
    hiddenlayer2_bias = newBiases.calculate(delta_value_times_zo_over_ah2_times_ah2_over_zh2, learningrate, hiddenlayer2.biases)
    hiddenlayer2.updating_weights_biases(hidden_layer2_new_weights, hiddenlayer2_bias)
    hiddenlayer2_new_weights, hiddenlayer2_new_biases, hiddenlayer2_old_weights, hiddenlayer2_old_biases = hiddenlayer2.updated_params
    print('Hiddenlayer2_New_Weights: \n', hiddenlayer2_new_weights)
    print('Hiddenlayer2_New_Biases: \n', hiddenlayer2_new_biases)
    print('Hiddenlayer2_old_Weights: \n', hiddenlayer2_old_weights)
    print('Hiddenlayer2_old_Biases: \n', hiddenlayer2_old_biases)


    # Equation for hidden layer 1 weight gradient.
    # dL/dWh1 = dL/dYpredicted * dYpredicted/dZo * dZo/dAh2 * dAh2/dZh2 * dZh2/dAh1 * dAh1/dZh1 * dZh1/dWh1


    # dL/dYpredicted * dYpredicted/dZo * dZo/dAh2 * dAh2/ dZh2
    delta_value_times_zo_over_ah2_times_ah2_over_zh2

    derivative_zh2_over_ah1 = transposing.calculate(hiddenlayer2.old_weights)
    
    hiddenlayer_activation.backward(hiddenlayer1.z)
    derivative_ah1_over_zh1 = hiddenlayer_activation.da

    derivative_zh1_over_wh1 = transposing.calculate(input)

    delta_value_times_zo_over_ah2_times_ah2_over_zh2_times_zh2_over_ah1  = dotproduct.calculate(delta_value_times_zo_over_ah2_times_ah2_over_zh2, 
                                                                                           derivative_zh2_over_ah1)

    delta_value_times_zo_over_ah2_times_ah2_over_zh2_times_zh2_over_ah1_times_ah1_over_zh1 = delta_value_times_zo_over_ah2_times_ah2_over_zh2_times_zh2_over_ah1 * derivative_ah1_over_zh1
    
    delta_value_times_zo_over_ah2_times_ah2_over_zh2_times_zh2_over_ah1_times_ah1_over_zh1_times_zh1_over_wh1 = dotproductflipped.calculate(
        derivative_zh1_over_wh1, delta_value_times_zo_over_ah2_times_ah2_over_zh2_times_zh2_over_ah1_times_ah1_over_zh1,
    )

    hidden_layer1_new_weights = newWeights.calculate(delta_value_times_zo_over_ah2_times_ah2_over_zh2_times_zh2_over_ah1_times_ah1_over_zh1_times_zh1_over_wh1,
                                                     learningrate,
                                                     hiddenlayer1.weights)

    # Equation for hidden layer 1 bias gradient.
    # dL/dBh2 = dL/dYpredicted * dYpredicted/dZo * dZo/dAh2 * dAh2/ dZh2 * dZh2/dAh1 * dAh1/dZh1 * dZh1/dBh1

    hidden_layer1_new_bias = newBiases.calculate(delta_value_times_zo_over_ah2_times_ah2_over_zh2_times_zh2_over_ah1_times_ah1_over_zh1, 
                                            learningrate, hiddenlayer1.biases)

    hiddenlayer1.updating_weights_biases(hidden_layer1_new_weights, hidden_layer1_new_bias)

    hiddenlayer1_new_weights, hiddenlayer1_new_biases, hiddenlayer1_old_weights, hiddenlayer1_old_biases = hiddenlayer1.updated_params

    print('Hiddenlayer1_New_Weights: \n', hiddenlayer1_new_weights)
    print('Hiddenlayer1_New_Biases: \n', hiddenlayer1_new_biases)
    print('Hiddenlayer1_old_Weights: \n', hiddenlayer1_old_weights)
    print('Hiddenlayer1_old_Biases: \n', hiddenlayer1_old_biases)

    # Figuring out pattern for calculating gradients for each layer.
    
    # Seems like for output layer first two derivatives get dot product(DeltaA) and the last gets transposed. Once that is transposed 
    # then it comes at the front and gets dot product by DeltaA.

    # For hidden layer2
    # We can just use the DeltaA value that we calculated from output. DeltaA then gets dot product by the next derivative(transposed) and 
    # becomes DeltaB. DeltaB then gets elementwise operation with the next derivative and becomes DeltaC. We finally take the next derivative(transposed)
    # and bring it at the front and dot product with DeltaC.


    # dL/dWh1 = dL/dYpredicted * dYpredicted/dZo * dZo/dAh2 * dAh2/dZh2 * dZh2/dAh1 * dAh1/dZh1 * dZh1/dWh1

    # For hidden layer
    # We can also use the DeltaC value and dot product with the next derivative(transposed) which becaomses DeltaD. DeltaD then gets elementwise
    # operation with the next derivative which then becomes DeltaE. The next derivative(transposed) comes at the front and gets dot product with
    # DeltaE.

    # Output 3 
    # Last(transposed) dot product after the first two dot

    # hidden 2
    # Use value from output. Dot that with the tranposed value. Elementwise that with the next value.
    # Finally bring the last tranposed at the front and dot it with the result of elementwise value.

    # hidden
    # Use value from hidden2. Dot that with the tranposed value. Elementwise that with the next value.
    # Finally bring the last transposed at the front and dot it with the result of elementwise value.


    # Maybe a stack. 
    # We can use a stack to push useful information and when back propgating pop information out of it.
    # Popping from stack goes last in first out which is useful when it comes to calculating each layer.

if __name__ == "__main__":
    main()