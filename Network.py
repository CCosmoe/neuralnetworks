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
        self.inputs = None

        # Initializing forward and backward calculated variables.
        # Example: dL/dWh2 = dL/dYpredicted * dYpredicted/dZo * dZo/dAh2 * dAh2/ dZh2 * dZh2/dWh2

        self.z = None
        self.secondLastProduct = None


    def forward_pass(self, input):
        print("Current Input: \n", input)
        print("Current weights: \n", self.weights)
        self.inputs = input
        self.z = np.dot(input, self.weights) + self.biases
        print("Output after multiplying: \n", self.z)
        return self.z
    
    def secondlLastProductSetter(self, secondlastproduct):
        self.secondLastProduct = secondlastproduct

    def updating_weights_biases(self, layerWeights, layerBiases):
        self.old_weights = self.weights
        self.old_biases = self.biases
        self.weights = layerWeights
        self.biases = layerBiases
        self.updated_params =  self.weights, self.biases, self.old_weights, self.old_biases
        return self.updated_params


class RELU_Activation:
    # RELU either returns 0 if input is less than or equal 0. Otherwise return input itself.
    def forward_pass(self, inputs):
        # We have an self.input to save the input values. They are needed for back propagations.
        self.inputs = inputs
        self.a = np.maximum(0, inputs)
        print("Output after forward RELU: \n", self.a)
        return self.a

    # The derivative of RELU returns 1 if the value is greater than 0. Otherwise it returns 0. This indicates which neuron was active.
    def backward(self, inputs):
        self.da = np.where(inputs > 0, 1.0, 0.0)

class SoftMax_Activation:
    def forward_pass(self, inputs): 
        get_max_each_row = np.max(inputs, axis = 1, keepdims=True)
        set_to_zero = np.subtract(inputs, get_max_each_row)
        x = np.exp(set_to_zero)
        y = np.sum(x, axis=1, keepdims=True)
        normval = x / y 
        self.a = normval
        print("Output after forward Softmax: \n", self.a)
        return self.a
        # exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # prob = exp_values/np.sum(exp_values, axis=1, keepdims=True)
        # self.output = prob
    def backward(self, y_predicted):
        # YPredicted(1 - YPredicted)
        One_Minus_Y_Predicted =  y_predicted * (np.subtract(1, y_predicted))
        return One_Minus_Y_Predicted
    
class Categorical_Loss:
    def forward_pass(self, y_pred, y_true):
        y = np.sum(y_pred * y_true, axis=1)
        natural_log = -np.log(y)
        takemean = np.mean(natural_log)
        self.meanloss = takemean
        return self.meanloss

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
    def calculate(self, delta, xj):
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


class Container:
    def __init__(self):
        self.instances = []
        self.ypreds = None

    def add(self, instance):
        self.instances.append(instance)

    def forward(self, x, y_true):
        input = x
        for instance in self.instances:
            # We go inside this if statement last for calculating loss at the end. if not last then we 
            # created instances in wrong order.
            if isinstance(instance, Categorical_Loss):
                self.ypreds = input
                lossoutput = instance.forward_pass(input, y_true)
                return lossoutput
            else:
                output = instance.forward_pass(input)
                input = output

    # This is where we are going to create back propgation function.

    def backward(self, y_pred, y_true, learningRate):
        dotHelper = DotProduct()
        transposedHelper = Transposed()
        dotProductFlippedHelper = DotProductFlipped()
        newWeightsHelper = NewWeights()
        newBiasesHelper = NewBiases()

        gradient = None
        savingSecondLastProduct = None
        previousLayerWeights = None

        for instance in reversed(self.instances):
            if isinstance(instance, Categorical_Loss):
                gradient = instance.backward(y_pred, y_true)
                print("First iteration. Calculating gradient for loss: \n", gradient)

            elif isinstance(instance, SoftMax_Activation):
                softmaxGradient = instance.backward(y_pred)
                gradient = dotHelper.calculate(gradient, softmaxGradient)
                
                # Assigning local variable value
                savingSecondLastProduct = gradient
                print("Calculating Softmax: \n", gradient)

            elif isinstance(instance, RELU_Activation):
                transposing = transposedHelper.calculate(previousLayerWeights)
                gradient = dotHelper.calculate(savingSecondLastProduct, transposing)

            elif isinstance(instance, Layer_Creation): 
                layerGradient = transposedHelper.calculate(instance.inputs)
                gradient = dotProductFlippedHelper.calculate(gradient, layerGradient)

                newWeights = newWeightsHelper.calculate(gradient, learningRate, instance.weights)
                newBiases = newBiasesHelper.calculate(savingSecondLastProduct, learningRate, instance.biases)

                layerNewWeights, layerNewBiases, layerOldWeights, layerOldBiases = instance.updating_weights_biases(newWeights, newBiases)
                
                # Assigning local variable value
                previousLayerWeights = layerOldWeights
                print("layerOldWeights: \n", layerOldWeights)
                print("layerOldBiases: \n", layerOldBiases)
                print("layerNewWeights: \n", layerNewWeights)
                print("layerNewBiases: \n", layerNewBiases)



def main():

    input =  np.array([[1,    15,   3,     30],
                       [63,   65,  -1,    500],
                       [-15, -15,   3.3, -100]])
    
    one_hot = np.array([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]])
    
    

    # Wrapper
    container = Container()
    
    # Creating layers
    hiddenlayer1 = Layer_Creation(4, 3)
    hiddenlayer_activation = RELU_Activation()

    hiddenlayer2 = Layer_Creation(3, 4)
    hiddenlayer_activation2 = RELU_Activation()

    output_layer = Layer_Creation(4, 3)
    outputlayer_activation = SoftMax_Activation()
    loss = Categorical_Loss()

    learningRate = 0.01

    # Adding them to container
    container.add(hiddenlayer1)
    container.add(hiddenlayer_activation)

    container.add(hiddenlayer2)
    container.add(hiddenlayer_activation2)

    container.add(output_layer)
    container.add(outputlayer_activation)
    container.add(loss)

    # One epoch forward
    forward_output = container.forward(input, one_hot)
    print("This is loss: \n", forward_output)

    backward_output = container.backward(container.ypreds, one_hot, learningRate)
    # backward_output = container.backward()
    # This is where back propgation will be called.

if __name__ == "__main__":
    main()