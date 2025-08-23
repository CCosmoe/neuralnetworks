import numpy as np
import matplotlib.pyplot as plt

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
        # print("Current Input: \n", input)
        # print("Current weights: \n", self.weights)
        self.inputs = input
        self.z = np.dot(input, self.weights) + self.biases
        # print("Output after multiplying: \n", self.z)
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
        # print("Output after forward RELU: \n", self.a)
        return self.a

    # The derivative of RELU returns 1 if the value is greater than 0. Otherwise it returns 0. This indicates which neuron was active.
    def backward(self, inputs):
        self.da = np.where(inputs > 0, 1.0, 0.0)
        return self.da

class SoftMax_Activation:
    def forward_pass(self, inputs): 
        get_max_each_row = np.max(inputs, axis = 1, keepdims=True)
        set_to_zero = np.subtract(inputs, get_max_each_row)
        x = np.exp(set_to_zero)
        y = np.sum(x, axis=1, keepdims=True)
        normval = x / y 
        self.a = normval
        # print("Output after forward Softmax: \n", self.a)
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
        # print("Categorical Gradient: \n", l_over_ypredicted.shape)
        # print("Softmax Gradient: \n", ypredicted_over_z.shape)
        delta =  np.dot(l_over_ypredicted, ypredicted_over_z)
        return delta


class DotProductFlipped:
    def calculate(self, delta, xj):
        multiplying_with_delta = np.dot(xj, delta)
        return multiplying_with_delta

class NewWeights: 
    def calculate(self, derivative_l_over_derivative_w, learningrate, layerweights):
        multiply =  learningrate * derivative_l_over_derivative_w
        updateweight = np.subtract(layerweights, multiply)
        return updateweight

class NewBiases: 
    def calculate(self, derivative_l_over_derivative_b, learningrate, layerbiases):
        # The only difference between calculating weights and biases is the sum we calculate in bias. This is done because original biases have a different shape than the calculated bias. 
        multiply = learningrate * derivative_l_over_derivative_b
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
        outputLayer = True
        layerRELU = None

        for instance in reversed(self.instances):
            if isinstance(instance, Categorical_Loss):
                gradient = instance.backward(y_pred, y_true)
                # print("First iteration. Calculating gradient for loss: \n", gradient)
                savingSecondLastProduct = gradient

            # elif isinstance(instance, SoftMax_Activation):
            #     softmaxGradient = instance.backward(y_pred)
            #     gradient = dotHelper.calculate(gradient, softmaxGradient)
                
            #     # Assigning local variable value
            #     savingSecondLastProduct = gradient
            #     # print("Calculating Softmax: \n", gradient)

            elif isinstance(instance, RELU_Activation):
                transposing = transposedHelper.calculate(previousLayerWeights)
                gradient = dotHelper.calculate(savingSecondLastProduct, transposing)
                layerRELU = instance

            elif isinstance(instance, Layer_Creation): 
                
                if (outputLayer):
                    layerGradient = transposedHelper.calculate(instance.inputs)
                    gradient = dotProductFlippedHelper.calculate(gradient, layerGradient)

                    newWeights = newWeightsHelper.calculate(gradient, learningRate, instance.weights)
                    newBiases = newBiasesHelper.calculate(savingSecondLastProduct, learningRate, instance.biases)

                    layerNewWeights, layerNewBiases, layerOldWeights, layerOldBiases = instance.updating_weights_biases(newWeights, newBiases)

                    # Assigning local variable value
                    previousLayerWeights = layerOldWeights
                    # print("layerOldWeights: \n", layerOldWeights)
                    # print("layerOldBiases: \n", layerOldBiases)
                    # print("layerNewWeights: \n", layerNewWeights)
                    # print("layerNewBiases: \n", layerNewBiases)

                    outputLayer = False

                else:
                    elementWise = layerRELU.backward(instance.z)
                    gradient = gradient * elementWise

                    # saving for current layer bias calculation and also next layer's calculation
                    savingSecondLastProduct = gradient

                    transposed = transposedHelper.calculate(instance.inputs)
                    gradient = dotProductFlippedHelper.calculate(gradient, transposed)

                    newWeights = newWeightsHelper.calculate(gradient, learningRate, instance.weights)
                    newBiases = newBiasesHelper.calculate(savingSecondLastProduct, learningRate, instance.biases)

                    layerNewWeights, layerNewBiases, layerOldWeights, layerOldBiases = instance.updating_weights_biases(newWeights, newBiases)

                    previousLayerWeights = layerOldWeights
                    # print("layerOldWeights: \n", layerOldWeights)
                    # print("layerOldBiases: \n", layerOldBiases)
                    # print("layerNewWeights: \n", layerNewWeights)
                    # print("layerNewBiases: \n", layerNewBiases)


def generate_spiral_data(points, classes):
    X = np.zeros((points*classes, 2))  # features
    y = np.zeros((points*classes, classes))  # one-hot labels
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)  
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix, class_number] = 1
    return X, y

def main():

    input, one_hot = generate_spiral_data(100, 3)
    print("Input shape:", input.shape)    # (300, 2)
    print("Labels shape:", one_hot.shape) # (300, 3)

    # Wrapper
    container = Container()
    
    # ✅ Adjust layers for 2D input
    hiddenlayer1 = Layer_Creation(2, 64)
    hiddenlayer_activation = RELU_Activation()

    hiddenlayer2 = Layer_Creation(64, 64)
    hiddenlayer_activation2 = RELU_Activation()

    output_layer = Layer_Creation(64, 3)
    outputlayer_activation = SoftMax_Activation()
    loss = Categorical_Loss()

    learningRate = 0.001

    # Add them to container
    container.add(hiddenlayer1)
    container.add(hiddenlayer_activation)
    container.add(hiddenlayer2)
    container.add(hiddenlayer_activation2)
    container.add(output_layer)
    container.add(outputlayer_activation)
    container.add(loss)


    losses = []
    epochs = 300
    for i in range(epochs):
        forward_output = container.forward(input, one_hot)
        losses.append(forward_output)
        container.backward(container.ypreds, one_hot, learningRate)
        if i % 100 == 0:
            print(f"Epoch {i}, Loss: {forward_output:.4f}")



    # ✅ Plot loss curve
    plt.plot(losses)
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    # ✅ Decision boundary visualization
    def forward_only(x):
        out = x
        for instance in container.instances:
            if isinstance(instance, Categorical_Loss):
                break
            out = instance.forward_pass(out)
        return out

    def plot_decision_boundary(model_forward, X, y):
        h = 0.02
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        grid = np.c_[xx.ravel(), yy.ravel()]
        preds = model_forward(grid)
        Z = np.argmax(preds, axis=1)
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap="brg", alpha=0.3)
        plt.scatter(X[:,0], X[:,1], c=np.argmax(y, axis=1), cmap="brg")
        plt.title("Decision Boundary")
        plt.show()

    plot_decision_boundary(forward_only, input, one_hot)

if __name__ == "__main__":
    main()