import numpy as np

input =  np.array([[0.33878681, 0.32976665, 0.33144654],
                   [0.40427911, 0.27927375, 0.31644715],
                   [0.33414047, 0.34834511, 0.31751442]])
    
one_hot = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]])
print(input)

Y_Predicted_Minus_Y =  np.subtract(input, one_hot)
print("Derivative of L with respect to Y_Predicted: \n", Y_Predicted_Minus_Y)
