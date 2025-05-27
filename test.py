import numpy as np

input =  np.array([[0.33878681, 0.32976665, 0.33144654],
                   [0.40427911, 0.27927375, 0.31644715],
                   [0.33414047, 0.34834511, 0.31751442]])
    
print(input)

transposing =  np.transpose(input)
print("Derivative of L with respect to Y_Predicted: \n", transposing)
