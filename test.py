import numpy as np

derivative_l_over_derivative_b =  np.array([[ -0.3382036,  0.02460542, -0.00745131],
                                            [ 0.18038728,  0.01465984,  0.17802348],
                                            [ 0.00177756, -0.02409236, -0.02121836]])



array  = np.where(derivative_l_over_derivative_b > 0, 1.0, 0.0)

print("Neuron status: \n", array)

# upadtebiases = np.subtract(biases, multiply)



# -0.012139357151330386 0.003607622339451142 -0.0559570551995455
# -0.00372281127711012  0.002164460471255996  0.119566300502666988
# 0.029938144410794459  -0.010614443001419634  -0.08350783917938898
