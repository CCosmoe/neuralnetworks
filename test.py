import numpy as np

derivative_l_over_derivative_b =  np.array([[ -0.3382036,  0.02460542, -0.00745131],
                                            [ 0.18038728,  0.01465984,  0.17802348],
                                            [ 0.00177756, -0.02409236, -0.02121836]])

learningrate = 0.01

biases = np.array([1, 2, 3])


multiply =  np.dot(learningrate, derivative_l_over_derivative_b)
print("Multiple by 0.01: \n", multiply)
summing = np.sum(multiply, axis=0, keepdims=True)
print("Adding: \n", summing)

newBiases = np.subtract(biases, summing)
print("NewBiases: \n", newBiases)

# upadtebiases = np.subtract(biases, multiply)



# -0.0121393571510386 0.0036076339451142 -0.055950551995455
# -0.0037228112774012  0.0021660471255996  0.1195300502666988
# 0.0299381410794459  -0.0106143001419634  -0.0835073917938898
