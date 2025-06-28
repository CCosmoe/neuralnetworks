import numpy as np

x1 =  np.array([[ -0.3382036  , 0.02460542,   -0.00745131],
                  [0.18038728, 0.01465984 ,  0.17802348],
                  [0.00177756, -0.02409236 , -0.02121836]])

x2 = np.array([[0.1, -0.5, -0.44],
               [-0.14, 0.12, 0.73],
               [0.5, -0.33, -0.13]])

x3 = np.array([[1,1,1],
               [1,0,0],
               [1,1,1]])


x4 = np.array([[1,2,-1.5],
               [2,5,2.7],
               [3,-1,3.3],
               [2.5,2.0,-0.8]])


x5 = np.array([[0.2, 0.5, -0.26],
               [0.8, -0.91, -0.27],
               [-0.5, 0.26, 0.17],
               [1.0, -0.5, 0.87]])

biases = [1, 2, 3]
mul = np.dot(x1, x2)

mul2 = np.dot(mul, x3)

mul3 = np.dot(x4, mul2)
mul4 = np.dot(0.01, mul3)

subbing = np.subtract(x5, mul4)
# reluvals  = np.maximum(0, addedbias)

mul5 = np.dot(0.01, mul2)


print("Neuron status: \n", mul)
print("Neuron status: \n", mul2)
print("Neuron status: \n", mul3)
print("Neuron status: \n", mul4)
print("Neuron status: \n", subbing)
print("Neuron status: \n", mul5)
added = np.sum(mul5, axis=0, keepdims=True)
# upadtebiases = np.subtract(biases, multiply)
print("added \n", added)

subbing2 = np.subtract(biases, added)
print("subbing \n", subbing2)


# -0.012139357151330386 0.003607622339451142 -0.0559570551995455
# -0.00372281127711012  0.002164460471255996  0.119566300502666988
# 0.029938144410794459  -0.010614443001419634  -0.08350783917938898
