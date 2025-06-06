import numpy as np

output_layer_weights =  np.array([[ 0.09089969, -0.03761097,  0.06690646],
                                  [ 0.05787504,  0.36574873,  0.15447709],
                                  [ 0.13360337,  0.10051279,  0.00654227],
                                  [ 0.10835819,  0.08419912, -0.02765926]])

learningrate = 0.01

derivative_l_over_derivative_w =  np.array([[ 0.,         0.,           0.,        ],
                                            [ 0.00255721, -0.0150004,   0.01734683,],
                                            [ 0.,          0.,          0.,        ],
                                            [-0.00076774,  0.00393547, -0.00425434]])


multiply =  np.dot(learningrate, derivative_l_over_derivative_w)
print('product: \n', multiply)

updateweight = np.subtract(output_layer_weights, multiply)
print('Updated weight: \n', updateweight)

# -0.0121393571510386 0.0036076339451142 -0.055950551995455
# -0.0037228112774012  0.0021660471255996  0.1195300502666988
# 0.0299381410794459  -0.0106143001419634  -0.0835073917938898
