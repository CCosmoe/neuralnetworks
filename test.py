import numpy as np

l_over_ypredicted =  np.array([[-0.66291657,  0.33512171,  0.32779485],
                               [ 0.55595716, -0.69645777,  0.14050061],
                               [ 0.33896514,  0.33564222, -0.67460736]])
    
ypredicted_over_z =  np.array([[0.22345819, 0.22281515, 0.22034539],
                               [0.2468688,  0.21140434, 0.12076019],
                               [0.22406777, 0.22298652, 0.21951227]])

xj =  np.array([[0.09183786, 2.78843292,  0.19121411],
                [0.08857933, 4.60067402,  0.12507937],
                [0.,         0.,          0.28771973],
                [0.,         0.,          0.        ]])

delta =  np.dot(l_over_ypredicted, ypredicted_over_z)
print("Delta: \n", delta)

multiplying_with_delta = np.dot(xj, delta)

print("3Products: \n", multiplying_with_delta)
