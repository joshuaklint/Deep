import numpy as np


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def relu(x):
    s = np.maximum(0, x)
    return s


def initialize_weights(shape):
    L = len(shape)
    parameter = {}
    for i in range(1, L):
        parameter['W' + str(i)] = np.random.randn(shape[i], shape[i - 1]) * np.sqrt(1 / shape[i - 1])
        parameter['b' + str(i)] = np.zeros((shape[i], 1))
    return parameter


'''def forward_propagation(X, parameter):
    W1 = parameter['W1']
    b1 = parameter['b1']
    W2 = parameter['W2']
    b2 = parameter['b2']
    W3 = parameter['W3']
    b3 = parameter['b3']

    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)
    forward = {'Z1': Z1, 'A1': A1, 'Z2': Z2,
               'A2': A2, 'Z3': Z3, 'A3': A3}
    return forward'''


def forward_propagation_2(X, parameter, keep_prob):
    W1 = parameter['W1']
    b1 = parameter['b1']
    W2 = parameter['W2']
    b2 = parameter['b2']
    W3 = parameter['W3']
    b3 = parameter['b3']

    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    D1 = np.random.rand(A1.shape[0], A1.shape[1]) < keep_prob
    A1 = A1 * D1.astype(int)
    A1 = A1 / keep_prob

    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    D2 = np.random.rand(A2.shape[0], A2.shape[1]) < keep_prob
    A2 = A2 * D2.astype(int)
    A2 = A2 / keep_prob

    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    forward = {'Z1': Z1, 'A1': A1, 'D1': D1,
               'A2': A2, 'Z2': Z2, 'D2': D2,
               'Z3': Z3, 'A3': A3}
    return forward


def cost_function(Y, forward):
    m = Y.shape[1]
    A3 = forward['A3']
    cost = -1 / m * (np.dot(Y, np.log(A3).T) + np.dot(1 - Y, np.log(1 - A3).T))
    cost = np.squeeze(cost)
    return cost


'''def cost_function_2(Y, forward, parameter, lambed):
    W1 = parameter['W1']
    W2 = parameter['W2']
    W3 = parameter['W3']
    m = Y.shape[1]
    cost1 = cost_function(Y, forward)
    cost2 = lambed / (2 * m) * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3)))
    cost = cost1 + cost2
    return cost'''

'''def backward_propagation(X, Y, forward, parameter):
    m = Y.shape[1]
    W3 = parameter['W3']
    W2 = parameter['W2']
    A2 = forward['A2']
    A3 = forward['A3']
    A1 = forward['A1']

    dZ3 = A3 - Y
    dW3 = 1 / m * np.dot(dZ3, A2.T)
    db3 = 1 / m * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.dot(W3.T, dZ3)
    dZ2 = dA2 * np.int64(A2 > 0)
    dW2 = 1 / m * np.dot(dZ2, A2.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = dA1 * np.int64(A1 > 0)
    dW1 = 1 / m * np.dot(dZ1, X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

    backward = {'dW3': dW3, 'db3': db3, 'dW2': dW2,
                'db2': db2, 'dW1': dW1, 'db1': db1}
    return backward'''

'''def backward_propagation_2(X, Y, forward, parameter, lambed):
    W1 = parameter['W1']
    W2 = parameter['W2']
    W3 = parameter['W3']
    m = Y.shape[1]

    A3 = forward['A3']
    A2 = forward['A2']
    A1 = forward['A1']

    dZ3 = A3 - Y
    dW3 = 1 / m * np.dot(dZ3, A2.T) + (lambed / m) * W3
    db3 = 1 / m * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.dot(W3.T, dZ3)
    dZ2 = dA2 * np.int64(A2 > 0)
    dW2 = 1 / m * np.dot(dZ2, A2.T) + (lambed / m) * W2
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = dA1 * np.int64(A1 > 0)
    dW1 = 1 / m * np.dot(dZ1, A1.T) + (lambed / m) * W1
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

    backward = {'dW3': dW3, 'db3': db3, 'dW2': dW2,
                'db2': db2, 'dW1': dW1, 'db1': db1}
    return backward'''


def backward_propagation_3(X, Y, forward, parameter, keep_prob):
    m = Y.shape[1]
    W3 = parameter['W3']
    W2 = parameter['W2']
    A3 = forward['A3']
    A2 = forward['A2']
    A1 = forward['A1']
    D1 = forward['D1']
    D2 = forward['D2']

    dZ3 = A3 - Y
    dW3 = 1 / m * np.dot(dZ3, A2.T)
    db3 = 1 / m * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.dot(W3.T, dZ3)
    dA2 = dA2 * D2
    dA2 = dA2 / keep_prob
    dZ2 = dA2 * np.int64(A2 > 0)
    dW2 = 1 / m * np.dot(dZ2, A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dA1 = dA1 * D1
    dA1 = dA1 / keep_prob
    dZ1 = dA1 * np.int64(A1 > 0)
    dW1 = 1 / m * np.dot(dZ1, X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
    backward = {'dW3': dW3, 'db3': db3, 'dW2': dW2,
                'db2': db2, 'dW1': dW1, 'db1': db1}
    return backward


def update_parameters(parameter, backward, learning_rate=0.01):
    W1 = parameter['W1']
    W2 = parameter['W2']
    W3 = parameter['W3']
    b1 = parameter['b1']
    b2 = parameter['b2']
    b3 = parameter['b3']

    dW1 = backward['dW1']
    db1 = backward['db1']
    dW2 = backward['dW2']
    db2 = backward['db2']
    dW3 = backward['dW3']
    db3 = backward['db3']

    W1 = W1 - learning_rate * dW1
    W2 = W2 - learning_rate * dW2
    W3 = W3 - learning_rate * dW3
    b1 = b1 - learning_rate * db1
    b2 = b2 - learning_rate * db2
    b3 = b3 - learning_rate * db3

    parameter = {'W1': W1, 'W2': W2, 'W3': W3,
                 'b1': b1, 'b2': b2, 'b3': b3}
    return parameter


def Models(X, Y, keep_prob, iters, print_cost=False):
    shape = [X.shape[0], 10, 3, 1]
    costs = []
    iterations = []

    parameter = initialize_weights(shape)

    for i in range(iters):
        forward = forward_propagation_2(X, parameter, keep_prob)
        cost = cost_function(Y, forward)
        backward = backward_propagation_3(X, Y, forward, parameter, keep_prob)
        parameter = update_parameters(parameter, backward, learning_rate=0.001)
        if print_cost and i % 1000 == 0:
            a = f'Epochs: {i}: >>>>>>>>>>>>>>>>>>>>>>>>>>>>> Cost: {cost}'
            print(a)
            costs.append(cost)
            # total_cost = np.squeeze(total_cost)
            iterations.append(i)
    all_p = {'costs': costs, 'iterations': iterations, 'backward': backward, 'forward': forward}
    return all_p
