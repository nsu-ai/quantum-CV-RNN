import tensorflow as tf
import pennylane as qml
from numpy import pi


# Encoding 
def encode_input(input_vec, q):
    if len(q) < input_vec.shape[0]:
        raise Exception('Not enough wires to encode the input vector')
    for i, x in enumerate(input_vec):
        qml.Displacement(x, 0, wires = q[i])


def encode_input_inv(input_vec, q):
    if len(q) < input_vec.shape[0]:
        raise Exception('Not enough wires to encode the input vector')
    for i, x in enumerate(input_vec):
         qml.Displacement(x, pi, wires = q[i])


# Interferometer
def get_interferometer_params_n(wires_n):
    N = wires_n
    return N * (N - 1) + max(1, N - 1)


def split_interferometer_params(params, wires_n):
    N = wires_n
    theta = params[:N*(N-1)//2]
    phi = params[N*(N-1)//2:N*(N-1)]
    rphi = params[-N+1:]
    return theta, phi, rphi


def interferometer(params, q):
    N = len(q)
    theta, phi, rphi = split_interferometer_params(params, N)

    if N == 1:
        # the interferometer is a single rotation
        qml.Rotation(rphi[0], wires = q[0])
        return

    n = 0  # keep track of free parameters

    # Apply the rectangular beamsplitter array
    # The array depth is N
    for l in range(N):
        for k, (q1, q2) in enumerate(zip(q[:-1], q[1:])):
            # skip even or odd pairs depending on layer
            if (l + k) % 2 != 1:
                qml.Beamsplitter(theta[n], phi[n], wires = [q1, q2])
                n += 1

    # apply the final local phase shifts to all modes except the last one
    for i in range(max(1, N - 1)):
        qml.Rotation(rphi[i], wires = q[i])
        

def interferometer_inv(params, q):
    N = len(q)
    theta, phi, rphi = split_interferometer_params(params, N)

    if N == 1:
        # the interferometer is a single rotation
        qml.Rotation(-rphi[0], wires = q[0])
        return
    
    # apply the final local phase shifts to all modes except the last one
    for i in range(max(1, N - 1)):
        qml.Rotation(-rphi[i], wires = q[i])

    pairs = list()

    # Apply the rectangular beamsplitter array
    # The array depth is N
    for l in range(N):
        for k, (q1, q2) in enumerate(zip(q[:-1], q[1:])):
            # skip even or odd pairs depending on layer
            if (l + k) % 2 != 1:
                pairs.append((q1, q2))
    
    for i in range(len(theta) - 1, -1, -1):
        qml.Beamsplitter(-theta[i], phi[i], wires = pairs[i])


# Linear layer
def get_linear_layer_params_n(wires_n):
    N = wires_n
    M = N * (N - 1) + max(1, N - 1)
    return 2*M+3*N


def split_linear_layer(params, wires_n):
    N = wires_n
    M = int(N * (N - 1)) + max(1, N - 1)
    int1 = params[:M]
    s = params[M:M+N]
    int2 = params[M+N:2*M+N]
    dr = params[2*M+N:2*M+2*N]
    dp = params[2*M+2*N:2*M+3*N]
    return int1, s, int2, dr, dp


def linear_layer(params, q):
    N = len(q)
    int1, s, int2, dr, dp = split_linear_layer(params, N)

    # begin layer
    interferometer(int1, q)

    for i in range(N):
        qml.Squeezing(s[i], 0, wires = q[i])

    interferometer(int2, q)

    for i in range(N):
        qml.Displacement(dr[i], dp[i], wires = q[i])

        
def linear_layer_inv(params, q):
    N = len(q)
    int1, s, int2, dr, dp = split_linear_layer(params, N)
    
    for i in range(N):
        qml.Displacement(dr[i], dp[i] + pi, wires = q[i])

    interferometer_inv(int2, q)

    for i in range(N):
        qml.Squeezing(s[i], pi, wires = q[i])

    interferometer_inv(int1, q)


# Activation layer
def get_activation_layer_params_n(wires_n):
    return wires_n


def activation_layer(params, q):
    N = len(q)
    for i in range(N):
        qml.Kerr(params[i], wires=q[i])


# Measurements
def get_x_quad_expectations(modes_it):
    return  [qml.expval(qml.X(position)) for position in modes_it]


def trace(state):
    return tf.math.reduce_sum(tf.math.abs(state) ** 2)
