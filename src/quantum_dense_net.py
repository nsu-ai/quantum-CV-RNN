import tensorflow as tf
import pennylane as qml
import numpy as np
from .quantum_basic_blocks import *


def quantum_dense_net_circuit(input, params, modes, output_size, layers, last_activation, return_state):
    linear_params, activiation_params, last_activation_params = params
    q = np.arange(modes)
    encode_input(input, q)
    for i in range(layers - 1):
        linear_layer(linear_params[i], q)
        activation_layer(activiation_params[i], q)
    linear_layer(linear_params[layers - 1], q)
    if last_activation:
        activation_layer(last_activation_params, q[: output_size])
    if return_state:
        # this cannot be optimized!
        return qml.state()
    else:
        return get_x_quad_expectations(q[: output_size])


class QuantumDenseNet(tf.keras.layers.Layer):
    def __init__(self, modes, output_size, layers, dev, last_activation = True):
        super(QuantumDenseNet, self).__init__()
        self.modes = modes
        self.output_size = output_size
        self.layers = layers
        self.dev = dev
        self.last_activation = last_activation
        
        self.qnode = qml.QNode(quantum_dense_net_circuit, self.dev, interface='tf')
        

    def build(self, input_shape):        
        self.linear_params = tf.random.normal(shape=[self.layers, get_linear_layer_params_n(self.modes)], stddev=0.001)
        self.linear_params = tf.Variable(self.linear_params)
        self.activiation_params = tf.random.normal(shape=[self.layers - 1, get_activation_layer_params_n(self.modes)], stddev=0.001)
        self.activiation_params = tf.Variable(self.activiation_params)
        self.last_activation_params = None
        if self.last_activation:
            self.last_activation_params = tf.random.normal(shape=[get_activation_layer_params_n(self.modes)], stddev=0.001)
            self.last_activation_params = tf.Variable(self.last_activation_params)
    

    def call(self, batch, return_state = False):
        # TODO do it somehow in parallel 
        outs = list()
        for x in batch:
            out = self.qnode(x, [self.linear_params, self.activiation_params, self.last_activation_params], self.modes, self.output_size,
                             self.layers, self.last_activation, return_state)
            outs.append(out)
   
        if return_state:
            for i in range(len(outs)):
                outs[i] = trace(outs[i])
        else:
            for i in range(len(outs)):
                outs[i] = tf.concat(outs[i], axis = 0)
        return tf.stack(outs)
