import tensorflow as tf
import pennylane as qml
import numpy as np
from .quantum_basic_blocks import *
from .quantum_rnn_blocks import *


def quantum_rnn_circuit(input, params, hidden_modes, output_size, return_state):
    rnn_params, linear_params, activiation_params = params
    
    quantum_RNN(input, rnn_params, np.arange(hidden_modes), np.arange(input.shape[1]) + hidden_modes)
    linear_layer(linear_params, np.arange(hidden_modes))
    activation_layer(activiation_params, np.arange(hidden_modes))
    
    if return_state:
        # this cannot be optimized!
        return qml.state()
    else:
        return get_x_quad_expectations(np.arange(output_size))


class QuantumRNN(tf.keras.layers.Layer):
    def __init__(self, hidden_modes, output_size, dev):
        super(QuantumRNN, self).__init__()
        self.hidden_modes = hidden_modes
        self.output_size = output_size
        self.dev = dev
        
        self.qnode = qml.QNode(quantum_rnn_circuit, self.dev, interface='tf')
        

    def build(self, input_shape):
        x_step_dim = input_shape[2]
        self.rnn_params = tf.random.normal(shape=[get_quantum_RNN_params_n(x_step_dim, self.hidden_modes)], stddev=0.001)
        self.rnn_params = tf.Variable(self.rnn_params)
        self.linear_params = tf.random.normal(shape=[get_linear_layer_params_n(self.hidden_modes)], stddev=0.001)
        self.linear_params = tf.Variable(self.linear_params)
        self.activiation_params = tf.random.normal(shape=[get_activation_layer_params_n(self.hidden_modes)], stddev=0.001)
        self.activiation_params = tf.Variable(self.activiation_params)
    

    def call(self, batch, return_state = False):
        # TODO do it somehow in parallel 
        outs = list()
        for x in batch:
            out = self.qnode(x, [self.rnn_params, self.linear_params, self.activiation_params], self.hidden_modes, self.output_size, return_state)
            outs.append(out)
   
        if return_state:
            for i in range(len(outs)):
                outs[i] = trace(outs[i])
        else:
            for i in range(len(outs)):
                outs[i] = tf.concat(outs[i], axis = 0)
        return tf.stack(outs)
