import pennylane as qml
import numpy as np
from .quantum_basic_blocks import *


def get_quantum_RNN_params_n(working_wires_n, hidden_wires_n):
    params_n = 0
    params_n += get_linear_layer_params_n(working_wires_n)
    params_n += get_linear_layer_params_n(hidden_wires_n)
    # params_n += hidden_wires_n # ControlledPhase
    params_n += hidden_wires_n # activation_like_layer
    return params_n


def split_quantum_RNN_params(params, working_wires_n, hidden_wires_n):
    ending = get_linear_layer_params_n(working_wires_n)
    working_linear_params = params[: ending]
    
    beginning = ending
    ending += get_linear_layer_params_n(hidden_wires_n)
    hidden_linear_params = params[beginning: ending]
    
    # beginning = ending
    # ending += hidden_wires_n
    # control_params = params[beginning: ending]
    # control_params = None
    control_params = np.ones(hidden_wires_n)
    
    beginning = ending
    ending += hidden_wires_n
    activation_params = params[beginning: ending]
    
    return working_linear_params, hidden_linear_params, control_params, activation_params


def quantum_RNN_cell(x_current, params, hidden_wires, working_wires):
    working_linear_params, hidden_linear_params, control_params, activation_params = split_quantum_RNN_params(params, len(working_wires), len(hidden_wires))
        
    # prepare new input state 
    encode_input(x_current, working_wires)
    linear_layer(working_linear_params, working_wires)
    linear_layer(hidden_linear_params, hidden_wires)
    
    # this is how the input and hidden states are connected
    for i in range(min(len(hidden_wires), x_current.shape[0])):
        qml.ControlledPhase(control_params[i], wires = [working_wires[i], hidden_wires[i]])
    
    linear_layer_inv(working_linear_params, working_wires)
    encode_input_inv(x_current, working_wires)
    
    activation_layer(activation_params, hidden_wires)


def quantum_RNN(x, params, hidden_wires, working_wires):
    working_wires = working_wires[: x.shape[1]]
    
    for x_current in x:
        quantum_RNN_cell(x_current, params, hidden_wires, working_wires)
