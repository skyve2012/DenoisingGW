import tensorflow as tf
import numpy as np
import math
import glob
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
#get_ipython().magic(u'matplotlib inline')
from matplotlib.patches import Ellipse
from tensorflow.examples.tutorials.mnist import input_data
import os, sys, shutil, re
from tensorflow.contrib import rnn
import tensorflow.contrib.slim as slim
import random
from sklearn.preprocessing import scale
import random
import time




##########
# encoder module
##########
def encoder(inputs, cell, hidden_num, is_bidirectional=False, activation_fn=tf.nn.tanh):

    '''
    Inputs:
    inputs: input tensor with shape [None, self.sample_rate] or [None, self.sample_rate, 1]
    cell: the basic LSTM cell in RNN that forms the stcuture of encoder. 
    should be a list of cell models in tensorflow libraries.             
    hidden_num: number of hideen units, should be written in list. Also decides the depth for encoder.
    is_bidirectional: a boolean, indicating if we would use tf.contrib.rnn.stack_bidirectional_dynamic_rnn function instead of the original tf.nn.dynamic_rnn funciton

    Notice that the cell length and hidden_num are two lists, and the function will make then equal to one with greater length

    Output:
    z_codes for the bottle net layer with shape [None, hiddem_num[-1], 1]and the enc_states with is either tuple type or a signle state depending on is_bidireactional
    '''



    # check input shape
    if len(inputs.get_shape().as_list()) == 2:
        inputs = tf.expand_dims(inputs, axis=-1)

    # construct cell lists
    if len(cell) != len(hidden_num): # decide if the cell len is same as hidden len (they are lists)
        if len(cell) < len(hidden_num): # this is to duplicate cells to have same length a hidden_num
            cell_table = []
            for i in range(len(hidden_num)):
                cell_table.append(cell[0])
            assert len(cell_table) == len(hidden_num)
            hidden_num_table = hidden_num
        elif len(cell) > len(hidden_num):
            hidden_num_table = []
            for i in range(len(cell)):
                hidden_num_table.appen(hidden_num[0])
            assert len(hidden_num_table) == len(cell)
            cell_table = cell
        if is_bidirectional is True:
            cells_table1 = []
            cells_table2 = []
            for i in range(max(len(cell), len(hidden_num))):
                cells_table1.append(cell_table[i](hidden_num_table[i], activation=activation_fn))
                cells_table2.append(cell_table[i](hidden_num_table[i], activation=activation_fn))
            input_cells = {'bidirectionalcells1': cells_table1,
                           'bidirectionalcells2': cells_table2} # output a dictionary with two cells used for bidirectional layers

        else:
            cells_table = []
            for i in range(max(len(cell), len(hidden_num))):
                cells_table.append(cell_table[i](hidden_num_table[i], activation=activation_fn))
            input_cells = {'cells': rnn.MultiRNNCell(cells_table)} # output a cell list with different cell and differen hidden_num combined

    else:
        if len(cell) == len(hidden_num) and len(cell) > 1:
            if is_bidirectional is True:
                cells_table1_diff = []
                cells_table2_diff = []
                for i in range(len(cell)):
                    cells_table1_diff.append(cell[i](hidden_num[i], activation=activation_fn))
                    cells_table2_diff.append(cell[i](hidden_num[i], activation=activation_fn))
                input_cells = {'bidirectionalcells1': cells_table1_diff,
                               'bidirectionalcells2': cells_table2_diff}
            else:
                cell_table = cell
                hidden_num_table = hidden_num
                cells_table_diff = []
                for i in range(len(cell)):
                    cells_table_diff.append(cell_table[i](hidden_num_table[i], activation=activation_fn))
                input_cells = {'cells': rnn.MultiRNNCell(cells_table_diff)}

        else:
            # happens when the cell and hidden_num only have one value
            # in each of the list: len(cell)==1, len(hidden_num)==1
            if is_bidirectional is True:
                cells_table1 = []
                cells_table2 = []
                for i in range(max(len(cell), len(hidden_num))):
                    cells_table1.append(cell[i](hidden_num[i], activation=activation_fn))
                    cells_table2.append(cell[i](hidden_num[i], activation=activation_fn))
                input_cells = {'bidirectionalcells1': cells_table1,
                               'bidirectionalcells2': cells_table2} # output a dictionary with two cells used for bidirectional layers
            else:
                cell_table = cell
                hidden_num_table = hidden_num
                input_cells = {'cells': cell_table[0](hidden_num_table[0], activation=activation_fn)}
                
                
    # where the actual calculation happens

    if is_bidirectional is True:
        z_codes, enc_state_bi1, enc_state_bi2 = rnn.stack_bidirectional_dynamic_rnn(input_cells['bidirectionalcells1'],
                                                                                        input_cells['bidirectionalcells2'],
                                                                                        inputs, dtype=tf.float32, scope='enc')

        return z_codes, (enc_state_bi1, enc_state_bi2)
    else:
        #print(input_cells['cells'])
        z_codes, enc_state = tf.nn.dynamic_rnn(input_cells['cells'], inputs, dtype=tf.float32)
        return z_codes, enc_state
    
    
    
##########
# decoder module
##########
def decoder(inputs, cell, hidden_num, is_bidirectional=False, activation_fn=tf.nn.tanh):
    '''
    Inputs:
    inputs: input tensor with shape [None, self.sample_rate, hidden_num[-1]]
    cell: the basic LSTM cell in RNN that forms the stcuture of encoder. 
    should be a list of cell models in tensorflow libraries.             
    hidden_num: number of hideen units, should be written in list. Also decides the depth for encoder.
    is_bidirectional: a boolean, indicating if we would use tf.contrib.rnn.stack_bidirectional_dynamic_rnn function instead of the original tf.nn.dynamic_rnn funciton
        
    Notice that the cell length and hidden_num are two lists, and the function will make then equal to one with greater length
                      
    Output:
    outputs from decoder with shape [None, self.sample_rate, 1] and the enc_states with is either tuple type or a signle state depending on is_bidireactional
        
    '''       
    # check input shape
    if len(inputs.get_shape().as_list()) == 2:
        inputs = tf.expand_dims(inputs, axis=-1)
        
    # construct cell lists
    if len(cell) != len(hidden_num): # decide if the cell len is same as hidden len (they are lists)
        if len(cell) < len(hidden_num): # this is to duplicate cells to have same length a hidden_num
            cell_table = []
            for i in range(len(hidden_num)):
                cell_table.append(cell[0])
            assert len(cell_table) == len(hidden_num)
            hidden_num_table = hidden_num
        elif len(cell) > len(hidden_num):
            hidden_num_table = []
            for i in range(len(cell)):
                hidden_num_table.appen(hidden_num[0])
            assert len(hidden_num_table) == len(cell)
            cell_table = cell
        if is_bidirectional is True:
            cells_table1 = []
            cells_table2 = []
            for i in range(max(len(cell), len(hidden_num))):
                cells_table1.append(cell_table[i](hidden_num_table[i], activation=activation_fn))                
                cells_table2.append(cell_table[i](hidden_num_table[i], activation=activation_fn))
            cells_table1.reverse()
            cells_table2.reverse()
            input_cells = {'bidirectionalcells1': cells_table1,
                           'bidirectionalcells2': cells_table2} # output a dictionary with two cells used for bidirectional layers
                    
        else:
            cells_table = []
            for i in range(max(len(cell), len(hidden_num))):
                cells_table.append(cell_table[i](hidden_num_table[i], activation=activation_fn))
            cells_table.reverse()
            input_cells = {'cells': rnn.MultiRNNCell(cells_table)} # output a cell list with different cell and differen hidden_num combined

    else:
        if len(cell) == len(hidden_num) and len(cell) > 1:
            if is_bidirectional is True:
                cells_table1_diff = []
                cells_table2_diff = []
                for i in range(len(cell)):
                    cells_table1_diff.append(cell[i](hidden_num[i], activation=activation_fn))
                    cells_table2_diff.append(cell[i](hidden_num[i], activation=activation_fn))
                cells_table1_diff.reverse()
                cells_table2_diff.reverse()
                input_cells = {'bidirectionalcells1': cells_table1_diff,
                               'bidirectionalcells2': cells_table2_diff}
            else:
                cell_table = cell
                hidden_num_table = hidden_num
                cells_table_diff = []
                for i in range(len(cell)):
                    cells_table_diff.append(cell_table[i](hidden_num_table[i], activation=activation_fn))
                cells_table_diff.reverse()
                input_cells = {'cells': rnn.MultiRNNCell(cells_table_diff)}
        else:
            # happens when the cell and hidden_num only have one value
            # in each of the list: len(cell)==1, len(hidden_num)==1
            if is_bidirectional is True:
                cells_table1 = []
                cells_table2 = []
                for i in range(max(len(cell), len(hidden_num))):
                    cells_table1.append(cell[i](hidden_num[i], activation=activation_fn))                
                    cells_table2.append(cell[i](hidden_num[i], activation=activation_fn))
                input_cells = {'bidirectionalcells1': cells_table1,
                               'bidirectionalcells2': cells_table2} # output a dictionary with two cells used for bidirectional layers
            else:
                cell_table = cell 
                hidden_num_table = hidden_num  
                input_cells = {'cells': cell_table[0](hidden_num_table[0], activation=activation_fn)}


    if is_bidirectional is True:
        outputs, dec_state_bi1, dec_state_bi2 = rnn.stack_bidirectional_dynamic_rnn(input_cells['bidirectionalcells1'],
                                                                                    input_cells['bidirectionalcells2'],
                                                                inputs, dtype=tf.float32,sequence_length=self.length(inputs))
            #bidirectional_cell1_final = [rnn.LSTMCell(1)]
            #bidirectional_cell2_final = [rnn.LSTMCell(1)]
            #outputs, dec_state_bi1, dec_state_bi2 = rnn.stack_bidirectional_dynamic_rnn(bidirectional_cell1_final,
            #                                                                            bidirectional_cell2_final,
            #                                                                            outputs, dtype=tf.float32, scope='dec')
            #print(tf.Variable(tf.random_normal([tf.stack(tf.shape(outputs)[0]),
            #                                                   outputs.get_shape().as_list()[-1],
            #                                                   1])))
            
        weights_bidirectional_lastlayer = tf.get_variable('final_weights_bi',
                                                          [outputs.get_shape().as_list()[-1],1],
                                                          initializer=tf.random_normal_initializer(0.,.8))
        biases_bidirectional_lastlayer = tf.get_variable('final_biases_bi', [1,],
                                                         initializer=tf.constant_initializer(0.3))

        outputs = tf.reshape(outputs, [-1, outputs.get_shape().as_list()[-1]])
        outputs = tf.add(tf.matmul(outputs, weights_bidirectional_lastlayer), biases_bidirectional_lastlayer)
        outputs = tf.reshape(outputs, [-1, self.sample_rate, 1])
        return outputs, (dec_state_bi1, dec_state_bi2)
    else:
        outputs, dec_state = tf.nn.dynamic_rnn(input_cells['cells'], inputs,
                                               dtype=tf.float32, sequence_length=self.length(inputs))
        weights_rnn_lastlayer = tf.get_variable('final_weights_rnn',
                                                [outputs.get_shape().as_list()[-1],1],
                                                initializer=tf.random_normal_initializer(0.,.8))
        biases_rnn_lastlayer = tf.get_variable('final_biases_rnn', [1,],
                                               initializer=tf.constant_initializer(0.3))
        outputs = tf.reshape(outputs, [-1, outputs.get_shape().as_list()[-1]])
        outputs = tf.add(tf.matmul(outputs, weights_rnn_lastlayer), biases_rnn_lastlayer)
        outputs = tf.reshape(outputs, [-1, self.sample_rate, 1])
        return outputs, dec_state
        
        
##########
# decoder module no last layer
##########
def decoder_nolast(inputs, cell, hidden_num, is_bidirectional=False, activation_fn=tf.nn.tanh):
    '''
    Inputs:
    inputs: input tensor with shape [None, self.sample_rate, hidden_num[-1]]
    cell: the basic LSTM cell in RNN that forms the stcuture of encoder. 
    should be a list of cell models in tensorflow libraries.             
    hidden_num: number of hideen units, should be written in list. Also decides the depth for encoder.
    is_bidirectional: a boolean, indicating if we would use tf.contrib.rnn.stack_bidirectional_dynamic_rnn function instead of the original tf.nn.dynamic_rnn funciton
        
    Notice that the cell length and hidden_num are two lists, and the function will make then equal to one with greater length
                      
    Output:
    outputs from decoder with shape [None, self.sample_rate, 1] and the enc_states with is either tuple type or a signle state depending on is_bidireactional
        
    '''       
    # check input shape
    if len(inputs.get_shape().as_list()) == 2:
        inputs = tf.expand_dims(inputs, axis=-1)
        
    # construct cell lists
    if len(cell) != len(hidden_num): # decide if the cell len is same as hidden len (they are lists)
        if len(cell) < len(hidden_num): # this is to duplicate cells to have same length a hidden_num
            cell_table = []
            for i in range(len(hidden_num)):
                cell_table.append(cell[0])
            assert len(cell_table) == len(hidden_num)
            hidden_num_table = hidden_num
            #print(cell_table)
            #print(hidden_num_table)
        elif len(cell) > len(hidden_num):
            hidden_num_table = []
            for i in range(len(cell)):
                hidden_num_table.appen(hidden_num[0])
            assert len(hidden_num_table) == len(cell)
            cell_table = cell
        if is_bidirectional is True:
            cells_table1 = []
            cells_table2 = []
            for i in range(max(len(cell), len(hidden_num))):
                cells_table1.append(cell_table[i](hidden_num_table[i], activation=activation_fn))                
                cells_table2.append(cell_table[i](hidden_num_table[i], activation=activation_fn))
            cells_table1.reverse()
            cells_table2.reverse()
            input_cells = {'bidirectionalcells1': cells_table1,
                           'bidirectionalcells2': cells_table2} # output a dictionary with two cells used for bidirectional layers
                    
        else:
            cells_table = []
            for i in range(max(len(cell), len(hidden_num))):
                cells_table.append(cell_table[i](hidden_num_table[i], activation=activation_fn))
            cells_table.reverse()
            input_cells = {'cells': rnn.MultiRNNCell(cells_table)} # output a cell list with different cell and differen hidden_num combined

    else:
        if len(cell) == len(hidden_num) and len(cell) > 1:
            if is_bidirectional is True:
                cells_table1_diff = []
                cells_table2_diff = []
                for i in range(len(cell)):
                    cells_table1_diff.append(cell[i](hidden_num[i], activation=activation_fn))
                    cells_table2_diff.append(cell[i](hidden_num[i], activation=activation_fn))
                cells_table1_diff.reverse()
                cells_table2_diff.reverse()
                input_cells = {'bidirectionalcells1': cells_table1_diff,
                               'bidirectionalcells2': cells_table2_diff}
            else:
                cell_table = cell
                hidden_num_table = hidden_num
                cells_table_diff = []
                for i in range(len(cell)):
                    cells_table_diff.append(cell_table[i](hidden_num_table[i], activation=activation_fn))
                cells_table_diff.reverse()
                input_cells = {'cells': rnn.MultiRNNCell(cells_table_diff)}
        else:
            # happens when the cell and hidden_num only have one value
            # in each of the list: len(cell)==1, len(hidden_num)==1
            if is_bidirectional is True:
                cells_table1 = []
                cells_table2 = []
                for i in range(max(len(cell), len(hidden_num))):
                    cells_table1.append(cell[i](hidden_num[i], activation=activation_fn))                
                    cells_table2.append(cell[i](hidden_num[i], activation=activation_fn))
                input_cells = {'bidirectionalcells1': cells_table1,
                               'bidirectionalcells2': cells_table2} # output a dictionary with two cells used for bidirectional layers
            else:
                cell_table = cell 
                hidden_num_table = hidden_num  
                input_cells = {'cells': cell_table[0](hidden_num_table[0], activation=activation_fn)}


    if is_bidirectional is True:
        outputs, dec_state_bi1, dec_state_bi2 = rnn.stack_bidirectional_dynamic_rnn(input_cells['bidirectionalcells1'],
                                                                                    input_cells['bidirectionalcells2'],
                                                                                    inputs, dtype=tf.float32)

            
        return outputs, (dec_state_bi1, dec_state_bi2)
        
    else:
        outputs, dec_state = tf.nn.dynamic_rnn(input_cells['cells'], inputs, dtype=tf.float32)


            
        return outputs, dec_state
    

    
##########
# stair decoder module
##########
def stair_decoder(inputs, cell, hidden_num, sample_length, states=None, is_bidirectional=False, activation_fn=tf.nn.tanh):
    '''
    Inputs:
    inputs: input tensor with shape [None, self.sample_rate, hidden_num[-1]]
    cell: the basic LSTM cell in RNN that forms the stcuture of encoder. 
    should be a list of cell models in tensorflow libraries.             
    hidden_num: number of hideen units, should be written in list. Also decides the depth for encoder.
    is_bidirectional: a boolean, indicating if we would use tf.contrib.rnn.stack_bidirectional_dynamic_rnn function instead of the original tf.nn.dynamic_rnn funciton
        
    Notice that the cell length and hidden_num are two lists, and the function will make then equal to one with greater length
                      
    Output:
    outputs from decoder with shape [None, self.sample_rate, 1] and the enc_states with is either tuple type or a signle state depending on is_bidireactional
        
    '''  
    assert states is not None
    # check input shape
    if len(inputs.get_shape().as_list()) == 2:
        inputs = tf.expand_dims(inputs, axis=-1)
        
    # construct cell lists
    if len(cell) != len(hidden_num): # decide if the cell len is same as hidden len (they are lists)
        if len(cell) < len(hidden_num): # this is to duplicate cells to have same length a hidden_num
            cell_table = []
            for i in range(len(hidden_num)):
                cell_table.append(cell[0])
            assert len(cell_table) == len(hidden_num)
            hidden_num_table = hidden_num
        elif len(cell) > len(hidden_num):
            hidden_num_table = []
            for i in range(len(cell)):
                hidden_num_table.appen(hidden_num[0])
            assert len(hidden_num_table) == len(cell)
            cell_table = cell
        if is_bidirectional is True:
            cells_table1 = []
            cells_table2 = []
            for i in range(max(len(cell), len(hidden_num))):
                cells_table1.append(cell_table[i](hidden_num_table[i], activation=activation_fn))                
                cells_table2.append(cell_table[i](hidden_num_table[i], activation=activation_fn))
            cells_table1.reverse()
            cells_table2.reverse()
            input_cells = {'bidirectionalcells1': cells_table1,
                           'bidirectionalcells2': cells_table2} # output a dictionary with two cells used for bidirectional layers
                    
        else:
            cells_table = []
            for i in range(max(len(cell), len(hidden_num))):
                cells_table.append(cell_table[i](hidden_num_table[i], activation=activation_fn))
            cells_table.reverse()
            input_cells = {'cells': rnn.MultiRNNCell(cells_table)} # output a cell list with different cell and differen hidden_num combined

    else:
        if len(cell) == len(hidden_num) and len(cell) > 1:
            if is_bidirectional is True:
                cells_table1_diff = []
                cells_table2_diff = []
                for i in range(len(cell)):
                    cells_table1_diff.append(cell[i](hidden_num[i], activation=activation_fn))
                    cells_table2_diff.append(cell[i](hidden_num[i], activation=activation_fn))
                cells_table1_diff.reverse()
                cells_table2_diff.reverse()
                input_cells = {'bidirectionalcells1': cells_table1_diff,
                               'bidirectionalcells2': cells_table2_diff}
            else:
                cell_table = cell
                hidden_num_table = hidden_num
                cells_table_diff = []
                for i in range(len(cell)):
                    cells_table_diff.append(cell_table[i](hidden_num_table[i], activation=activation_fn))
                cells_table_diff.reverse()
                input_cells = {'cells': rnn.MultiRNNCell(cells_table_diff)}
        else:
            # happens when the cell and hidden_num only have one value
            # in each of the list: len(cell)==1, len(hidden_num)==1
            if is_bidirectional is True:
                cells_table1 = []
                cells_table2 = []
                for i in range(max(len(cell), len(hidden_num))):
                    cells_table1.append(cell[i](hidden_num[i], activation=activation_fn))                
                    cells_table2.append(cell[i](hidden_num[i], activation=activation_fn))
                input_cells = {'bidirectionalcells1': cells_table1,
                               'bidirectionalcells2': cells_table2} # output a dictionary with two cells used for bidirectional layers
            else:
                cell_table = cell 
                hidden_num_table = hidden_num  
                input_cells = {'cells': cell_table[0](hidden_num_table[0], activation=activation_fn)}


    if is_bidirectional is True:
        outputs, dec_state_bi1, dec_state_bi2 = rnn.stack_bidirectional_dynamic_rnn(input_cells['bidirectionalcells1'],
                                                                                    input_cells['bidirectionalcells2'],
                                                                                    inputs, states[0], states[1])

        weights_bidirectional_lastlayer = tf.get_variable('final_weights_bi',
                                                          [outputs.get_shape().as_list()[-1],1],
                                                          initializer=tf.random_normal_initializer(0.,.8))
        biases_bidirectional_lastlayer = tf.get_variable('final_biases_bi', [1,],
                                                         initializer=tf.constant_initializer(0.3))

        outputs = tf.reshape(outputs, [-1, outputs.get_shape().as_list()[-1]])
        outputs = tf.add(tf.matmul(outputs, weights_bidirectional_lastlayer), biases_bidirectional_lastlayer)
        outputs = tf.reshape(outputs, [-1, sample_length, 1])
        return outputs, (dec_state_bi1, dec_state_bi2)
    else:
        outputs, dec_state = tf.nn.dynamic_rnn(input_cells['cells'], inputs, dtype=tf.float32, initial_state=states[0])
        weights_rnn_lastlayer = tf.get_variable('final_weights_rnn',
                                                [outputs.get_shape().as_list()[-1],1],
                                                initializer=tf.random_normal_initializer(0.,.8))
        biases_rnn_lastlayer = tf.get_variable('final_biases_rnn', [1,],
                                               initializer=tf.constant_initializer(0.3))
        outputs = tf.reshape(outputs, [-1, outputs.get_shape().as_list()[-1]])
        outputs = tf.add(tf.matmul(outputs, weights_rnn_lastlayer), biases_rnn_lastlayer)
        outputs = tf.reshape(outputs, [-1, sample_length, 1])
        return outputs, dec_state
        
    
    
        
##########
# decoder module no last layer (stair)
##########
def stair_decoder_nolast(inputs, cell, hidden_num, is_bidirectional=False, activation_fn=tf.nn.tanh, states=None):
    '''
    Inputs:
    inputs: input tensor with shape [None, self.sample_rate, hidden_num[-1]]
    cell: the basic LSTM cell in RNN that forms the stcuture of encoder. 
    should be a list of cell models in tensorflow libraries.             
    hidden_num: number of hideen units, should be written in list. Also decides the depth for encoder.
    is_bidirectional: a boolean, indicating if we would use tf.contrib.rnn.stack_bidirectional_dynamic_rnn function instead of the original tf.nn.dynamic_rnn funciton
        
    Notice that the cell length and hidden_num are two lists, and the function will make then equal to one with greater length
                      
    Output:
    outputs from decoder with shape [None, self.sample_rate, 1] and the enc_states with is either tuple type or a signle state depending on is_bidireactional
        
    '''    
    assert states is not None
    # check input shape
    if len(inputs.get_shape().as_list()) == 2:
        inputs = tf.expand_dims(inputs, axis=-1)
        
    # construct cell lists
    if len(cell) != len(hidden_num): # decide if the cell len is same as hidden len (they are lists)
        if len(cell) < len(hidden_num): # this is to duplicate cells to have same length a hidden_num
            cell_table = []
            for i in range(len(hidden_num)):
                cell_table.append(cell[0])
            assert len(cell_table) == len(hidden_num)
            hidden_num_table = hidden_num
                #print(cell_table)
                #print(hidden_num_table)
        elif len(cell) > len(hidden_num):
            hidden_num_table = []
            for i in range(len(cell)):
                hidden_num_table.appen(hidden_num[0])
            assert len(hidden_num_table) == len(cell)
            cell_table = cell
        if is_bidirectional is True:
            cells_table1 = []
            cells_table2 = []
            for i in range(max(len(cell), len(hidden_num))):
                cells_table1.append(cell_table[i](hidden_num_table[i], activation=activation_fn))                
                cells_table2.append(cell_table[i](hidden_num_table[i], activation=activation_fn))
            cells_table1.reverse()
            cells_table2.reverse()
            input_cells = {'bidirectionalcells1': cells_table1,
                           'bidirectionalcells2': cells_table2} # output a dictionary with two cells used for bidirectional layers
                    
        else:
            cells_table = []
            for i in range(max(len(cell), len(hidden_num))):
                cells_table.append(cell_table[i](hidden_num_table[i], activation=activation_fn))
            cells_table.reverse()
            input_cells = {'cells': rnn.MultiRNNCell(cells_table)} # output a cell list with different cell and differen hidden_num combined

    else:
        if len(cell) == len(hidden_num) and len(cell) > 1:
            if is_bidirectional is True:
                cells_table1_diff = []
                cells_table2_diff = []
                for i in range(len(cell)):
                    cells_table1_diff.append(cell[i](hidden_num[i], activation=activation_fn))
                    cells_table2_diff.append(cell[i](hidden_num[i], activation=activation_fn))
                cells_table1_diff.reverse()
                cells_table2_diff.reverse()
                input_cells = {'bidirectionalcells1': cells_table1_diff,
                               'bidirectionalcells2': cells_table2_diff}
            else:
                cell_table = cell
                hidden_num_table = hidden_num
                cells_table_diff = []
                for i in range(len(cell)):
                    cells_table_diff.append(cell_table[i](hidden_num_table[i], activation=activation_fn))
                cells_table_diff.reverse()
                input_cells = {'cells': rnn.MultiRNNCell(cells_table_diff)}
        else:
            # happens when the cell and hidden_num only have one value
            # in each of the list: len(cell)==1, len(hidden_num)==1
            if is_bidirectional is True:
                cells_table1 = []
                cells_table2 = []
                for i in range(max(len(cell), len(hidden_num))):
                    cells_table1.append(cell[i](hidden_num[i], activation=activation_fn))                
                    cells_table2.append(cell[i](hidden_num[i], activation=activation_fn))
                input_cells = {'bidirectionalcells1': cells_table1,
                               'bidirectionalcells2': cells_table2} # output a dictionary with two cells used for bidirectional layers
            else:
                cell_table = cell 
                hidden_num_table = hidden_num  
                input_cells = {'cells': cell_table[0](hidden_num_table[0], activation=activation_fn)}


    if is_bidirectional is True:
        outputs, dec_state_bi1, dec_state_bi2 = rnn.stack_bidirectional_dynamic_rnn(input_cells['bidirectionalcells1'],
                                                                                    input_cells['bidirectionalcells2'],
                                                                                    inputs, states[0], states[1])

        return outputs, (dec_state_bi1, dec_state_bi2)
        
    else:
        outputs, dec_state = tf.nn.dynamic_rnn(input_cells['cells'], inputs, dtype=tf.float32)


            
        return outputs, dec_state
        
  
        
    
  

            
            
            