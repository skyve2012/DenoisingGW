import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from sklearn.preprocessing import scale
import tensorflow.contrib.slim as slim
import random
#from modules.modules import network_modules
import copy
import time
import sys
import os
import math
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from tensorflow.contrib import rnn
import matplotlib
from functions import *
matplotlib.use('Agg')





##########
# parameters setup
##########

hidden_num = [64]
batch_size = 32
cell = [rnn.LSTMCell]
learning_rate = 0.0005
latent_dim = 128
sample_length = 8192
print_step = 50



##########
# build model
##########


class denoiser_model():
    def __init__(self, hidden_num, batch_size, cell, learning_rate, latent_dim, sample_length):
        self.hidden_num = hidden_num
        self.cell = cell
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.laten_dim = latent_dim
        self.sample_length = sample_length
        
        
        
        
    def build_model(self):
        noisy_input = tf.placeholder(tf.float32, [None, self.sample_length], name='noisy_input')
        reference_input = tf.placeholder(tf.float32, [None, self.sample_length], name='ref_input')
        
        
        
        with tf.variable_scope('encoder') as enc:
            enc_outputs, enc_states = encoder(noisy_input,
                                                   copy.deepcopy(self.cell), self.hidden_num,
                                                   is_bidirectional=True,
                                                   activation_fn=tf.nn.tanh)
            
            
        with tf.variable_scope('decoder') as dec:
            outputs_, dec_states_ = stair_decoder_nolast(enc_outputs,
                                               copy.deepcopy(self.cell), self.hidden_num,
                                               is_bidirectional=True,
                                               activation_fn=tf.nn.tanh, states=enc_states)



        with tf.variable_scope('decoder1'):
            outputs, dec_states = stair_decoder(outputs_,
                                               copy.deepcopy(self.cell), self.hidden_num,
                                               is_bidirectional=True,
                                               activation_fn=tf.nn.tanh, states=dec_states_, sample_length=self.sample_length)
            
            
            
        scalars = tf.Variable(tf.constant(1.,tf.float32, [1]))
        outputs = outputs * scalars
        outputs = tf.squeeze(outputs, axis=-1)

        loss = tf.reduce_mean(tf.square(reference_input - outputs), keep_dims=False)
        
        
        self.noisy_input = noisy_input
        self.ref_input = reference_input
        self.output = outputs
        self.loss = loss
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        print('model is built')
        
        
    def train(self, batch_input, batch_reference, sess):
        feed_dict = {self.noisy_input: batch_input, self.ref_input: batch_reference}
        train_time = time.time()
        _, loss_return, denoised_return = sess.run([self.train_op, self.loss, self.output], feed_dict=feed_dict)
        print('train_time:')
        print(time.time() - train_time)
        return loss_return, denoised_return
    
    
    def test(self, batch_noise, sess):
        feed_dict = {self.noisy_input: batch_noise}
        test_time = time.time()
        denoised_return = sess.run([self.output], feed_dict=feed_dict)
        print('test_time:')
        print(time.time() - test_time)
        return denoised_return
        
        

        
        
        
        
##########
# dataset preparation
##########

# prepare the dataset and get training and testing sets

# get data ready for batch_input, batch_reference, batch_input_for_test variables




##########
# setup model and start training
##########

model_for_train = denoiser_model(hidden_num=hidden_num,
                                 batch_size=batch_size, cell=cell,
                                 learning_rate=learning_rate, latent_dim=latent_dim,
                                 sample_length=sample_length)
model_for_train.build_model()



config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())


for i in range(FLAGS.total_iteration):
    ##########
    # Here: need to write function to load batch_input, batch_reference and batch_input_for_test
    ##########
    loss_out, input_out, output_out = model_for_train.train(batch_input, batch_reference, sess)
    if i % print_step == 0:
        saver.save(sess, "./models/" + FLAGS.model_name, write_meta_graph=True)
        print("iteration: {0}, loss:{2:.3f}".format(i, loss))
        print("start testing the model")
        denoised_test = model_for_train.test(batch_input_for_test, sess)
        # some function for visulization here
        
        
        
    
 





                                               
        
        
        
        
