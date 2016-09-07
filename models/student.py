import time
from tqdm import tqdm
import random
import numpy as np
import tensorflow as tf
from collections import deque

# from .base import Model



import os
from history import History
from replay_memory import ReplayMemory
import cPickle as pickle

class DQN:
    def __init__(self, config):

        #init replay memory
        self.session = tf.Session()
        self.config = config
        
        self.memory = self.load_replay_memory(config)
        self.history = History(config)
        #init parameters
        self.timeStep = 0
        self.epsilon = config.INITIAL_EPSILON
        self.actions = config.NUM_ACTIONS

        self.stateInput = tf.placeholder(tf.int32, [None, self.config.seq_length])


        embed = tf.get_variable("embed", [self.config.vocab_size, self.config.embed_dim])
        

        word_embeds = tf.nn.embedding_lookup(embed, self.stateInput) # @codewalk: What is this line doing ?
        

        self.initializer = tf.truncated_normal_initializer(stddev = 0.02)

        self.cell = tf.nn.rnn_cell.LSTMCell(self.config.rnn_size, initializer = self.initializer)
        

        initial_state = self.cell.zero_state(self.config.BATCH_SIZE, tf.float32)
        

        # early_stop = tf.constant(self.config.seq_length, dtype = tf.int32)

        outputs, _ = tf.nn.rnn(self.cell, [tf.reshape(embed_t, [-1, self.config.embed_dim]) for embed_t in tf.split(1, self.config.seq_length, word_embeds)], dtype=tf.float32, initial_state = initial_state, scope = "LSTMN")
        

        output_embed = tf.transpose(tf.pack(outputs), [1, 0, 2])
        

        mean_pool = tf.nn.relu(tf.reduce_mean(output_embed, 1))
        

        linear_output = tf.nn.relu(tf.nn.rnn_cell._linear(mean_pool, int(output_embed.get_shape()[2]), 0.0, scope="linearN"))
        

        #we calculate the Q values. For the Student Network
        self.action_value = tf.nn.rnn_cell._linear(linear_output, self.config.num_actions, 0.0, scope="actionN")
        self.object_value = tf.nn.rnn_cell._linear(linear_output, self.config.num_objects, 0.0, scope="objectN")
        

        #here we will input the teachers q value
        self.target_action_value = tf.placeholder(tf.float32, [None,self.config.seq_length])
        self.target_object_value = tf.placeholder(tf.float32, [None,self.config.seq_length])
        #here we calculate the probabilities for the teacher network
        self.target_action_prob = tf.nn.softmax(tf.truediv(self.target_action_value,self.temperature))
        self.target_object_prob = tf.nn.softmax(tf.truediv(self.target_object_value,self.temperature))

        #here we calculate the probabilities for the student network
        self.pred_action_prob = tf.nn.softmax(self.action_value)
        self.pred_object_prob = tf.nn.softmax(self.object_value)


        cross_entropy_action = -tf.reduce_sum(self.target_action_prob*tf.log(self.pred_action_prob))
        entropy_action = -tf.reduce_sum(self.target_action_prob*tf.log(self.target_action_prob))

        cross_entropy_object = -tf.reduce_sum(self.target_object_prob*tf.log(self.pred_object_prob))
        entropy_object = -tf.reduce_sum(self.object_action_prob*tf.log(self.target_object_prob))

        self.kl_divergence = 0.5 * (cross_entropy_action - entropy_action + cross_entropy_object - entropy_object)

        if self.config.clipDelta:
            self.loss = tf.clip_by_norm(self.kl_divergence, 100, name='loss') #@codewalk: discuss this

        # Clipping gradients

        self.optim_ = tf.train.RMSPropOptimizer(learning_rate = self.config.LEARNING_RATE)
        tvars = tf.trainable_variables()
        def ClipIfNotNone(grad,var):
            if grad is None:
                return grad
            return tf.clip_by_norm(grad,20)
        grads = [ClipIfNotNone(i,var) for i,var in zip(tf.gradients(self.loss, tvars),tvars)]
        self.optim = self.optim_.apply_gradients(zip(grads, tvars))


        if not(self.config.LOAD_WEIGHTS and self.load_weights()):
            self.session.run(tf.initialize_all_variables())
        self.saver = tf.train.Saver()

    def train(self):

        s_t, Q_action, Q_object = self.memory.sample()
        state_batch = s_t
        target_action_batch = Q_action
        target_object_batch = Q_object

        self.optim.run(feed_dict={
                self.target_action_value : target_action_batch,
                self.target_object_value : target_object_batch,
                self.stateInput : state_batch
                },session = self.session)

        # save network every 10000 iteration
        if self.timeStep % 10000 == 0:
            if not os.path.exists(os.getcwd()+'/Savednetworks'):
                os.makedirs(os.getcwd()+'/Savednetworks')
            self.saver.save(self.session, os.getcwd()+'/Savednetworks/'+'network' + '-dqn', global_step = self.timeStep)

    
    def load_weights(self):
        print 'inload weights'
        if not os.path.exists(os.getcwd()+'/Savednetworks'):
            return False    
        
        list_dir = sorted(os.listdir(os.getcwd()+'/Savednetworks'))
        if not any(item.startswith('network-dqn') for item in list_dir):
            return False
        
        print 'weights loaded'
        self.saver.restore(self.session, os.getcwd()+'/Savednetworks/'+list_dir[-2])        
        return True

    
    def load_replay_memory(self,config):
        if os.path.exists(config.model_dir+'/replay_file.save'):
            fp = open(config.model_dir+'/replay_file.save','rb')
            memory = pickle.load(fp)
            fp.close()
        else:
            memory = ReplayMemory(config)
        return memory
          

