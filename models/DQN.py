import time
from tqdm import tqdm
import random
import numpy as np
import tensorflow as tf
from collections import deque
import sys
# from .base import Model

#set the seeds
random.seed(1)
np.random.seed(10)
tf.set_random_seed(100)


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
        self.history = History()
        #init parameters
        self.timeStep = 0
        self.epsilon = config.INITIAL_EPSILON

        self.stateInput = tf.placeholder(tf.int32, [None, self.config.seq_length])
        self.stateInputT = tf.placeholder(tf.int32, [None, self.config.seq_length])


        # embed = tf.get_variable("embed", [self.config.vocab_size, self.config.embed_dim]) #this is wrong way to initialize
        embed = tf.Variable(tf.random_uniform([self.config.vocab_size, self.config.embed_dim], -0.01, 0.01),name="embed")
        # embedT = tf.get_variable("embedT", [self.config.vocab_size, self.config.embed_dim])
        # print '$'*100
        word_embeds = tf.nn.embedding_lookup(embed, self.stateInput) # @codewalk: What is this line doing ?
        word_embedsT = tf.nn.embedding_lookup(embed, self.stateInputT) # @codewalk: What is this line doing ?
        # print '$'*100
        self.initializer = tf.truncated_normal_initializer(stddev = 0.02)
        # self.initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0, seed=None, dtype=tf.float32)        
        # self.initializer = tf.contrib.layers.xavier_initializer()
        # print '$'*100
        self.cell = tf.nn.rnn_cell.LSTMCell(self.config.rnn_size, initializer = self.initializer, state_is_tuple=True)
        self.cellT = tf.nn.rnn_cell.LSTMCell(self.config.rnn_size, initializer = self.initializer, state_is_tuple=True)
        # print '$'*100
        initial_state = self.cell.zero_state(self.config.BATCH_SIZE, tf.float32)
        initial_stateT = self.cellT.zero_state(self.config.BATCH_SIZE, tf.float32)
        # print '$'*100
        # early_stop = tf.constant(self.config.seq_length, dtype = tf.int32)
        # print '$'*100
        outputs, _ = tf.nn.rnn(self.cell, [tf.reshape(embed_t, [-1, self.config.embed_dim]) for embed_t in tf.split(1, self.config.seq_length, word_embeds)], dtype=tf.float32, initial_state = initial_state, scope = "LSTMN")
        outputsT, _ = tf.nn.rnn(self.cellT, [tf.reshape(embed_tT, [-1, self.config.embed_dim]) for embed_tT in tf.split(1, self.config.seq_length, word_embedsT)], dtype=tf.float32, initial_state = initial_stateT, scope = "LSTMT")
        # print '$'*100
        output_embed = tf.transpose(tf.pack(outputs), [1, 0, 2])
        output_embedT = tf.transpose(tf.pack(outputsT), [1, 0, 2])
        # print '$'*100
        mean_pool = tf.reduce_mean(output_embed, 1)
        mean_poolT = tf.reduce_mean(output_embedT, 1)
        # print '$'*100
        linear_output = tf.nn.relu(tf.nn.rnn_cell._linear(mean_pool, int(output_embed.get_shape()[2]), 1,0.01, scope="linearN"))
        linear_outputT = tf.nn.relu(tf.nn.rnn_cell._linear(mean_poolT, int(output_embedT.get_shape()[2]),1, 0.01, scope="linearT"))
        # print '$'*100

        self.action_value = tf.nn.rnn_cell._linear(linear_output, self.config.num_actions, 1,0.01, scope="actionN")
        self.action_valueT = tf.nn.rnn_cell._linear(linear_outputT, self.config.num_actions, 1,0.01, scope="actionT")

        self.object_value = tf.nn.rnn_cell._linear(linear_output, self.config.num_objects, 1,0.01, scope="objectN")
        self.object_valueT = tf.nn.rnn_cell._linear(linear_outputT, self.config.num_objects, 1,0.01, scope="objectT")

        self.target_action_value = tf.placeholder(tf.float32, [None])
        self.target_object_value = tf.placeholder(tf.float32, [None])

        self.action_indicator = tf.placeholder(tf.float32, [None, self.config.num_actions])
        self.object_indicator = tf.placeholder(tf.float32, [None, self.config.num_objects])

        self.pred_action_value = tf.reduce_sum(tf.mul(self.action_indicator, self.action_value), 1)
        self.pred_object_value = tf.reduce_sum(tf.mul(self.object_indicator, self.object_value), 1)

        self.target_qpred = tf.truediv(tf.add(self.target_action_value,self.target_object_value),2.0)
        self.qpred = tf.truediv(tf.add(self.pred_action_value,self.pred_object_value),2.0)

        summary_list = []        
        with tf.name_scope('delta'):
            self.delta_a = self.target_action_value - self.pred_action_value
            self.delta_o = self.target_object_value - self.pred_object_value
            self.variable_summaries(self.delta_a, 'delta_a',summary_list)
            self.variable_summaries(self.delta_o, 'delta_o',summary_list)
            # self.delta = self.target_qpred - self.qpred
            # self.variable_summaries(self.delta, 'delta',summary_list)

        if self.config.clipDelta:
                with tf.name_scope('clippeddelta'):
                    # self.delta = tf.clip_by_value(self.delta, self.config.minDelta, self.config.maxDelta, name='clipped_delta')

                    self.quadratic_part_a = tf.minimum(abs(self.delta_a), config.maxDelta)
                    self.linear_part_a = abs(self.delta_a) - self.quadratic_part_a

                    
                    self.quadratic_part_o = tf.minimum(abs(self.delta_o), config.maxDelta)
                    self.linear_part_o = abs(self.delta_o) - self.quadratic_part_o

                    self.quadratic_part = tf.concat(0,[self.quadratic_part_a,self.quadratic_part_o])
                    self.linear_part = tf.concat(0,[self.linear_part_a,self.linear_part_o])

                    # self.quadratic_part = tf.minimum(abs(self.delta), config.maxDelta)
                    # self.linear_part = abs(self.delta) - self.quadratic_part

                    # self.variable_summaries(self.delta, 'clippeddelta',summary_list)

                    # self.variable_summaries(self.linear_part_a, 'linear_part_a',summary_list)
                    # self.variable_summaries(self.quadratic_part_a, 'quadratic_part_a',summary_list)

                    # self.variable_summaries(self.linear_part_o, 'linear_part_o',summary_list)
                    # self.variable_summaries(self.quadratic_part_o, 'quadratic_part_o',summary_list)

                    self.variable_summaries(self.linear_part, 'linear_part',summary_list)
                    self.variable_summaries(self.quadratic_part, 'quadratic_part',summary_list)

                    

        

        with tf.name_scope('loss'):
            #self.loss = 0.5*tf.reduce_mean(tf.square(self.delta), name='loss')
            # self.loss_a = tf.reduce_mean(0.5*tf.square(self.quadratic_part_a) + config.clipDelta * self.linear_part_a, name='loss_a')  
            # self.variable_summaries(self.loss_a, 'loss_a',summary_list)

            # self.loss_o = tf.reduce_mean(0.5*tf.square(self.quadratic_part_o) + config.clipDelta * self.linear_part_o, name='loss_o')  
            # self.variable_summaries(self.loss_o, 'loss_o',summary_list)

            self.loss = tf.reduce_mean(0.5*tf.square(self.quadratic_part) + config.clipDelta * self.linear_part, name='loss')  
            self.variable_summaries(self.loss, 'loss',summary_list)            

        self.W = ["LSTMN", "linearN", "actionN", "objectN"]
        self.target_W = ["LSTMT", "linearT", "actionT", "objectT"]

        # for i in range(len(self.W)):
        #     vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = self.W[i])
        #     varsT = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = self.target_W[i])

        #     with tf.name_scope('activationsN'):
        #         summary_list.extend(map(lambda x:tf.histogram_summary('activations/'+str(x.name), x), vars))
        #     with tf.name_scope('activationsT'):
        #         summary_list.extend(map(lambda x:tf.histogram_summary('activations/'+str(x.name), x), varsT))

        self.summary_placeholders = {}
        self.summary_ops = {}
        if self.config.TUTORIAL_WORLD:
            scalar_summary_tags = ['average.q_a','average.q_o','average_reward','average_num_pos_reward','number_of_episodes','quest1_average_reward_cnt', \
                    'quest2_average_reward_cnt','quest3_average_reward_cnt']
        else:
            scalar_summary_tags = ['average.q_a','average.q_o','average_reward','average_num_pos_reward','number_of_episodes','quest1_average_reward_cnt']

        for tag in scalar_summary_tags:
            self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
            self.summary_ops[tag]  = tf.scalar_summary('evaluation_data/'+tag, self.summary_placeholders[tag])

        # Clipping gradients

        # self.optim_ = tf.train.RMSPropOptimizer(learning_rate = self.config.LEARNING_RATE)
        # tvars = tf.trainable_variables()
        # def ClipIfNotNone(grad,var):
        #     if grad is None:
        #         return (grad, var)
        #     return (tf.clip_by_norm(grad,10), var)
        # grads = [ClipIfNotNone(i,var) for i,var in self.optim_.compute_gradients(self.loss, tvars)]

        # self.optim = self.optim_.apply_gradients(grads)
        # self.optim = tf.train.RMSPropOptimizer(learning_rate = self.config.LEARNING_RATE).minimize(self.loss_a + self.loss_o)
        # self.optim = tf.train.RMSPropOptimizer(learning_rate = self.config.LEARNING_RATE).minimize(self.loss)
        # self.optim = tf.train.AdagradOptimizer(learning_rate = self.config.LEARNING_RATE).minimize(self.loss)
        # self.optim_a = tf.train.AdagradOptimizer(learning_rate = self.config.LEARNING_RATE).minimize(self.loss_a)
        # self.optim_o = tf.train.AdagradOptimizer(learning_rate = self.config.LEARNING_RATE).minimize(self.loss_o)
        # self.optim1 = tf.train.AdamOptimizer(learning_rate = self.config.LEARNING_RATE).minimize(self.loss_a)
        # self.optim2 = tf.train.AdamOptimizer(learning_rate = self.config.LEARNING_RATE).minimize(self.loss_o)
        self.optim = tf.train.AdamOptimizer(learning_rate = self.config.LEARNING_RATE).minimize(self.loss)
        if not(self.config.LOAD_WEIGHTS and self.load_weights()):
            # self.merged = tf.merge_all_summaries()
            self.merged = tf.merge_summary(summary_list)
            self.train_writer = tf.train.SummaryWriter(self.config.summaries_dir + '/train/'+str(self.config.game_num),self.session.graph)            
            self.session.run(tf.initialize_all_variables())


        self.copyTargetQNetworkOperation()
        self.saver = tf.train.Saver()


    def variable_summaries(self, var, name,list_summary):
        """Attach a lot of summaries to a Tensor."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            list_summary.append(tf.scalar_summary('training_data/mean/' + name, mean))
            # with tf.name_scope('stddev'):
            #     stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
            # list_summary.append(tf.scalar_summary('training_data/sttdev/' + name, stddev))
            list_summary.append(tf.scalar_summary('training_data/max/' + name, tf.reduce_max(var)))
            list_summary.append(tf.scalar_summary('training_data/min/' + name, tf.reduce_min(var)))
            # list_summary.append(tf.histogram_summary(name, var))

    def inject_summary(self, tag_dict, step):
        summary_str_lists = self.session.run([self.summary_ops[tag] for tag in tag_dict.keys()], { \
        self.summary_placeholders[tag]: value for tag, value in tag_dict.items()})
        for summary_str in summary_str_lists:
            self.train_writer.add_summary(summary_str, self.timeStep)                    

    def copyTargetQNetworkOperation(self):
        for i in range(len(self.W)):
            vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = self.W[i])
            varsT = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = self.target_W[i])
            # for i in vars:
            # 	print i.name
            # print "Over ."
            # for i in varsT:
            # 	print i.name
            # print len(vars)
            # print len(varsT)
            copy_op = zip(varsT, vars)
            self.session.run(map(lambda (x,y): x.assign(y.eval(session = self.session)),copy_op))

            # with tf.name_scope('activations'):
            #     map(lambda x:tf.histogram_summary('activations/'+str(x.name), x), vars)
            # value1 = self.session.run(vars)
            # value2 = self.session.run(varsT)
            # print len(value1)
            # print len(value2)
            # val_op = zip(value1, value2)
            # for x, y in val_op:
            # 	res = x - y
            # 	print sum(res)





    def train(self):

        s_t, action, obj, reward, s_t_plus_1, terminal = self.memory.sample()
        state_batch = s_t
        action_batch = action
        obj_batch = obj
        reward_batch = reward
        nextState_batch = s_t_plus_1

        # Step 2: calculate y
        target_action_batch = []
        target_object_batch = []
        QValue_action_batch = self.action_valueT.eval(feed_dict={self.stateInputT:nextState_batch},session = self.session)
        QValue_object_batch = self.object_valueT.eval(feed_dict={self.stateInputT:nextState_batch},session = self.session)


        for i in range(0,self.config.BATCH_SIZE):
            if terminal[i]:
                target_action_batch.append(reward_batch[i])
                target_object_batch.append(reward_batch[i])
            else:
                target_action_batch.append(reward_batch[i] + self.config.GAMMA* np.max(QValue_action_batch[i]))
                target_object_batch.append(reward_batch[i] + self.config.GAMMA* np.max(QValue_object_batch[i]))

        if self.timeStep%self.config.EVAL == self.config.trainfreq:
            _ , summary = self.session.run([self.optim, self.merged],feed_dict={
                    self.target_action_value : target_action_batch,
                    self.target_object_value : target_object_batch,
                    self.action_indicator : action_batch,
                    self.object_indicator : obj_batch,
                    self.stateInput : state_batch
                    })
            self.train_writer.add_summary(summary, self.timeStep)

            # _ , summary = self.session.run([self.optim2, self.merged],feed_dict={
            #         self.target_action_value : target_action_batch,
            #         self.target_object_value : target_object_batch,
            #         self.action_indicator : action_batch,
            #         self.object_indicator : obj_batch,
            #         self.stateInput : state_batch
            #         })
            # self.train_writer.add_summary(summary, self.timeStep)            
        else:
            _ = self.session.run([self.optim],feed_dict={
                    self.target_action_value : target_action_batch,
                    self.target_object_value : target_object_batch,
                    self.action_indicator : action_batch,
                    self.object_indicator : obj_batch,
                    self.stateInput : state_batch
                    })
            # _ = self.session.run([self.optim2],feed_dict={
            #         self.target_action_value : target_action_batch,
            #         self.target_object_value : target_object_batch,
            #         self.action_indicator : action_batch,
            #         self.object_indicator : obj_batch,
            #         self.stateInput : state_batch
            #         })


        if self.timeStep % self.config.UPDATE_FREQUENCY == 0:
            # print "Copying weights."
            self.copyTargetQNetworkOperation()

    def setPerception(self, state, reward, action_indicator, object_indicator, nextstate,terminal,evaluate = False): #nextObservation,action,reward,terminal):
        self.history.add(nextstate)
        if not evaluate:
            self.memory.add(state, action_indicator, object_indicator, reward, nextstate, terminal)
        if self.timeStep > self.config.REPLAY_START_SIZE and self.memory.count > self.config.REPLAY_START_SIZE:
            # Train the network
            if (not evaluate ) and (self.timeStep % self.config.trainfreq == 0):
            	# print "Started Training."
                self.train()
        if not evaluate:
            self.timeStep += 1


    def getAction(self, availableObjects, evaluate = False):
        action_index = 0
        object_index = 0
        curr_epsilon = self.epsilon
        if evaluate:
            curr_epsilon = 0.05
            
        if random.random() <= curr_epsilon:
            action_index = random.randrange(self.config.num_actions)
            object_index = random.randrange(self.config.num_objects)
        else:
            state_batch = np.zeros([self.config.batch_size, self.config.seq_length])
            state_batch[0] = self.history.get()
            QValue_action = self.action_value.eval(feed_dict={self.stateInput:state_batch},session = self.session)[0]
            bestAction = np.where(QValue_action == np.max(QValue_action))[0]
            QValue_object = self.object_value.eval(feed_dict={self.stateInput:state_batch},session = self.session)[0]
            for i in range(QValue_object.size):
                if i in availableObjects:
                    QValue_object[i] = -sys.maxint - 1
            bestObject = np.where(QValue_object == np.max(QValue_object))[0]
            action_index = bestAction[random.randrange(0,bestAction.shape[0])]
            object_index = bestObject[random.randrange(0,bestObject.shape[0])]


        if not evaluate:
            self.epsilon = self.config.FINAL_EPSILON + max(0, (self.config.INITIAL_EPSILON - self.config.FINAL_EPSILON) * (self.config.EXPLORE - max(0, self.timeStep - self.config.REPLAY_START_SIZE))/self.config.EXPLORE)


        return action_index, object_index

    def getQValues(self, availableObjects, evaluate = False):
            
        state_batch = np.zeros([self.config.batch_size, self.config.seq_length])
        state_batch[0] = self.history.get()
        QValue_action = self.action_value.eval(feed_dict={self.stateInput:state_batch},session = self.session)[0]
        QValue_object = self.object_value.eval(feed_dict={self.stateInput:state_batch},session = self.session)[0]

        return QValue_action, QValue_object
    
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
          

