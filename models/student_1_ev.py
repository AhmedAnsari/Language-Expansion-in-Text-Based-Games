import random
import numpy as np
import tensorflow as tf
import sys

# from .base import Model



import os
from history import History
class student:
    def __init__(self, config):

        #init replay memory
        conf = tf.ConfigProto()
        conf.gpu_options.allow_growth=True
        self.session = tf.Session(config=conf)
        self.config = config
        #init parameters
        self.timeStep = 0
        self.stateInput = tf.placeholder(tf.int32, [None, self.config.seq_length])
        self.data = {}
        self.history = [History(),History(),History()]
        self.BATCH_SIZE = 256

        #set config.final_vocab_size manually
        embed = tf.Variable(tf.random_uniform([self.config.final_vocab_size, self.config.embed_dim], -1.0, 1.0),name="embed")

        self.word_embeds = tf.nn.embedding_lookup(embed, self.stateInput)
        self.initializer = tf.truncated_normal_initializer(stddev = 0.02)
        self.cell = tf.nn.rnn_cell.LSTMCell(self.config.rnn_size, initializer = self.initializer, state_is_tuple=True)
        initial_state = self.cell.zero_state(self.BATCH_SIZE, tf.float32)
        outputs, _ = tf.nn.rnn(self.cell, [tf.reshape(embed_t, [-1, self.config.embed_dim]) for embed_t in tf.split(1, self.config.seq_length, self.word_embeds)], dtype=tf.float32, initial_state = initial_state, scope = "LSTMN")
        self.output_embed = tf.transpose(tf.pack(outputs), [1, 0, 2])
        self.mean_pool = tf.reduce_mean(self.output_embed, 1)
        linear_output = tf.nn.relu(tf.nn.rnn_cell._linear(self.mean_pool, int(self.output_embed.get_shape()[2]), 1.0, 0.01, scope="linearN"))


        #we calculate the Q values. For the Student Network
        self.action_value_1 = tf.nn.rnn_cell._linear(linear_output, self.config.num_actions, 1.0, 0.01, scope="actionN1")
        self.object_value_1 = tf.nn.rnn_cell._linear(linear_output, self.config.num_objects, 1.0, 0.01, scope="objectN1")

        self.action_value_2 = tf.nn.rnn_cell._linear(linear_output, self.config.num_actions, 1.0, 0.01, scope="actionN2")
        self.object_value_2 = tf.nn.rnn_cell._linear(linear_output, self.config.num_objects, 1.0, 0.01, scope="objectN2")

        self.action_value_3 = tf.nn.rnn_cell._linear(linear_output, self.config.num_actions, 1.0, 0.01, scope="actionN3")
        self.object_value_3 = tf.nn.rnn_cell._linear(linear_output, self.config.num_objects, 1.0, 0.01, scope="objectN3")



        #here we will input the teachers q value
        self.target_action_value = tf.placeholder(tf.float32, [None,self.config.num_actions])
        self.target_object_value = tf.placeholder(tf.float32, [None,self.config.num_objects])

        #here we calculate the probabilities for the teacher network
        self.target_action_prob = tf.nn.softmax(tf.truediv(self.target_action_value,self.config.temperature))
        self.target_object_prob = tf.nn.softmax(tf.truediv(self.target_object_value,self.config.temperature))

        #here we calculate the probabilities for the student network
        self.pred_action_prob_1 = tf.nn.softmax(self.action_value_1)
        self.pred_object_prob_1 = tf.nn.softmax(self.object_value_1)

        self.pred_action_prob_2 = tf.nn.softmax(self.action_value_2)
        self.pred_object_prob_2 = tf.nn.softmax(self.object_value_2)

        self.pred_action_prob_3 = tf.nn.softmax(self.action_value_3)
        self.pred_object_prob_3 = tf.nn.softmax(self.object_value_3)


        entropy_action = -tf.reduce_sum(self.target_action_prob*tf.log(self.target_action_prob),reduction_indices = [1])
        entropy_object = -tf.reduce_sum(self.target_object_prob*tf.log(self.target_object_prob),reduction_indices = [1])

        cross_entropy_action_1 = -tf.reduce_sum(self.target_action_prob*tf.log(self.pred_action_prob_1),reduction_indices = [1])
        cross_entropy_object_1 = -tf.reduce_sum(self.target_object_prob*tf.log(self.pred_object_prob_1),reduction_indices = [1])

        cross_entropy_action_2 = -tf.reduce_sum(self.target_action_prob*tf.log(self.pred_action_prob_2),reduction_indices = [1])
        cross_entropy_object_2= -tf.reduce_sum(self.target_object_prob*tf.log(self.pred_object_prob_2),reduction_indices = [1])

        cross_entropy_action_3 = -tf.reduce_sum(self.target_action_prob*tf.log(self.pred_action_prob_3),reduction_indices = [1])
        cross_entropy_object_3= -tf.reduce_sum(self.target_object_prob*tf.log(self.pred_object_prob_3),reduction_indices = [1])




        self.kl_divergence_1 = tf.reduce_mean(0.5 * (cross_entropy_action_1 - entropy_action + cross_entropy_object_1 - entropy_object))

        self.kl_divergence_2 = tf.reduce_mean(0.5 * (cross_entropy_action_2 - entropy_action + cross_entropy_object_2 - entropy_object))

        self.kl_divergence_3 = tf.reduce_mean(0.5 * (cross_entropy_action_3 - entropy_action + cross_entropy_object_3 - entropy_object))



        self.optim_1 = tf.train.AdamOptimizer(learning_rate = self.config.LEARNING_RATE).minimize(self.kl_divergence_1)
        self.optim_2 = tf.train.AdamOptimizer(learning_rate = self.config.LEARNING_RATE).minimize(self.kl_divergence_2)
        self.optim_3 = tf.train.AdamOptimizer(learning_rate = self.config.LEARNING_RATE).minimize(self.kl_divergence_3)


        self.summary_placeholders = {}
        self.summary_ops = {}
        
        tags = ['average_reward','average_numrewards','number_of_episodes','quest1_average_reward_cnt']
        scalar_summary_tags = []
        for i in range(1,4):
            scalar_summary_tags.append([tag + str(i) for tag in tags])

        for i in range(3):            
            for tag in scalar_summary_tags[i]:
                self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
                self.summary_ops[tag]  = tf.scalar_summary('evaluation_data/'+tag, self.summary_placeholders[tag])


        self.saver = tf.train.Saver()
        self.train_writer = tf.train.SummaryWriter(self.config.summaries_dir + '/train/'+str(self.config.game_num),self.session.graph)
        if not(self.config.LOAD_WEIGHTS and self.load_weights()):            
            self.session.run(tf.initialize_all_variables())


    def inject_summary(self, tag_dict, step):
        summary_str_lists = self.session.run([self.summary_ops[tag] for tag in tag_dict.keys()], { \
        self.summary_placeholders[tag]: value for tag, value in tag_dict.items()})
        for summary_str in summary_str_lists:
            self.train_writer.add_summary(summary_str, self.timeStep)

    def train(self,game_id):

        s_t, Q_action, Q_object = self.sample(self.data[game_id])
        state_batch = s_t
        target_action_batch = Q_action
        target_object_batch = Q_object
        if game_id == 1:
            self.optim_1.run(feed_dict={
                    self.target_action_value : target_action_batch,
                    self.target_object_value : target_object_batch,
                    self.stateInput : state_batch
                    },session = self.session)
        elif game_id == 2:
            self.optim_2.run(feed_dict={
                    self.target_action_value : target_action_batch,
                    self.target_object_value : target_object_batch,
                    self.stateInput : state_batch
                    },session = self.session)
        elif game_id == 3:
            self.optim_3.run(feed_dict={
                    self.target_action_value : target_action_batch,
                    self.target_object_value : target_object_batch,
                    self.stateInput : state_batch
                    },session = self.session)

    def load_weights(self):
        print 'inload weights'
        if not os.path.exists(os.getcwd()+'/StudentSavednetworks'):
            return False

        list_dir = sorted(os.listdir(os.getcwd()+'/StudentSavednetworks'))
        if not any(item.startswith('network-student') for item in list_dir):
            return False

        print 'weights loaded'
        self.saver.restore(self.session, os.getcwd()+'/StudentSavednetworks/'+list_dir[-2])
        return True

    def sample(self,memory):
        # print "$"*100
        # print len(memory)
        # print "$"*100
        batch = random.sample(memory,self.BATCH_SIZE)
        s_t = [mem[0] for mem in batch]
        action_values = [mem[1] for mem in batch]
        object_values = [mem[2] for mem in batch]
        return s_t, action_values, object_values

    def getAction(self, availableObjects, game_id):
        action_index = 0
        object_index = 0
        curr_epsilon = 0.0

        if random.random() <= curr_epsilon:
            action_index = random.randrange(self.config.num_actions)
            object_index = random.randrange(self.config.num_objects)
        else:
            state_batch = np.zeros([self.BATCH_SIZE, self.config.seq_length])
            state_batch[0] = self.history[game_id-1].get()
            if game_id==1:
                action_value = self.action_value_1
                object_value = self.object_value_1
            elif game_id==2:
                action_value = self.action_value_2
                object_value = self.object_value_2

            elif game_id==3:
                action_value = self.action_value_3
                object_value = self.object_value_3

            QValue_action = action_value.eval(feed_dict={self.stateInput:state_batch},session = self.session)[0]
            bestAction = np.where(QValue_action == np.max(QValue_action))[0]
            QValue_object = object_value.eval(feed_dict={self.stateInput:state_batch},session = self.session)[0]
            # for i in range(QValue_object.size):
            #     if i not in availableObjects:
            #         QValue_object[i] = -sys.maxint - 1
            bestObject = np.where(QValue_object == np.max(QValue_object))[0]
            action_index = bestAction[random.randrange(0,bestAction.shape[0])]
            object_index = bestObject[random.randrange(0,bestObject.shape[0])]
        return action_index, object_index




