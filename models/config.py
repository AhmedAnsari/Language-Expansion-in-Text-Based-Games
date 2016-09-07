#Configurations for Atari Games
import os
class Config:
    def __init__(self):
        #replaymemory
        self.batch_size = 64 # size of minibatch
        self.BATCH_SIZE = 64 # size of minibatch
        self.memory_size = 100000 # number of previous transitions to remember,
        self.GAMMA = 0.5 # decay rate of past observations
        self.model_dir = os.getcwd()+'/Savednetworks/'

        #dqn
        self.vocab_size = 100
        self.embed_dim = 100
        self.rnn_size = 100
        self.clipDelta = True
        self.minDelta = -1.0
        self.maxDelta = 1.0
        self.LEARNING_RATE = 0.0005
        # self.GRADIENT_MOMENTUM = 0.95
        # self.SQUARED_GRADIENT_MOMENTUM = 0.95
        self.UPDATE_FREQUENCY = 10000 # Number of parameter updates after which the target parameters are updated
        self.LOAD_WEIGHTS = False
        self.REPLAY_START_SIZE = 1000 #minimum number of previous transitions to be stored before training starts
        self.INITIAL_EPSILON = 1 # starting value of epsilon
        self.FINAL_EPSILON = 0.2 # final value of epsilon
        self.trainfreq = 4
        self.EXPLORE = 1000000. # frames over which to anneal epsilon

        #parameters for the game
        self.MAX_FRAMES = 1000000
        self.max_episode_length = 30
        self.game_dir = '../text-world'
        self.seq_length = 30
        

        self.device = '/cpu:0'        
        
    
    def setnumactions(self,numactions):
        self.num_actions = numactions
        self.NUM_ACTIONS = numactions
    def setnumobjects(self,numobjects):
        self.num_objects = numobjects
        self.NUM_OBJECTS = numobjects

