#Configurations for Atari Games
import os
class Config:
    def __init__(self):
        #parameters for DQN
        self.weighted = False
        self.BATCH_SIZE = 32 # size of minibatch
        self.REPLAY_MEMORY = 1000000 # number of previous transitions to remember,
        self.GAMMA = 0.99 # decay rate of past observations
        self.UPDATE_FREQUENCY = 10000 # Number of parameter updates after which the target parameters are updated
        self.LEARNING_RATE = 0.00025
        self.GRADIENT_MOMENTUM = 0.95
        self.SQUARED_GRADIENT_MOMENTUM = 0.95
        self.MIN_SQUARED_GRADIENT = 0.01
        self.INITIAL_EPSILON = 1 # starting value of epsilon
        self.FINAL_EPSILON = 0.01 # final value of epsilon
        self.EXPLORE = 1000000. # frames over which to anneal epsilon
        self.REPLAY_START_SIZE = 50000 #minimum number of previous transitions to be stored before training starts
        self.model_dir = os.getcwd()+'/Savednetworks/'
        self.noopmax = 30
        self.trainfreq = 4
        self.heads = 10
        #parameters for the game
        self.SAMPLE_STATES = 32
        self.MAX_FRAMES = 50000000
        self.EVAL = 50000
        self.NUM_EVAL_STEPS = 10000
        self.GAME = 'SpaceInvaders-v0'
        self.K = 4
        self.LOAD_WEIGHTS = True
        self.Dueling = False
        self.Double_DQN = False
        self.maxDelta = 1
        self.minDelta = -1
        self.clipDelta = True
        self.minR = -1
        self.maxR = 1
        self.clipR = True
        self.DISPLAY = False       
        #from devsisters for History
        self.batch_size = self.BATCH_SIZE
        self.history_length = 4
        self.screen_height = 84
        self.screen_width = 84
        
        self.device = '/cpu:0'        
        
    def test(self):
        self.REPLAY_START_SIZE = 500 #minimum number of previous transitions to be stored before training starts
        self.SAMPLE_STATES = 32
        self.EVAL = 200
        self.NUM_EVAL_STEPS = 200
        self.DISPLAY = True        

    def train(self):
        self.REPLAY_START_SIZE = 50000 #minimum number of previous transitions to be stored before training starts
        self.SAMPLE_STATES = 32
        self.EVAL = 50000
        self.NUM_EVAL_STEPS = 10000
        self.DISPLAY = False     
    
    def setnumactions(self,numactions):
        self.NUM_ACTIONS = numactions
