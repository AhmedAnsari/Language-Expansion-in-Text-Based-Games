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
        self.summaries_dir = os.getcwd()+'/logs/'

        #dqn
        # self.vocab_size =
        self.state_dim = 85 #used for for bow case
        self.vocab_size = 100
        self.embed_dim = 20 #@Ansari check this
        self.rnn_size = 100 #@Ansari check this
        self.clipDelta = True
        self.minDelta = -1.0
        self.maxDelta = 1.0
        self.minReward = -1.0
        self.maxReward = 10.0
        self.LEARNING_RATE = 0.0005
        # self.GRADIENT_MOMENTUM = 0.95
        # self.SQUARED_GRADIENT_MOMENTUM = 0.95
        self.UPDATE_FREQUENCY = 1000 # Number of parameter updates after which the target parameters are updated
        self.LOAD_WEIGHTS = True
        self.REPLAY_START_SIZE = 1000 #minimum number of previous transitions to be stored before training starts
        self.INITIAL_EPSILON = 1 # starting value of epsilon
        self.FINAL_EPSILON = 0.2 # final value of epsilon
        self.trainfreq = 4
        self.EXPLORE =  100000# frames over which to anneal epsilon

        #parameters for the game
        self.MAX_FRAMES = 2000000
        self.max_episode_length = 20
        self.game_dir = '../text-world'
        self.seq_length = 100 #@Ansari check this
        self.EVAL = 1000
        self.NUM_EVAL_STEPS = self.EVAL
        self.SAMPLE_STATES = 64
        self.device = '/cpu:0'


        self.TUTORIAL_WORLD = False
        self.game_num = 1
        self.testepsilon = 0.05
        #for student
        self.temperature =0.01
        self.final_vocab_size = 160

    def setnumactions(self,numactions):
        self.num_actions = numactions
        self.NUM_ACTIONS = numactions
    def setnumobjects(self,numobjects):
        self.num_objects = numobjects
        self.NUM_OBJECTS = numobjects
    def setvocabsize(self,vocab_size):
        self.vocab_size = vocab_size

