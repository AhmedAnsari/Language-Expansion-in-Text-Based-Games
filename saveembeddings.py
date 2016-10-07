# -------------------------
# Project: DQN Nature implementation
# Author: ghulamahmedansari,Rakesh Menon
# -------------------------

import os
from models.DQN import DQN
# from models.bow_DQN import DQN
# from models.lstdq import DQN
import numpy as np
import cPickle as cpickle
from models.config import Config
from tqdm import tqdm
import random
import sys
from environment import Environment


def savegame(config):
    # Step 1: init Game
    env = Environment(config.game_num) #1 is for main game 2 is for evaluation
    ###################
    # Step 2: init DQN
    actions = env.action_size()
    objects = env.object_size()
    config.setnumactions(actions)
    config.setnumobjects(objects)
    config.setvocabsize(env.vocab_size())

    brain = DQN(config)

    # checkStates = None
    #adding progress bar for training
    dic = {}
    with open("symbolMapping.txt", 'r') as fp:
        data = fp.read().split('\n')
        for i in data:
            splitdata = i.split(' ')
            dic[int(splitdata[1])] = splitdata[0]


    fp = open("lstm_100_embeddings.txt","w")
    for i in range(config.vocab_size):
        state = np.zeros([config.batch_size,config.seq_length])
        state[0,-1]=i
        embedding = brain.output_embedT.eval(feed_dict={brain.stateInputT = state},session=brain.session)[0,0,:]
        print >> fp,dic[i]
        print >> fp,embedding
    brain.session.close()

def main():
    config = Config()
 #   config.test()
    config.game_num = sys.argv[1]
    savegame(config)

if __name__ == '__main__':
    main()
