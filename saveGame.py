# -------------------------
# Project: DQN Nature implementation
# Author: ghulamahmedansari,Rakesh Menon
# -------------------------

import os
# from models.DQN import DQN
# from models.bow_DQN import DQN
from models.lstdq import DQN
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
    pbar = tqdm(total = config.MAX_FRAMES, desc='Training Progress')
    episode_length = 0
    num_episodes = 0
    total_reward = 0
    MAX_STEPS = 1000000
    totalSteps = 0
    MEM_STEPS = 10000
    memory = []
    while totalSteps < MAX_STEPS:
        totalSteps += 1
        if env.START_NEW_GAME:
            episode_length = 0
            env.START_NEW_GAME = False
            state, reward, terminal, availableObjects = env.newGame()
            brain.history.add(state)
        action_indicator = np.zeros(actions)
        object_indicator = np.zeros(objects)
        #predict
        action_index,object_index = brain.getAction(availableObjects)
        Qactions, Qobjects = brain.getQValues(availableObjects)
        action_indicator[action_index] = 1
        object_indicator[object_index] = 1
        memory.append((state, Qactions, Qobjects))
        #act
        nextstate,reward,terminal, availableObjects = env.step(action_index,object_index)
        total_reward += reward
        episode_length += 1
        #observe
        brain.setPerception(state, reward, action_indicator, object_indicator, nextstate, terminal, False)
        state = nextstate

        if (totalSteps % MEM_STEPS == 0):
            fileName = str(totalSteps % MEM_STEPS) + "_mem.txt"
            with open(fileName, "a") as fp:
                for i in range(len(memory)):
                    for j in memory[0]:
                        print >> fp, j,
                    print >> fp
                    for j in memory[1]:
                        print >> fp, j,
                    print >> fp
                    for j in memory[2]:
                        print >> fp, j,
                    print >> fp
            memory = []


        if ((terminal) or ((episode_length % config.max_episode_length) == 0)):
            num_episodes += 1
            with open("train_reward.txt", "a") as fp:
                print >> fp, (total_reward / (num_episodes * 1.0))    
            env.START_NEW_GAME = True
            
        pbar.update(1)
            
            
        if (brain.timeStep) > config.MAX_FRAMES:
            brain.train_writer.close()
            break

    brain.session.close()

def main():
    config = Config()
 #   config.test()
    config.game_num = sys.argv[1]
    savegame(config)

if __name__ == '__main__':
    main()
