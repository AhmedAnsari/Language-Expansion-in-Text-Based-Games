# -------------------------
# Project: DQN Nature implementation
# Author: ghulamahmedansari,Rakesh Menon
# -------------------------

import os
from models.DQN import DQN
import numpy as np
import cPickle as cpickle
from models.config import Config
from tqdm import tqdm
import random
from games import HomeGame, FantasyGame

def playgame(config,game):
    # Step 1: init Game
    ###################
    # Step 2: init DQN
    actions = len(game.actions)
    objects = len(game.objects)
    config.setnumactions(actions)
    config.setnumobjects(objects)

    brain = DQN(config)

    # checkStates = None
    #adding progress bar for training
    pbar = tqdm(total = config.MAX_FRAMES, desc='Training Progress')
    episode_length = 0
    num_episodes = 0
    total_reward = 0
    while True:
        if game.START_NEW_GAME:
            episode_length = 0
            game.START_NEW_GAME = False
            state, reward, terminal, _ = game.new_game()
            brain.history.add(state)
        action_indicator = np.zeros(actions)
        object_indicator = np.zeros(objects)
        #predict
        action_index,object_index = brain.getAction()
        action_indicator[action_index] = 1
        object_indicator[object_index] = 1
        #act
        nextstate,reward,terminal,percentage = game.do(action_index,object_index)
        total_reward += reward
        episode_length += 1
        #observe
        brain.setPerception(state, reward, action_indicator, object_indicator, nextstate, terminal, False)
        nextstate = state
#####################################################################
        if ((terminal) or ((episode_length % config.max_episode_length) == 0)):
            with open("quest.txt", "a") as fp:
                print >> fp, percentage
            num_episodes += 1
            with open("reward.txt", "a") as fp:
                print >> fp, (total_reward / (num_episodes * 1.0))    
            game.START_NEW_GAME = True


        # if (brain.timeStep%1000)==0:
            # pbar.update(1000)
        pbar.update(1)
            
#        if (brain.timeStep%100000)==0:
#            brain.memory.save_memory()
            
        if (brain.timeStep) > config.MAX_FRAMES:
            break

    brain.session.close()

def main():
    config = Config()
 #   config.test()
    game = HomeGame(game_dir=config.game_dir, seq_length=config.seq_length)
    playgame(config,game)

if __name__ == '__main__':
    main()
