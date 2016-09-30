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
# from games import HomeGame, FantasyGame
from environment import Environment

def playgame(config):
    # Step 1: init Game
    env = Environment()
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
    while True:
        if env.START_NEW_GAME:
            episode_length = 0
            env.START_NEW_GAME = False
            state, reward, terminal, availableObjects = env.newGame()
            brain.history.add(state)
        action_indicator = np.zeros(actions)
        object_indicator = np.zeros(objects)
        #predict
        action_index,object_index = brain.getAction(availableObjects)
        action_indicator[action_index] = 1
        object_indicator[object_index] = 1
        #act
        nextstate,reward,terminal, availableObjects = env.step(action_index,object_index)
        total_reward += reward
        episode_length += 1
        #observe
        brain.setPerception(state, reward, action_indicator, object_indicator, nextstate, terminal, False)
        nextstate = state
#####################################################################
        if ((terminal) or ((episode_length % config.max_episode_length) == 0)):
            num_episodes += 1
            with open("train_reward.txt", "a") as fp:
                print >> fp, (total_reward / (num_episodes * 1.0))    
            env.START_NEW_GAME = True

#####################################################################
        #for evaluating qvalues
        if (brain.timeStep % config.EVAL == 0) and (brain.timeStep != 0):
            if (brain.timeStep / config.EVAL == 1):
                if not ((os.path.exists("checkStates.txt")) and (os.path.getsize("checkStates.txt") > 0)):                    
                    assert config.SAMPLE_STATES % config.BATCH_SIZE == 0 
                    assert config.SAMPLE_STATES < brain.memory.count
                    checkStates, _1, _2, _3, _4, _5 = brain.memory.sample()
                    with open("checkStates.txt", "w") as fp:
                        cpickle.dump(checkStates,fp)
                else:
                    with open("checkStates.txt", 'r') as fp:
                        checkStates = cpickle.load(fp)
####################################################################
            evalQValues_a = brain.action_valueT.eval(feed_dict={brain.stateInputT:checkStates},session = brain.session)
            maxEvalQValues_a = np.max(evalQValues_a, axis = 1)
            avgEvalQValues_a = np.mean(maxEvalQValues_a)

            with open("evalQValue_a.txt", "a") as fp:
                print >>fp,avgEvalQValues_a

            evalQValues_o = brain.object_valueT.eval(feed_dict={brain.stateInputT:checkStates},session = brain.session)
            maxEvalQValues_o = np.max(evalQValues_o, axis = 1)
            avgEvalQValues_o = np.mean(maxEvalQValues_o)

            with open("evalQValue_o.txt", "a") as fp:
                print >>fp,avgEvalQValues_o
#####################################################################
            brain.inject_summary({
                'average.q_a': avgEvalQValues_a,
                'average.q_o': avgEvalQValues_o,
              }, brain.timeStep)
#####################################################################

        # if (brain.timeStep%1000)==0:
            # pbar.update(1000)
        pbar.update(1)
            
#        if (brain.timeStep%100000)==0:
#            brain.memory.save_memory()
            
        if (brain.timeStep) > config.MAX_FRAMES:
            brain.train_writer.close()
            break

    brain.session.close()

def main():
    config = Config()
 #   config.test()
    # game = HomeGame(game_dir=config.game_dir, seq_length=config.seq_length)
    # playgame(config,game)
    playgame(config)

if __name__ == '__main__':
    main()
