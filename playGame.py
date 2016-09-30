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

def evaluate(brain)

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
            #save current history before starting evaluation
            temp_history_data =brain.history.history.copy()  
            #now let us evaluate avg reward                        
            #create alternate environment for EVALUATION
            env_eval = Environment()

            reward,num_of_games = evaluate(brain, env_eval, config)
            with open("reward.txt", "a") as fp:
                print >> fp, reward

            #setting the best network        
            if len(env_eval.reward_history)==0 or reward > max(env_eval.reward_history):
                # save best network
                if not os.path.exists(os.getcwd()+'/Savednetworks'):
                    os.makedirs(os.getcwd()+'/Savednetworks')
                brain.saver.save(brain.session, os.getcwd()+'/Savednetworks/'+'network' + '-dqn', global_step = brain.timeStep)                

            env_eval.reward_history.append(reward) #doing this for keeping track of best network    
            
            #go back to saved frame after evaluation completed
            brain.history.history = temp_history_data

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
#####################################################################

    #--Testing
    if brain.timeStep % config.eval_freq == 0 and brain.timeStep > config.learn_start:
        print('Testing Starts ... ')
        quest3_reward_cnt = 0
        quest2_reward_cnt = 0
        quest1_reward_cnt = 0
        # test_avg_Q = test_avg_Q or optim.Logger(paths.concat(opt.exp_folder , 'test_avgQ.log'))
        # test_avg_R = test_avg_R or optim.Logger(paths.concat(opt.exp_folder , 'test_avgR.log'))
        # test_quest1 = test_quest1 or optim.Logger(paths.concat(opt.exp_folder , 'test_quest1.log'))
        test_avg_Q = 0
        test_avg_R = 0
        test_quest1 = 0
        if TUTORIAL_WORLD:
            # test_quest2 = test_quest2 or optim.Logger(paths.concat(opt.exp_folder , 'test_quest2.log'))
            # test_quest3 = test_quest3 or optim.Logger(paths.concat(opt.exp_folder , 'test_quest3.log'))
            test_quest2 = 0
            test_quest3 = 0

        # gameLogger = gameLogger or io.open(paths.concat(opt.exp_folder, 'game.log'), 'w')
        gameLogger = None

        state, reward, terminal, available_objects = env.newGame()
        brain.history.add(state)

        total_reward = 0
        nrewards = 0
        nepisodes = 0
        episode_reward = 0

        for estep in range(1,opt.eval_steps+1):
            #@TODO:add progress bar here

            action_indicator = np.zeros(actions)
            object_indicator = np.zeros(objects)
            #predict
            action_index,object_index = brain.getAction(availableObjects, True)
            action_indicator[action_index] = 1
            object_indicator[object_index] = 1

            
            ##-- Play game in test mode (episodes don't end when losing a life)
            nextstate,reward,terminal, availableObjects = env.step(action_index,object_index)

            #observe
            brain.setPerception(state, reward, action_indicator, object_indicator, nextstate, terminal, True)
            nextstate = state            


            if TUTORIAL_WORLD:
                if(reward > 9):
                    quest1_reward_cnt =quest1_reward_cnt+1

                elif reward > 0.9:
                    quest2_reward_cnt = quest2_reward_cnt + 1
                elif reward > 0:
                    quest3_reward_cnt = quest3_reward_cnt + 1 #--defeat guardian
            else
                if(reward > 0.9):
                    quest1_reward_cnt =quest1_reward_cnt+1

            #-- record every reward
            episode_reward = episode_reward + reward

            if reward != 0:
               nrewards = nrewards + 1

            if terminal:
                total_reward = total_reward + episode_reward
                episode_reward = 0
                nepisodes = nepisodes + 1
                state, reward, terminal, available_objects = env.newGame()

        local ind = #reward_history+1
        total_reward = total_reward/math.max(1, nepisodes)

        if #reward_history == 0 or total_reward > torch.Tensor(reward_history):max() then
            agent.best_network = agent.network:clone()
        end

        if agent.v_avg then
            v_history[ind] = agent.v_avg
            td_history[ind] = agent.tderr_avg
            qmax_history[ind] = agent.q_max
        end
        print("V", v_history[ind], "TD error", td_history[ind], "V avg:", v_history[ind])

        --saving and plotting
        test_avg_R:add{['% Average Reward'] = total_reward}
        test_avg_Q:add{['% Average Q'] = agent.v_avg}
        test_quest1:add{['% Quest 1'] = quest1_reward_cnt/nepisodes}
        if TUTORIAL_WORLD then
            test_quest2:add{['% Quest 2'] = quest2_reward_cnt/nepisodes}
            test_quest3:add{['% Quest 3'] = quest3_reward_cnt/nepisodes}
        end

        test_avg_R:style{['% Average Reward'] = '-'}; test_avg_R:plot()
        test_avg_Q:style{['% Average Q'] = '-'}; test_avg_Q:plot()
        test_quest1:style{['% Quest 1'] = '-'}; test_quest1:plot()
        if TUTORIAL_WORLD then
            test_quest2:style{['% Quest 2'] = '-'}; test_quest2:plot()
            test_quest3:style{['% Quest 3'] = '-'}; test_quest3:plot()
        end


        reward_history[ind] = total_reward
        reward_counts[ind] = nrewards
        episode_counts[ind] = nepisodes

        time_history[ind+1] = sys.clock() - start_time

        local time_dif = time_history[ind+1] - time_history[ind]

        local training_rate = opt.actrep*opt.eval_freq/time_dif

        print(string.format(
            '\nSteps: %d (frames: %d), reward: %.2f, epsilon: %.2f, lr: %G, ' ..
            'training time: %ds, training rate: %dfps, testing time: %ds, ' ..
            'testing rate: %dfps,  num. ep.: %d,  num. rewards: %d, completion rate: %.2f',
            step, step*opt.actrep, total_reward, agent.ep, agent.lr, time_dif,
            training_rate, eval_time, opt.actrep*opt.eval_steps/eval_time,
            nepisodes, nrewards, pos_reward_cnt/nepisodes))


        pos_reward_cnt = 0
        quest1_reward_cnt = 0
        gameLogger:write("###############\n\n") --end of testing epoch
        print('Testing Ends ... ')
        collectgarbage()
    end
#####################################################################
def main():
    config = Config()
 #   config.test()
    # game = HomeGame(game_dir=config.game_dir, seq_length=config.seq_length)
    # playgame(config,game)
    playgame(config)

if __name__ == '__main__':
    main()
