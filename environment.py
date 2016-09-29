# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 15:05:34 2016

@author: ghulamahmedansari
"""

import numpy as np
from collections import deque
import zmq
import sys

class Environment(object):
    def __init__(self, config):
        self.START_NEW_GAME = True
        self._env = gym.make(config.GAME)
        self.CURR_REWARD = 0 
        self.config = config
        #for best network saving
        self.reward_history = deque()
        # ZeroMQ Context
        context = zmq.Context()
        # Define the socket using the "Context"
        sock = context.socket(zmq.REQ)

        
    def step(self,action):
        socket.send('step_game')
        self._screen_, self.reward, self.terminal, _ = 
               
    def action_size(self):
        return self._env.action_space.n        
            
    def act(self,action):
        Reward = 0
        start_lives = self._env.ale.lives()
        for _ in xrange(self.config.K):
            prevframe=self._screen_.copy()
            self.render()
            self.step(action)
            observation,localreward,terminal = self._screen_,self.reward,self.terminal
            Reward += localreward
            if start_lives > self._env.ale.lives():
                # Reward -= 1.0 #@code_walk: check this
# Not terminating on losing life                
                terminal = True
                
            if terminal:
                break
        indices = np.where(observation>prevframe)
        prevframe[indices] = observation[indices]
        self._screen_ = prevframe

        self.preprocess()
        
        self.CURR_REWARD = Reward
        
        change_reward = 0
        if self.config.clipR:
            change_reward = np.clip(Reward,-1,1)        
    
        if terminal:
            self.START_NEW_GAME = True
        return self._screen, action, change_reward, terminal

class Eval_Environment(Environment):
    def __init__(self, config):
        super(Eval_Environment, self).__init__(config)
        
    def act(self,action):
        Reward = 0
        for _ in xrange(self.config.K):
            prevframe=self._screen_.copy()
            self.render()
            self.step(action)
            observation,localreward,terminal = self._screen_,self.reward,self.terminal
            Reward += localreward
            if terminal:
                break
        indices = np.where(observation>prevframe)
        prevframe[indices] = observation[indices]
        self._screen_ = prevframe
        
        self.CURR_REWARD = Reward
    
        self.preprocess()
        
        change_reward = 0
        if self.config.clipR:
            change_reward = np.clip(Reward,-1,1)
        if terminal:
            self.START_NEW_GAME = True
        self.close_render()
        return self._screen, action, change_reward, terminal