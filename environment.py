# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 15:05:34 2016

@author: ghulamahmedansari
"""

import zmq
import numpy as np
from collections import deque
import zmq
import sys

class Environment(object):
    def __init__(self, config):
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect ("tcp://localhost:12345")
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

    
    def getScrRewTer(self, msg):
        msgsplit = msg.split("#")
        self._screen_ = [float(i) for i in msgsplit[0].split(" ")]
        self.reward = float(msgsplit[1])
        self.terminal = float(msgsplit[2])
        return self._screen_, self.reward, self.terminal

    def interact(self, str):
        self.socket.send(str)
        msg = self.socket.recv()
        return msg


    def newGame(self):
        str = 'newGame'
        msg = self.interact(str)
        self._screen_, self.reward, self.terminal = getScrRewTer(msg)
        return self._screen_, self.reward, self.terminal

    def step(self,action):
        str = 'step_game'
        msg = self.interact(str)
        self._screen_, self.reward, self.terminal = getScrRewTer(msg)
        return self._screen_, self.reward, self.terminal
        # self._screen_, self.reward, self.terminal, _ = self._env.step(action)         
               
    def action_size(self):
        str = 'getActions'
        msg = self.interact(str)
        return int(msg)
            
    def object_size(self):
        str = 'getObjects'
        msg = self.interact(str)
        return int(msg)

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