# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 17:09:18 2016

@author: ghulamahmedansari
"""

import random
import numpy as np
import cPickle as pickle
from collections import deque

class ReplayMemory:
  def __init__(self, config):
    self.batch_size = config.batch_size
    self.memory_size = config.memory_size
    self.count = 0
    self.model_dir = config.model_dir
    self.memory = deque()
    self.priority_memory = deque()
    self.len_mem = 0
    self.len_prioritized_mem = 0

  def add(self, state, reward, nextstate, action, obj, terminal):

    if self.len_mem + self.len_prioritized_mem >= memory_size:
      pop1 = self.priority_memory[0][6]
      pop2 = self.memory[0][6]

      if pop1 < pop2:
        self.priority_memory.popleft()
        self.len_prioritized_mem -= 1
      else:
        self.memory.popleft()
        self.len_mem -= 1


    if reward > 0:
      self.priority_memory.append((state, action, obj, reward, nextstate, terminal, self.count))
      self.len_prioritized_mem += 1
    else:
      self.memory.append((state, action, obj, reward, nextstate, terminal, self.count))
      self.len_mem += 1

    self.count += 1

  def sample(self):
    n_sampled = 0
    if self.len_prioritized_mem >= self.batch_size/4:
      batch = random.sample(self.priority_memory,self.batch_size/4)
      n_sampled = self.batch_size/4
    else:
      batch = random.sample(self.priority_memory,self.len_prioritized_mem)
      n_sampled = int(self.len_prioritized_mem)
    batch.extend(random.sample(self.memory, self.batch_size - n_sampled))
    return batch 
    
  def save_memory(self):
      fp = open(self.model_dir+'/replay_file.save','wb')
      pickle.dump(self,fp,protocol=pickle.HIGHEST_PROTOCOL)
      fp.close()
