# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 17:09:18 2016

@author: ghulamahmedansari
"""

import numpy as np
from copy import deepcopy
class History:
  def __init__(self, config):
    self.sequence_length = config.seq_length
    self.history = np.zeros(
        [self.sequence_length], dtype=np.int32)

  def add(self, state):
    self.history = deepcopy(state)

  def reset(self):
    self.history *= 0

  def get(self):
    return self.history
      
