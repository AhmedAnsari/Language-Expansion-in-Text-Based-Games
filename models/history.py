# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 17:09:18 2016

@author: ghulamahmedansari
"""

from copy import deepcopy
class History:
  def __init__(self):
    self.history = None

  def add(self, state):
    self.history = deepcopy(state)

  def get(self):
    return self.history

  def copy(self):
    return deepcopy(self.history)
      
