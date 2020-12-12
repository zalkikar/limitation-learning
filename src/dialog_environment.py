"""
OpenAI Gym inspired environment for this NLP task. 

Upon resetting environment, returns the state and expert action in raw and embedding form. 

In our case conversations are only pairs, although this is a scalable approach, and as a
starting point for that we include a .step(action) function which simply returns done=True. This also allows us
to make our framework as similar as possible to previously successful approaches using GAIL. 

"""


import torch


import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
import numpy as np
from io import open
import itertools
import math
import matplotlib.pyplot as plt

class DialogEnvironment(object):
    """
    
    Gym environment for dialog.
    
    """
    def __init__(self, mode='train'):
        
        # TODO: fix path - fixed?
        self.conversations = torch.load('/scratch/nsk367/deepRL/limitation-learning/apps/dat/preprocess/padded_vectorized_states.pt')
        self.raw_conversations = torch.load('/scratch/nsk367/deepRL/limitation-learning/apps/dat/preprocess/raw_states.pt')
        
        
        self.conversations_visited = []
        
    def clear(self):
        self.conversations_visited = [] #
    def current_state(self):
        return i  # i for current conversation index, j for current word (these should be odd? )
    
    def reset(self):
        """
        Start a new trajectory, aka a new conversation. Environment does this by 
        picking a random i in the length of the total conversations. 

        Using random with replacement, so it is possible to revisit environments.

        I will leave this as a TODO in case without replacement is preferred. 
        """
        while True: #some indices have been removed, for various resions. This while loop 
                # allows us to keep trying until a viable conversation is selected. 
            try:
                self.i = random.randint(a=0,b=64)#b=len(self.conversations)) # test for over-fitting. 
                self.conversations_visited.append(self.i)
                self.conversation = self.conversations[self.i]
                break
            except:
                pass


        state = self.conversation[0]
        expert_action = self.conversation[0]
        
        raw_state = list(self.raw_conversations.keys())[self.i], 
        
        raw_expert_action = self.raw_conversations[list(self.raw_conversations.keys())[self.i]]
        #TODO: truncate sequences? 
        return state, expert_action, raw_state, raw_expert_action
    
    def step(self,action):
        done = True

        return done

