# dataGather.py
# Ash Robbins
# 5/26/2018
# With the ARM, move through full range of motion with various motor controls
# gathering data for the full state space, saving as a CSV file
# to later train an LSTM that will understand the underlying dynamics

import sys
import gym
import pylab
import random
import csv
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from numpy import random

from armEnvDataGather import armEnv


filename = 'ARM_data.csv'



if __name__ == "__main__":
  print("Beginning setup...")
  env = armEnv()
  env = env.make()
  state = env.reset()
  
  action = 0
  prev_action = 0
  
  min = -5
  max = 50
  print("Setup complete!\n")
  
  #Create file
  with open(filename,'w', newline = '') as File:  
    writer = csv.writer(File)
    writer.writerow(['state','action','next_state'])
    
    
  #Shape is returned as (X,)
  state_size = env.observation_space.shape[0]
  print('Observation Shape: ',state_size)
  
  action_size = env.action_space
  print('Action Shape: ', action_size)  
  
  
  print('\n\nBegin Data Gathering:\n')
  
  while(True):
    #Do an environment step
    if random.randint(1,5) == 1:
      action = prev_action + (4*(random.rand(1)-.5))**3
    if action > 1:
      action = 1
    if action < -1:
      action = -1
      
   
      
    if int(state[0]) > max:
     action = 1
    if state[0] < min:
     action = -1
     
    if not isinstance(action, np.ndarray):
      action = np.array(action)
      
    
    print("State, Action\t", state[0],'\t',action)
    
      
    
    next_state, reward, done, info = env.step(action)
    # print('State', state[0])
    
    
    with open(filename,'a', newline = '') as File:  
      writer = csv.writer(File)
      
      
      writer.writerow([state[0]] + [action] + [next_state[0]])
      
    state = next_state
      