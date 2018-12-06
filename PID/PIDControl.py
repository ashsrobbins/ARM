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
import time 

import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from numpy import random



from armEnvPID import armEnv






if __name__ == "__main__":
  print("Beginning setup...")
  env = armEnv()
  env = env.make()
  state = env.reset()
  
  
  #PID
  error_prior = 0
  integral = 0
  kp = .6
  ki = 0
  kd = .005
  bias = 0
  
  goal = 45
  
  action = 0
  prev_action = 0
  
  min = -5
  max = 50
  print("Setup complete!\n")
  
  #Create file
  # with open(filename,'w', newline = '') as File:  
    # writer = csv.writer(File)
    # writer.writerow(['state','action','next_state'])
    
    
  #Shape is returned as (X,)
  state_size = env.observation_space.shape[0]
  print('Observation Shape: ',state_size)
  
  action_size = env.action_space
  print('Action Shape: ', action_size)  
  
  
  print('\n\nBegin Data Gathering:\n')
  
  t = time.time()
  
  while(True):
    #Do an environment step
    error = goal - state[0] 
    
    
    t = time.time()
    integral = integral + (error*.1)#.1 is iteration time
    derivative = (error - error_prior)/.1
    output = kp*error + ki*integral + kd*derivative + bias
    
    error_prior = error
    
    # if random.randint(1,5) == 1:
      # action = prev_action + (4*(random.rand(1)-.5))**3
    action = -output  
    # print('Action=',action)
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
      
    
    # print("Cur:", state[0],'\tGoal',goal)
    
      
    
    next_state, reward, done, info = env.step(action)
    # print('State', state[0])
    
    
    # with open(filename,'a', newline = '') as File:  
      # writer = csv.writer(File)
      
      
      # writer.writerow([state[0]] + [action] + [next_state[0]])
      
    state = next_state
    goal = state[1]
      