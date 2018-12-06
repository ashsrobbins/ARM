# dataGather.py
# Ash Robbins
# 5/26/2018
# With the ARM, move through full range of motion with various motor controls
# gathering data for the full state space, saving as a CSV file
# to later train an LSTM that will understand the underlying dynamics

import sys
import gym
# import pylab
import random
import csv
import time 

import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from numpy import random



from armEnvPIDShoulder import armEnv






if __name__ == "__main__":
  print("Beginning setup...")
  env = armEnv()
  env = env.make()
  # state = env.reset()
  state, reward, done, info  = env.step([0,0])
  env.saveData('test.csv')
  
  
  action = 0
  #PID
  error_prior = 0
  integral = 0
  kp = .1
  ki = .02
  kd = 0
  bias = 0
  
  goal = 45
 
  action1 = 0
  prev_action1 = 0
  
  min = -10
  max = 45
  
  
  #2nd
  error_prior_y = 0
  integral_y = 0
  kp_y = .025
  ki_y = .0015
  kd_y = 0
  bias_y = 0
  
  goal_y = 45
  
  action2 = 0
  prev_action2 = 0
  
  min_y = -60
  max_y = -14
  
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
    action1 = output  
    # print('Action=',action)
    if action1 > 1:
      action1 = 1
    if action1 < -1:
      action1 = -1
      
   
      
    if int(state[0]) > max:
     action1 = -.2
    if state[0] < min:
     action1 = .2
     
    if error < 2 and error > -2:
      action1 = 0

    
    if not isinstance(action1, np.ndarray):
      action1 = np.array(action1)
      
    
    # print("Cur:", state[0],'\tGoal',goal)
    
    
    
    #Do action 2
    
    error_y = goal_y - state[2] 
    
    
    integral_y = integral_y + (error_y*.1)#.1 is iteration time
    derivative_y = (error_y - error_prior_y)/.1
    output_y = kp_y*error_y + ki_y*integral_y + kd_y*derivative_y + bias_y
    
    error_prior_y = error_y
    
    # if random.randint(1,5) == 1:
      # action = prev_action + (4*(random.rand(1)-.5))**3
    action2 = -output_y  
    # print('Action=',action)
    if action2 > 1:
      action2 = 1
    if action2 < -1:
      action2 = -1
      
   
    
    if int(state[2]) > max_y:
     action2 = .2
     print('AT MAX')
    if state[2] < min_y:
     print('AT MIN')
     action2 = -.2
     
    if error_y < 1 and error_y > -1:
      action2 = 0
    
    
    #For testing
    # action2 = 0
    
    
    #Merge actions
    action = np.array([action1,action2])
    
    next_state, reward, done, info = env.step(action)
    
    if env.steps %25 == 0:
      print('**********************************')
      print('Actions =\t',action)
      print('Pitch Pos =\t',state[0])
      print('Pitch Goal =\t',state[1])
      print('Roll Pos =\t',state[2])
      print('Roll Goal =\t',state[3])
    
    # print('State', state[0])
    
    
    # with open(filename,'a', newline = '') as File:  
      # writer = csv.writer(File)
      
      
      # writer.writerow([state[0]] + [action] + [next_state[0]])
      
    state = next_state
    goal = state[1]
    goal_y= state[3]
      