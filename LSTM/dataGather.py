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
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential

from armEnv import armEnv






if __name__ == "__main__":
  env = armEnv()
  env = env.make()