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


EPISODES = 1000


# DQN Agent for the Cartpole
# it uses Neural Network to approximate q function
# and replay memory & target q network
class DQNAgent:
    def __init__(self, state_size, action_size):
        
        self.load_model = True

        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # These are hyper parameters for the DQN
        self.discount_factor = 0.95# was .99
        self.learning_rate = 0.002#was .001
        self.epsilon = 1.0#was 1.0
        self.epsilon_decay = 0.999#was .999
        self.epsilon_min = 0.01#was .01
        self.batch_size = 100
        self.train_start = 1000
        # create replay memory using deque
        self.memory = deque(maxlen=2000)

        # create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()

        # initialize target model
        self.update_target_model()

        if self.load_model:
            self.model.load_weights("./arm_dqn.h5")

    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):
        model = Sequential()
        model.add(Dense(16, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(16, activation='relu',
                        kernel_initializer='he_uniform'))      
        model.add(Dense(self.action_size, activation='linear',
                        kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # get action from model using epsilon-greedy policy
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])

    # save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # pick samples randomly from replay memory (with batch_size)
    def train_model(self):
        if len(self.memory) < self.train_start:
            return
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        update_input = np.zeros((batch_size, self.state_size))
        update_target = np.zeros((batch_size, self.state_size))
        action, reward, done = [], [], []

        for i in range(self.batch_size):
            update_input[i] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target[i] = mini_batch[i][3]
            done.append(mini_batch[i][4])

        target = self.model.predict(update_input)
        target_val = self.target_model.predict(update_target)

        for i in range(self.batch_size):
            # Q Learning: get maximum Q value at s' from target model
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.discount_factor * (
                    np.amax(target_val[i]))

        # and do the model fit!
        self.model.fit(update_input, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)


if __name__ == "__main__":
    
  # get size of state and action from environment
  env = armEnv()
  env = env.make()
  
  
  #Shape is returned as (X,)
  state_size = env.observation_space.shape[0]
  #Action_space.n is returned as just an int with the number of actions
  action_size = env.action_space

  #print(state_size, action_size)
  agent = DQNAgent(state_size, action_size)

  scores, episodes = [], []

  for e in range(EPISODES):
      done = False
      score = 0
      
    #Returns a length of 4 value, I believe NP ndarray(representative of the state)
    #This is where I should make the arm get back to the correct spot, could be blocking code!!
      state = env.reset()
  
  #I believe this converts it into a list of a list
      state = np.reshape(state, [1, state_size])

      while not done:
        

        # get action for the current state and go one step in environment
        action = agent.get_action(state)
        #print('Action chosen:', action)
        next_state, reward, done, info = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        # if an action make the episode end, then gives penalty of -100
        reward = reward

        # save the sample <s, a, r, s'> to the replay memory
        agent.append_sample(state, action, reward, next_state, done)
        # every time step do the training
        agent.train_model()
        score += reward
        state = next_state

        if done:
        
          #
          
          
          # every episode update the target model to be same with model
          agent.update_target_model()
          
          # every episode, plot the play time
          score = score# if score == 500 else score + 100
          scores.append(score)
          episodes.append(e)
          pylab.plot(episodes, scores, 'b')
          pylab.savefig("./arm_dqn.png")
          print("episode:", e, "  score:", score, "  memory length:",
                      len(agent.memory), "  epsilon:", agent.epsilon)
          # if the mean of scores of last 10 episode is bigger than 490
          # stop training
          if np.mean(scores[-min(10, len(scores)):]) > 3000:
            sys.exit()

      # save the model
      if e % 5 == 0:
        agent.model.save_weights("./arm_dqn.h5")