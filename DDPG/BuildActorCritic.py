
#Actor network to decide action based on the current state
def buildActorNetwork(self, state_size,action_size):
  state = Input(shape=[state_size])
  h0 = Dense(H0_UNITS, activation = 'relu')(state)
  h1 = Dense(H1_UNITS, activation = 'relu')(h0)
  
  
  #Give the output one output with tanh scaling(-1,1)
  motors = Dense(action_size,activation='tanh',init=lambda shape, name: normal(shape, scale = 1e-4, name = name))(h1)
  
  model = Model(input = state,output=motors)
  
  print('Actor Created')
  return model,model.trainable_weights,state
  
  
  
#Critic network to determine value from state and action
#--this is used to train the actor
def buildCriticNetwork(self,state_size,action_size):
  state = Input(shape=[state_size])
  action = Input(shape = [action_size], name = 'action2')
  
  
  
  w1 = Dense(H0_UNITS, activation = 'relu')(state)
  h1 = Dense(H1_UNITS, activation = 'relu')(w1)
  
  a1 = Dense(H1_UNITS, activation = 'relu')(action)
  
  h2 = merge([h1, h2], mode = 'sum')
  h3 = Dense(H1_UNITS, activation = 'relu')(h2)
  
  value = Dense(action_size, activation='linear')(h3)
  
  model = Model(input =[state, action],output = value)
  
  adam = Adam(lr = self.learning_rate)
  
  model.compile(loss = 'mse',optimizer = adam)
  
  print('Critic Created')
  
  return model,action,state
  
  
#Create the target network
def target_train(self):
  actor_weights = self.model.get_weights()
  actor_target_weights = self.target_model.get_weights()
  for i in xrange(len(actor_weights)):
    actor_target_weights[i] = self.TAU*actor_weights[i] + (1 - self.TAU)*actor_target_weights[i]
  self.target_model.set_weights(actor_target_weights)
  
  





  