#Math functions using LSTM
#This should be able to almost generalize to understand underlying dynamics
#Because dynamics are just complex (nonlinear) math operations
#Comparison of a Dense network vs LSTM


import random
from random import seed, randint
import numpy as np
from keras import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt
import keras
import csv

import plotLearning

#seed(10)


#Should build samples that are time reliant
#Every example has:
# n_features - representing different features
# n_timesteps - representing each timestep
# n_examples x n_features x n_time
def data_input(n_examples, n_features, n_timesteps, delta_t):
  
  #Create lists
  X = []
  y = []
  
  
  #Length of the timestep
  # Time = n_timesteps*deltaT
  
  state = []
  action = []
  next_state = []
  
  #Import files
  filename = '..\ARM_data1.csv'
  with open(filename, 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
      if row[0] == 'state':
        print(row[0])
        continue
      for i, _ in enumerate(row):
        row[i] = row[i].replace('[','')
        row[i] = row[i].replace(']','')
      
      #Create s,a,s' arrays
      state = state.append(float(row[0]))
      action = action.append(float(row[1]))
      next_state = next_state.append(float(row[2]))
      
  print('Length of input:',len(row))
  
  
  #Initialize values to random numbers
  accel = [0]#[randint(-10,10)]
  vel = [randint(-10,10)]
  pos = [randint(-10,10)]
  time = [0]
  
  
  #Starting from initial value, model forward integration based on random acceleration walk
  for i in range(0,n_timesteps):
    accel.append(accel[i-1] + delta_t*random.uniform(-1,1))
    
    vel.append(vel[i-1] + accel[i]*delta_t)
    pos.append(pos[i-1] + vel[i]*delta_t)
    time.append(time[i-1] + delta_t)
    
  
  #Plot data as a sanity check
  # plt.figure(1)
  # plt.subplot(311)
  # plt.plot(time,accel,'k')
  # plt.xlabel('Time(s)')
  # plt.ylabel('Acceleration(m/s)')
  
  # plt.subplot(312)
  # plt.plot(time,vel,'k')
  # plt.xlabel('Time(s)')
  # plt.ylabel('Velocity(m/s)')
  
  # plt.subplot(313)
  # plt.plot(time,pos,'k')
  # plt.xlabel('Time(s)')
  # plt.ylabel('Position(m/s)')
  
  # plt.show()

  
  x_data = []
  y_data = []
  #Make X an array of positions of size seq_length
  seq_length = 10
  for i in range(n_timesteps - seq_length):
    in_seq1 = accel[i:i + seq_length]
    in_seq2 = pos[i:i + seq_length]

    out_seq = pos[i + seq_length]
    
    x_data.append(in_seq1 + in_seq2)
    y_data.append(out_seq)
  
  n_seqs = len(x_data)
  
  print('Total Sequences:',n_seqs)
  x_data,y_data = np.array(x_data),np.array(y_data)
  #X = np.reshape(x_data,(n_seqs,seq_length,1))
  #y = np.reshape(y_data,(n_seqs,1,1))
  x_data = x_data/1000.0
  y_data = y_data/1000.0

  return x_data,y_data

  #Craft data, 100 samples, y:1x1 is sum of random list X: 2x1
  # for i in range(n_examples):
    # inputData = [randint(1,max) for _ in range(n_numbers)]
    # outputData = sum(inputData)
    # #print(inputData,outputData)
    # X.append(inputData)
    # y.append(outputData)
  
  #Convert to NumPy array
  #X,y = np.array(X), np.array(y)
  
  #Normalize to fit within activation bounds
  #To do this, divide by max possible sum 
  # (max number* n_numbers)
  #X = X.astype('float')/float(max*n_numbers)
  #y = y.astype('float')/float(max*n_numbers)
  
  
  #return X,y
  

#inverts normalization done in randomDataGenerator
def invert(value):
  return (value * float(1000))
  
def buildModel(seq_length):
  
  model = Sequential()
  model.add(LSTM(128, input_shape=(seq_length, 1)))#,return_sequences = True))
  #model.add(LSTM(8))
  model.add(Dense(128))
  model.add(Dense(1))
  model.compile(loss = 'mean_squared_error', optimizer='adam')
  return model

#Setting Variables
n_examples = 100
delta_t = .01
seq_length = 10
n_timesteps = int(n_examples/delta_t )
n_epoch = 100
n_batch = 50


plot_losses = plotLearning.PlotLearning()

model = buildModel(seq_length*2) 
data_input(n_examples,3,10000,.01)
  
#Train
for e in range(n_epoch):
    X,y = data_input(n_examples,3,n_timesteps,delta_t)
    X = X.reshape(n_timesteps - seq_length,seq_length*2,1)
    # X = X.reshape(n_examples*seq_length - s,20,1)
    # print('Shape of new X:',X.shape)
    model.fit(X,y,epochs=1,batch_size=n_batch,verbose=1)
    print('Epoch:',e)
#X,y = randomDataGenerator(1000,3,10000,.01)
#X = X.reshape(n_examples*seq_length - seq_length,seq_length*2,1)
#model.fit(X,y,epochs=n_epoch,batch_size=n_batch,verbose=1)
    
#Evaluation
X,y = randomDataGenerator(1000,3,10000,.01)

#Make it n_examples x n_numbers x 1
#Such as (1000, 2, 1)
X = X.reshape(9990,20,1)


#Predict
result = model.predict(X, batch_size = n_batch, verbose = 0)


#Error calculation
result = invert(result)
expected = invert(y)
print('Result = \n', result)
predicted = result[:,0]

print('Expected: \n',expected)
print('Predicted: \n',predicted)


#expected = [invert(i,n_numbers,max) for i in y]
#predicted = [invert(i, n_numbers, max) for i in result[:,0]]

rmse = math.sqrt(mean_squared_error(expected, predicted))

print('RMSE: %f' % rmse)

#Show:
for i in range(20):
	error = expected[i] - predicted[i]
	print('Expected=%f, Predicted=%f (err=%f)' % (expected[i], predicted[i], error))





# plt.figure(1)
# plt.subplot(211)
# plt.plot(time,accel,'k')
# plt.xlabel('Time(s)')
# plt.ylabel('Acceleration(m/s)')

# plt.subplot(212)
# plt.plot(time,vel,'k')
# plt.xlabel('Time(s)')
# plt.ylabel('Velocity(m/s)')

# plt.show()














