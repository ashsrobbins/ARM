#Math functions using LSTM
#This should be able to almost generalize to understand underlying dynamics
#Because dynamics are just complex (nonlinear) math operations

from random import seed, randint
import numpy as np
from keras import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
import math

#seed(10)



def randomDataGenerator(n_examples, n_numbers, max):
  
  #Create lists
  X = []
  y = []

  #Craft data, 100 samples, y:1x1 is sum of random list X: 2x1
  for i in range(n_examples):
    inputData = [randint(1,max) for _ in range(n_numbers)]
    outputData = sum(inputData)
    #print(inputData,outputData)
    X.append(inputData)
    y.append(outputData)
  
  #Convert to NumPy array
  X,y = np.array(X), np.array(y)
  
  #Normalize to fit within activation bounds
  #To do this, divide by max possible sum 
  # (max number* n_numbers)
  X = X.astype('float')/float(max*n_numbers)
  y = y.astype('float')/float(max*n_numbers)
  return X,y
  

#inverts normalization done in randomDataGenerator
def invert(value, n_numbers, max):
  return round(value * float(max* n_numbers))
  
def buildModel(n_numbers):
  model = Sequential()
  model.add(LSTM(6, input_shape=(n_numbers, 1), return_sequences=True))
  model.add(LSTM(6))
  model.add(Dense(1))
  model.compile(loss = 'mean_squared_error', optimizer='adam')
  return model

#Setting Variables
n_examples = 1000
n_numbers = 10
n_epoch = 1000
n_batch = 40
max = 100



model = buildModel(n_numbers) 

  
#Train
for e in range(n_epoch):
    X,y = randomDataGenerator(n_examples, n_numbers, max)
    print('Shape of X:',X.shape)
    X = X.reshape(n_examples,n_numbers, 1)
    print('Shape of new X:',X.shape)
    model.fit(X,y,epochs=1,batch_size=n_batch,verbose=2)
    print('Epoch:',e)


    
#Evaluation
X,y = randomDataGenerator(n_examples, n_numbers, max)


#Make it n_examples x n_numbers x 1
#Such as (1000, 2, 1)
X = X.reshape(n_examples, n_numbers, 1)


#Predict
result = model.predict(X, batch_size = n_batch, verbose = 0)


#Error calculation
expected = [invert(i,n_numbers,max) for i in y]
predicted = [invert(i, n_numbers, max) for i in result[:,0]]

rmse = math.sqrt(mean_squared_error(expected, predicted))

print('RMSE: %f' % rmse)

#Show:
for i in range(20):
	error = expected[i] - predicted[i]
	print('Expected=%d, Predicted=%d (err=%d)' % (expected[i], predicted[i], error))




















