#armBrain



import serial
import QLearning
import random

goal = (0,0,-10)
arm = QLearning.armAgent(goal)
reward = 0

ser = serial.Serial('COM4', 57600)
kp = 4
set_point = -30
step = 0
rewardWait = 0


while True:
    step += 1
    raw = ser.readline()
 
    raw = ser.readline()
    raw = raw.strip()
    dev_id,x,y,z = raw.split(',')
    x = float(x)
    y = float(y)
    z = float(z)
    
    if z <= (goal[2]+.5)  and z>= (goal[2]-.5):
      
      rewardWait += 1
      print 'I am waiting'
      if rewardWait >= 20:
        rewardWait = 0
        arm.setGoal(goal)
		#break
    else:
      reward = 0
    action = arm.chooseAction((x,y,z), reward)

    
    ser.write(action + '\n')
    
    
    
    
    #error = set_point - z
    #output = error * kp
    #print z,output
    #ser.write(str(output) + '\n')