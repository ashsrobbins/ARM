#Testing ArmEnv!

from armEnv import armEnv

env = armEnv()
env.make()

env.reset()

while(1):
  userIn = float(input('Enter a motor speed [-1, 1]:'))
  env.doAction(userIn)
  