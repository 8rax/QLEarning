import gym
import numpy as np

#https://pythonprogramming.net/q-learning-reinforcement-learning-python-tutorial/
#pip install gym
#The goal is to make the charriot go to the small flag on the right, when the chariot is going left you have to press left, when the chariot is going right, you have to press right :) 
# We are going to build a big table with position, velocity and pick up the largest Qvalue to get the right instruction (QLearning) 

#
#					0		1		2
# Combination1      Q0      Q1      Q2
# Combination2      Q0		Q1		Q2
# Combination3      ..		..		..
#
#
#
#

env=gym.make("MountainCar-v0")
env.reset()

#print(env.observation_space.high)
#print(env.observation_space.low)

#For the value at index 0, we can see the high value is 0.6, the low is -1.2, and then for the value at index 1, the high is 0.07, 
#and the low is -0.07. Okay, so these are the ranges, but from one of the above observation states that we output: [-0.27508804 -0.00268013], 
#we can see that these numbers can become quite granular. Can you imagine the size of a Q Table if we were going to have a value for every combination of 
#these ranges out to 8 decimal places? That'd be huge! And, more importantly, it'd be useless. We don't need that much granularity. So, instead, what we want 
#to do is conver these continuous values to discrete values. Basically, we want to bucket/group the ranges into something more manageable.


#Number of actions: 
#print(env.action_space.n)

DISCRETE_OS_SIZE=[20]*len(env.observation_space.high)
discrete_os_win_size=(env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

#20*20*3 table :)
#So, this is a 20x20x3 shape, which has initialized random Q values for us. 
#The 20 x 20 bit is every combination of the bucket slices of all possible states. The x3 bit is for every possible action we could take.

#We have put random values into the table between -2 and 0 
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))
print(q_table.shape)

'''
done=False

#List of actions
 #0 = push left
 #1 = do nothing
 #2 = push right

while not done:
	action = 2
	new_state, reward, done, _ = env.step(action)
	print(reward, new_state)
	env.render()



env.close()
'''
