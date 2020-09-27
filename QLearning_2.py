import gym
import numpy as np

#https://pythonprogramming.net/q-learning-algorithm-reinforcement-learning-python-tutorial/
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

LEARNING_RATE=0.1
#The DISCOUNT is a measure of how much we want to care about FUTURE reward rather than immediate reward. Typically, this value will be fairly high, 
#and is between 0 and 1. We want it high because the purpose of Q Learning is indeed to learn a chain of events that ends with a positive outcome, 
#so it's only natural that we put greater importance on long terms gains rather than short term ones.

#The LEARNING_RATE is between 0 and 1, same for discount. The EPISODES is how many iterations of the game we'd like to run.
DISCOUNT=0.95
EPISODES=25000

SHOW_EVERY = 2000

#For the value at index 0, we can see the high value is 0.6, the low is -1.2, and then for the value at index 1, the high is 0.07, 
#and the low is -0.07. Okay, so these are the ranges, but from one of the above observation states that we output: [-0.27508804 -0.00268013], 
#we can see that these numbers can become quite granular. Can you imagine the size of a Q Table if we were going to have a value for every combination of 
#these ranges out to 8 decimal places? That'd be huge! And, more importantly, it'd be useless. We don't need that much granularity. So, instead, what we want 
#to do is conver these continuous values to discrete values. Basically, we want to bucket/group the ranges into something more manageable.

#Number of actions: 
#print(env.action_space.n)

DISCRETE_OS_SIZE=[20]*len(env.observation_space.high)
discrete_os_win_size=(env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

#How much exploration we want to do : epsilon (add some randomness)
# Exploration settings
epsilon = 0.5
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2

epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

#20*20*3 table :)
#So, this is a 20x20x3 shape, which has initialized random Q values for us. 
#The 20 x 20 bit is every combination of the bucket slices of all possible states. The x3 bit is for every possible action we could take.

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

#Next, we need a quick helper-function that will convert our environment "state," 
#which currently contains continuous values that would wind up making our Q-Table absolutely gigantic and take forever to learn.... to a "discrete" state instead:
def get_discrete_state(state):
	discrete_state = (state-env.observation_space.low)/discrete_os_win_size
	return tuple(discrete_state.astype(np.int))

#Looping code for each episode
for episode in range(EPISODES):

	if episode%SHOW_EVERY ==0:
		print(episode)
		render=True
	else: 
		render=False

	discrete_state=get_discrete_state(env.reset())

	done=False

	#List of actions
	 #0 = push left
	 #1 = do nothing
	 #2 = push right

	while not done:
		#Now we just need to use epsilon. We'll use np.random.random() to randomly pick a number 0 to 1. 
		#If np.random.random() is greater than the epsilon value, then we'll go based off the max q value as usual. Otherwise, we will just move randomly:
		if np.random.random() > epsilon:
			action = np.argmax(q_table[discrete_state])
		else:
			action = np.random.randint(0, env.action_space.n)

		new_state, reward, done, _ = env.step(action)
		new_discrete_state=get_discrete_state(new_state)
		if render: 
			env.render()
		# If simulation did not end yet after last step - update Q table
		if not done:
			#The max_future_q is grabbed after we've performed our action already, and then we update our previous values based 
			#partially on the next-step's best Q value. Over time, once we've reached the objective once, this "reward" value gets 
			#slowly back-propagated, one step at a time, per episode. Super basic concept, but pretty neat how it works!
			# Maximum possible Q value in next step (for new state)
			max_future_q=np.max(q_table[new_discrete_state])
			# Current Q value (for current state and performed action)
			current_q=q_table[discrete_state+(action, )]
			#THE FORMULA FOR ALL Q VALUES (we update the Q wqith the new value)
			# And here's our equation for a new Q value for current state and action
			new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
			# Update Q table with new Q value
			q_table[discrete_state+(action,)]=new_q
		 # Simulation ended (for any reson) - if goal position is achived - update Q value with reward directly
		elif new_state[0] >= env.goal_position:
			print(f"We made it on episode{episode}")
			q_table[discrete_state + (action,)] = 0

		discrete_state=new_discrete_state
	# Decaying is being done every episode if episode number is within decaying range	
	if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
		epsilon -= epsilon_decay_value

env.close()
