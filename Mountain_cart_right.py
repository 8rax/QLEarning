import gym

#initialize environment
env = gym.make("MountainCar-v0")
env.reset()

#do an action in a loop
done = False
while not done:
	#make the cart go right
    action = 2  
    env.step(action)
    env.render()