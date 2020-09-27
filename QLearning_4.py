#imports
#We are building a new environment with blobs, player tries to eat food, enemy tries to touch player
#https://pythonprogramming.net/own-environment-q-learning-reinforcement-learning-python-tutorial/

import numpy as np  # for array stuff and random
from PIL import Image  # for creating visual of our env
import cv2  # for showing our visual live
import matplotlib.pyplot as plt  # for graphing our mean rewards over time
import pickle  # to save/load Q-Tables
from matplotlib import style  # to make pretty charts because it matters.
import time  # using this to keep track of our saved Q-Tables.

style.use("ggplot")  # setting our style!

#Constants
SIZE = 10
HM_EPISODES = 25000
MOVE_PENALTY = 1
ENEMY_PENALTY = 300
FOOD_REWARD = 25
epsilon = 0.0
EPS_DECAY = 0.9998
SHOW_EVERY = 1
#start_q_table = None #or filename
start_q_table = "qtable-1600269664.pickle" #or filename
LEARNING_RATE = 0.1
DISCOUNT = 0.95

PLAYER_N = 1  # player key in dict
FOOD_N = 2  # food key in dict
ENEMY_N = 3  # enemy key in dict

# the dict! Using just for colors
d = {1: (255, 175, 0),  # blueish color
     2: (0, 255, 0),  # green
     3: (0, 0, 255)}  # red

# What is a blob ?
class Blob:
    def __init__(self):
        self.x = np.random.randint(0,SIZE)
        self.y = np.random.randint(0,SIZE)

    def __str__(self):
        return f"{self.x},{self.y}"

    def __sub__(self, other):
        return (self.x - other.x, self.y - other.y)

    def action(self,choice):
        #4 discrete Actions
        if choice == 0:
            self.move(x=1,y=1)
        if choice == 1:
            self.move(x=-1,y=-1)    
        if choice == 2:
            self.move(x=-1,y=1) 
        if choice == 3:
            self.move(x=1,y=-1)

    def move(self, x=False, y=False):
        # If no value for x, move randomly
        if not x:
            self.x += np.random.randint(-1,2)
        else:
            self.x += x
        # If no value for y, move randomly  
        if not y:
            self.y += np.random.randint(-1,2)
        else:
            self.y += y

        # If we are out of bounds, fix!
        if self.x < 0:
            self.x=0
        elif self.x>SIZE-1:
            self.x=SIZE-1
        if self.y < 0:
            self.y=0
        elif self.y>SIZE-1:
            self.y=SIZE-1

#We create the Q table which lists all the different situations of our scenario and then will pass a Q value for every action
#If we don't have an existing table, we initialise a new one with random values
if start_q_table is None: 
    q_table = {}
    for x1 in range (-SIZE+1, SIZE):
            for y1 in range (-SIZE+1, SIZE):
                    for x2 in range (-SIZE+1, SIZE):
                            for y2 in range (-SIZE+1, SIZE):
                                #initialize
                                q_table[((x1, y1),(x2, y2))] = [np.random.uniform(-5,0) for i in range(4)]

#if we have an existing table, we load it
else: 
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)

episode_rewards=[]

for episode in range(HM_EPISODES):
    player= Blob()
    food = Blob()
    enemy = Blob()

    if episode%SHOW_EVERY==0:
        print(f"on * {episode}, epsilon: {epsilon}")
        print(f"{SHOW_EVERY} ep mean {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show=True
    else:
        show=False

    episode_reward=0

    for i in range(200):
        obs = (player-food, player-enemy)
        if np.random.random() > epsilon:
            action = np.argmax(q_table[obs])
        else:
            action = np.random.randint(0,4)

        player.action(action)

        #Handling collisions and rewarding the right ones
        if player.x == enemy.x and player.y == enemy.y:
            reward = -ENEMY_PENALTY
        elif player.x == food.x and player.y == food.y: 
            reward = FOOD_REWARD
        else:
            reward = -MOVE_PENALTY

        new_obs = (player-food, player-enemy)
        max_future_q = np.max(q_table[new_obs])
        current_q = q_table[obs][action]

        #QLEARNING
        #new_obs = (player-food, player-enemy)
        #new observation
        #max_future_q = np.max(q_table[new_obs])
        #max Q value for this new obs
        #current_q = q_table[obs][action]
        #current Q for our chosen action

        if reward == FOOD_REWARD:
            new_q = FOOD_REWARD
        elif reward == -ENEMY_PENALTY:
            new_q = -ENEMY_PENALTY
        else: 
            new_q = (1-LEARNING_RATE) * current_q + LEARNING_RATE * (reward+DISCOUNT*max_future_q) 

        q_table[obs][action] = new_q

        #Showing the screen and the image
        if show:
            env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)  # starts an rbg of our size
            env[food.x][food.y] = d[FOOD_N]  # sets the food location tile to green color
            env[player.x][player.y] = d[PLAYER_N]  # sets the player tile to blue
            env[enemy.x][enemy.y] = d[ENEMY_N]  # sets the enemy location to red
            img = Image.fromarray(env, 'RGB')  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
            img = img.resize((300, 300))  # resizing so we can see our agent in all its glory.
            cv2.imshow("image", np.array(img))  # show it!
            if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:  # crummy code to hang at the end if we reach abrupt end for good reasons or not.
                if cv2.waitKey(500) & 0xFF == ord('q'):
                    break
            else:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        episode_reward += reward
        if reward == FOOD_REWARD or reward ==- ENEMY_PENALTY:
            break
    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY

moving_avg=np.convolve(episode_rewards, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode="valid")

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"rward {SHOW_EVERY}ma")
plt.xlabel("episode #")
plt.show()
#After closing that graph, the q-table will save, along with the timestamp of that Q-Table. Now, we can load in this table, and either play, learn, or both. For example, we can just change SHOW_EVERY to 1:

with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table, f)