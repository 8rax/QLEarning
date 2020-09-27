#imports
#https://pythonprogramming.net/deep-q-learning-dqn-reinforcement-learning-python-tutorial/
#We are building a new environment with blobs, player tries to eat food, enemy tries to touch player

#With DQNs, instead of a Q Table to look up values, you have a model that you inference (make predictions from), and rather than updating the Q table, you fit (train) your model.

#The DQN neural network model is a regression model, which typically will output values for each of our possible actions. These values will be continuous float values, and they are directly our Q values.
#As we enage in the environment, we will do a .predict() to figure out our next move (or move randomly). When we do a .predict(), 
#we will get the 3 float values, which are our Q values that map to actions. We will then do an argmax on these, like we would with our Q Table's values. 
#We will then "update" our network by doing a .fit() based on updated Q values. When we do this, we will actually be fitting for all 3 Q values, even though we intend to just "update" one. 

#With the introduction of neural networks, rather than a Q table, the complexity of our environment can go up significantly, without necessarily requiring more memory. 
#As you can find quite quick with our Blob environment from previous tutorials, an environment of still fairly simple size, say, 50x50 will exhaust the memory of most people's computers. With a neural network, 
#we don't quite have this problem. Also, we can do what most people have done with DQNs and make them convolutional neural networks. 


from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from collections import deque
import time
import numpy as np

REPLAY_MEMORY_SIZE=50000
MODEL_NAME="256x2"

# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

	#The next thing you might be curious about here is self.tensorboard, which you can see is this ModifiedTensorBoard object. We're doing this to keep our log writing under control. 
	#Normally, Keras wants to write a logfile per .fit() which will give us a new ~200kb file per second. That's a lot of files and a lot of IO, where that IO can take longer even than the .fit(), 
	#so Daniel wrote a quick fix for that:

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)


class DQNAgent:
	def __init__(self):

		#Main model this is what gets trained every step
		self.model=self.create_model()

		#Target model this is what we .predict against every step
		#Here, you can see there are apparently two models: self.model and self.target_model. What's going on here? So every step we take, we want to update Q values, but we also are
		#trying to predict from our model. Especially initially, our model is starting off as random, and it's being updated every single step, per every single episode. 
		#What ensues here are massive fluctuations that are super confusing to our model. This is why we almost always train neural networks with batches (that and the time-savings).
		#One way this is solved is through a concept of memory replay, whereby we actually have two models.

		self.target_model=self.create_model()
		self.target_model.set_weights(self.model.get_weights())


		#Along these lines, we have a variable here called replay_memory. Replay memory is yet another way that we attempt to keep some sanity in a model that is 
		#getting trained every single step of an episode. We still have the issue of training/fitting a model on one sample of data. This is still a problem with neural networks. 
		#Thus, we're instead going to maintain a sort of "memory" for our agent. In our case, we'll remember 1000 previous actions, and then we will fit our model on a random selection 
		#of these previous 1000 actions. This helps to "smooth out" some of the crazy fluctuations that we'd otherwise be seeing. Like our target_model, we'll get a better idea of what's 
		#going on here when we actually get to the part of the code that deals with this I think.

		self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
		self.tensorboard=ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
		self.target_update_counter=0

	def create_model(self):
		model = Sequential()
		model.add(Conv2D(256, (3,3), input_shape=env.OBSERVATION_SPACE_VALUES))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(2,2))
		model.add(Dropout(0.2))

		model.add(Conv2D(256, (3,3)))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(2,2))
		model.add(Dropout(0.2))

		model.add(Flatten())
		model.add(Dense(64))

		model.add(Dense(env.ACTION_SPACE_SIZE, activation="linear"))
		model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
		return model

	def update_replay_memory(self, transition): 
		self.replay_memory.append(transition)

	def get_qs(self, terminal_state, step):
		return self.model_predict(np.array(state).reshape(-1,*state.shape)/255)[0]