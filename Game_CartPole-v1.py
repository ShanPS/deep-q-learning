from Agent_CartPole import Agent
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import gym


# make environment
env = gym.make('CartPole-v1')
# change to CartPole-v0 for v0 environment
lr = 0.005

# model
model = Sequential()
model.add(Dense(24, input_dim=env.observation_space.shape[0],
                activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(env.action_space.n, activation='relu'))

model.compile(loss='mse', optimizer=Adam(lr=lr))

# Create Agent, Train and Play the game using Agent.
agent = Agent(env, model)
agent.train(max_episodes=1000)
agent.play(trials=5)
