import random
import math
import numpy as np
from collections import deque


class Agent:
    # initialize all the hyper-parameters
    def __init__(self, env, model, memory_len=5000, gamma=1.0,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.985,
                 batch_size=256, train_batch=1, quiet=False):
        self.env = env
        self.model = model
        self.memory = deque(maxlen=memory_len)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.train_batch = train_batch
        self.quiet = quiet

    # util funtions
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state, epsilon):
        if(np.random.random() <= epsilon):
            return self.env.action_space.sample()
        else:
            return np.argmax(self.model.predict(state))

    def get_epsilon(self, t):
        return max(self.epsilon_min,
                   min(self.epsilon, 1.0-math.log10(t+1)*self.epsilon_decay))

    def preprocess_state(self, state):
        return np.reshape(state, [1, self.env.observation_space.shape[0]])

    def replay(self, epsilon):
        x_batch = []
        y_batch = []
        mini_batch = random.sample(self.memory,
                                   min(len(self.memory), self.batch_size))
        for state, action, reward, next_state, done in mini_batch:
            y_target = self.model.predict(state)
            if(done):
                y_target[0][action] = reward
            else:
                y_target[0][action] = reward + self.gamma * np.max(self.model.predict(next_state)[0])
            x_batch.append(state[0])
            y_batch.append(y_target[0])

        self.model.fit(np.array(x_batch), np.array(y_batch),
                       batch_size=self.train_batch, verbose=0)

        if(epsilon > self.epsilon_min):
            self.epsilon *= self.epsilon_decay

    # train function
    def train(self, max_episodes):
        print('Started Training for {} episodes.'.format(max_episodes))
        scores = deque(maxlen=100)

        for episode in range(max_episodes):
            state = self.preprocess_state(self.env.reset())
            done = False
            t = 0
            while(not done):
                action = self.choose_action(state, self.get_epsilon(episode))
                next_state, reward, done, info = self.env.step(action)
                next_state = self.preprocess_state(next_state)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                t += reward
            scores.append(t)
            mean_score = np.mean(scores)

            if(episode % 25 == 0 and not self.quiet):
                print('Episode {}: mean score={}'.format(episode, mean_score))
            self.replay(self.get_epsilon(episode))

    # play function
    def play(self, trials=3):
        print('Started Playing...')
        for trial in range(trials):
            state = self.preprocess_state(self.env.reset())
            done = False
            t = 0
            while(not done):
                action = np.argmax(self.model.predict(state))
                next_state, reward, done, info = self.env.step(action)
                self.env.render()
                next_state = self.preprocess_state(next_state)
                state = next_state
                t += 1
            print('Trail {}: score={}'.format(trial+1, t))
        self.env.close()
