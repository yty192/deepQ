import random
import scipy.special
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import InputLayer, Flatten

class Environment:
    def __init__(self, userNumber):
        self.userNumber = userNumber
        self.userBuffer = []
        for i in range(userNumber):
            self.userBuffer.append([])
        self.SNR = []
        self.beta = 0.5
        self.f0 = 100
        self.Ts = 0.5   # time of one symbol in ms
        self.m_UM = 500
        self.alpha = 5
        self.T_f = 20   # time of one frame in symbols = 10 ms
        self.dr = self.alpha * self.T_f

    def reset(self):
        self.userBuffer = []
        for i in range(self.userNumber):
            self.userBuffer.append([])
        self.SNR = np.zeros(self.userNumber)

    def updateBuffer(self):
        z = [0.170279632305101, 0.903701776799380, 2.251086629866130, 4.266700170287658, 7.045905402393464,
             10.758516010180998, 15.740678641278004, 22.863131736889265]
        w = [0.369188589341638, 0.418786780814343, 0.175794986637172, 0.033343492261216, 0.002794536235226,
             9.076508773358205e-05, 8.485746716272531e-07, 1.048001174871507e-09]
        for i in range(self.userNumber):
            for j in range(len(self.userBuffer[i])):
                self.userBuffer[i][j]['waitTime'] += 1
            newTask = {'taskSize': 3, 'waitTime': 0}
            self.userBuffer[i].append(newTask)
            self.SNR[i] = np.random.choice(z, p=w)

    def giveState(self):
        State = np.zeros((self.userNumber, 3))
        for i in range(self.userNumber):
            State[i,0] = self.userBuffer[i][0]['taskSize']
            State[i,1] = self.userBuffer[i][0]['waitTime']
            State[i,2] = self.SNR[i]
        return State

    def takeAction(self, action):
        reward = 0
        done = False
        for i in range(self.userNumber):
            if action[i] == 0:
                continue
            else:
                Dm = self.beta * self.userBuffer[i][0]['taskSize']
                cpu = action[i]
                mc = Dm / (cpu * self.f0 * self.Ts)
                m_MT = self.dr - self.m_UM - mc - self.userBuffer[i][0]['waitTime']
                self.userBuffer[i].pop(0)
                if m_MT <= 0:
                    reward = reward - 1 # punishment for long delay of calculation
                    continue
                else:
                    Qfunction = lambda x: 0.5 - 0.5 * scipy.special.erf(x / np.sqrt(2))
                    gamma = self.SNR[i]
                    QfunctionInput = (np.log2(1+gamma) - (Dm/m_MT)) / (np.log2(np.e)*np.sqrt((1-(1+gamma)**(-2))/m_MT))
                    decodeError = Qfunction(QfunctionInput)
                    reward = reward + (1-decodeError)
        self.updateBuffer()
        newState = self.giveState()
        for i in range(self.userNumber):
            if self.userBuffer[i][0]['waitTime'] == self.dr:
                reward = -10  # punishment for long waiting time
                done = True
        return newState, reward, done

class DQNAgent:
    def __init__(self, userNumber, totalCPU, W, modelType):
        self.state_size = 3 * userNumber
        self.userNumber = userNumber
        if userNumber == 2:
            self.action_size = totalCPU + 1
        if userNumber == 3:
            self.action_size = ((totalCPU + 1) * (totalCPU + 2)) / 2
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        if modelType == 1:
            self.model = self._build_FNN_model()
        if modelType == 2:
            self.model = self._build_Con_model()

    def _build_FNN_model(self):
        # FNN Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(InputLayer(input_shape=(self.userNumber, 3)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def _build_Con_model(self):
        # 1D Convolutional Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(InputLayer(input_shape=(self.state_size, W)))
        model.add(Flatten())
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = state.reshape((1, state.shape[0], state.shape[1]))
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = state.reshape((1,state.shape[0],state.shape[1]))
            next_state = next_state.reshape((1, next_state.shape[0], next_state.shape[1]))
            target = reward
            if not done:
                target = reward + self.gamma * np.max(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)[0]
            target_f[action] = target
            # self.model.train_on_batch(state, target_f)
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def actionTransfer(userNumber, totalCPU, actionIndex):
    if userNumber == 2:
        for i in range(totalCPU+1):
            if i == actionIndex:
                actionArray = [i,totalCPU-i]
                return actionArray

    if userNumber == 3:
        index = 0
        for i in range(totalCPU+1):
            for j in range(totalCPU+1-i):
                if index == actionIndex:
                    actionArray = [i,j,totalCPU-i-j]
                    return actionArray
                index += 1

if __name__ == "__main__":
    userNumber = 2
    totalCPU = 8
    W = 1

    batch_size = 32
    EPISODES = 5000
    done = False

    agent = DQNAgent(userNumber, totalCPU, W, 1)
    env = Environment(userNumber)

    for episode in range(EPISODES):
        env.reset()
        env.updateBuffer()
        state = env.giveState()

        for time in range(1000):
            actionIndex = agent.act(state)
            actionAarry = actionTransfer(userNumber,totalCPU,actionIndex)
            next_state, reward, done = env.takeAction(actionAarry)
            agent.remember(state, actionIndex, reward, next_state, done)
            state = next_state
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
            if done:
                print("episode: {}/{}, time: {}, e: {:.2}".format(episode, EPISODES, time, agent.epsilon))
                break

        # if episode % 10 == 0:
        #     agent.save("./save/MEC-dqn.h5")