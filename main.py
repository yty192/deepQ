import random
import scipy.special
import numpy as np
import os
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, InputLayer, Flatten, Conv2D
from keras.optimizers import Adam
from matplotlib import pylab as plt


class Environment:
    def __init__(self, userNumber, totalCPU):
        self.userNumber = userNumber
        self.totalCPU = totalCPU
        self.userBuffer = []
        for i in range(userNumber):
            self.userBuffer.append([])
        self.SNR = []
        self.channel_h = []
        self.beta = 0.5
        self.f0 = 3000     # calculation cycle per us
        self.Ts = 4.5  # time of one symbol in us
        self.m_UM = 200
        self.alpha = 4
        self.T_f = 600   # time of one frame in symbols
        self.dr = self.alpha * self.T_f * self.Ts   # delay time limitation in us
        self.L = 2640   # required calculation cycle per bit
        self.SNR_average = 10**(1.5)
        self.rho = 0.7
        self.punishment_done = -1000
        self.punishment_delay = -1000

    def reset(self):
        self.userBuffer = []
        for i in range(self.userNumber):
            self.userBuffer.append([])
        self.channel_h = [0 for i in range(self.userNumber)]
        self.SNR = np.zeros(self.userNumber)

    def updateBuffer(self):
        z = [0.170279632305101, 0.903701776799380, 2.251086629866130, 4.266700170287658, 7.045905402393464,
             10.758516010180998, 15.740678641278004, 22.863131736889265]
        w = [0.369188589341638, 0.418786780814343, 0.175794986637172, 0.033343492261216, 0.002794536235226,
             9.076508773358205e-05, 8.485746716272531e-07, 1.048001174871507e-09]
        taskSizeSpace = [1000,1500,2000]
        for i in range(self.userNumber):
            for j in range(len(self.userBuffer[i])):
                self.userBuffer[i][j]['waitTime'] += 1
            newTask = {'taskSize': np.random.choice(taskSizeSpace), 'waitTime': 0}
            self.userBuffer[i].append(newTask)
            # self.SNR[i] = np.random.choice(z, p=w)
        self.updateSNR()

    def updateSNR(self):
        for i in range(self.userNumber):
            h_bar = complex(np.random.randn(), np.random.randn()) / np.sqrt(2)
            if not self.SNR[i]:
                self.channel_h[i] = h_bar
            else:
                self.channel_h[i] = self.rho*self.channel_h[i]+(1-self.rho)*h_bar
            self.SNR[i] = abs(self.channel_h[i])**2

    def getState(self):
        State = np.zeros((self.userNumber, 3))
        for i in range(self.userNumber):
            State[i,0] = self.userBuffer[i][0]['taskSize']
            State[i,1] = self.userBuffer[i][0]['waitTime']
            State[i,2] = self.SNR[i]
        return State

    def equalAllocate(self):
        reward = 0
        for i in range(self.userNumber):
            Dm = self.userBuffer[i][0]['taskSize']
            cpu = np.floor(self.totalCPU / self.userNumber)
            mc = np.ceil(Dm * self.L / (cpu * self.f0 * self.Ts))
            m_MT = self.T_f - self.m_UM - mc
            if m_MT <= 0:
                reward = reward - self.punishment_delay # punishment for long delay of calculation
                continue
            else:
                Qfunction = lambda x: 0.5 - 0.5 * scipy.special.erf(x / np.sqrt(2))
                gamma = self.SNR[i] * self.SNR_average
                QfunctionInput = (np.log2(1+gamma) - (self.beta*Dm/m_MT)) / (np.log2(np.e)*np.sqrt((1-(1+gamma)**(-2))/m_MT))
                reward = reward + QfunctionInput
                # decodeError = Qfunction(QfunctionInput)
                # reward = 1 - decodeError
                # if decodeError == 1:
                #     continue
                #     # reward = reward - 10
                # elif decodeError == 0:
                #     reward = reward + 10
                # else:
                #     reward = reward + np.log10(1 / decodeError)
        reward = reward / self.userNumber
        return reward

    def takeAction(self, action):
        reward = 0
        done = False
        for i in range(self.userNumber):
            if action[i] == 0:
                continue
            else:
                Dm = self.userBuffer[i][0]['taskSize']
                cpu = action[i]
                mc = np.ceil(Dm * self.L / (cpu * self.f0 * self.Ts))
                m_MT = self.T_f - self.m_UM - mc
                self.userBuffer[i].pop(0)
                if m_MT <= 0:
                    reward = reward + self.punishment_delay # punishment for long delay of calculation
                    continue
                else:
                    # Qfunction = lambda x: 0.5 - 0.5 * scipy.special.erf(x / np.sqrt(2))
                    gamma = self.SNR[i] * self.SNR_average
                    QfunctionInput = (np.log2(1+gamma) - (self.beta*Dm/m_MT)) / (np.log2(np.e)*np.sqrt((1-(1+gamma)**(-2))/m_MT))
                    reward = reward + QfunctionInput
                    # decodeError = Qfunction(QfunctionInput)
                    # if decodeError == 1:
                    #     reward = reward - 10
                    # elif decodeError == 0:
                    #     reward = reward + 10
                    # else:
                    #     reward = reward + np.log10(1 / decodeError)
        self.updateBuffer()
        newState = self.getState()
        reward = reward / self.userNumber
        for i in range(self.userNumber):
            if self.userBuffer[i][0]['waitTime'] == self.alpha:
                reward = self.punishment_done  # punishment for long waiting time
                done = True
        return newState, reward, done

class DQNAgent:
    def __init__(self, userNumber, totalCPU, W, modelType):
        self.state_size = 3 * userNumber * W
        self.userNumber = userNumber
        if userNumber == 2:
            self.action_size = totalCPU + 1
        if userNumber == 3:
            self.action_size = int(((totalCPU + 1) * (totalCPU + 2)) / 2)
        self.memory = deque(maxlen=20000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 0.3  # exploration rate
        self.epsilon_min = 0.0001
        self.epsilon_decay = 0.999
        self.learning_rate = 0.001
        if modelType == 1:
            self.model = self._build_FNN_model()
        if modelType == 2:
            self.model = self._build_Con_model()

    def _build_FNN_model(self):
        # FNN Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(InputLayer(input_shape=(self.userNumber, 3, W)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def _build_Con_model(self):
        # 1D Convolutional Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Conv2D(20, (2, 2), input_shape=(self.userNumber, 3, W)))
        # model.add(Conv2D(40, (2, 2)))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            actionIndex = random.randrange(self.action_size)
        else:
            #   state = state.reshape((1, state.shape[0], state.shape[1]))
            act_values = self.model.predict(state)
            actionIndex = np.argmax(act_values[0])
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return actionIndex

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            #   state = state.reshape((1,state.shape[0],state.shape[1]))
            #   next_state = next_state.reshape((1, next_state.shape[0], next_state.shape[1]))
            target = reward
            if not done:
                target = reward + self.gamma * np.max(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)[0]
            target_f[action] = target
            # self.model.train_on_batch(state, target_f)
            self.model.fit(state, target_f[np.newaxis, ...], epochs=1, verbose=0)

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
    totalCPU = 7
    W = 4

    batch_size = 32
    EPISODES = 100
    done = False

    agent = DQNAgent(userNumber, totalCPU, W, 1)
    env = Environment(userNumber, totalCPU)

    reward_record = []
    reward_equal_record = []

    time_record = []

    for episode in range(EPISODES):
        env.reset()
        env.updateBuffer()
        x_t = env.getState()
        state = np.stack((x_t for i in range(W)),axis=2)
        state = state.reshape(1, state.shape[0], state.shape[1], state.shape[2])

        time = 0
        done = False
        while not done:
            time += 1

            actionIndex = agent.act(state)
            actionAarry = actionTransfer(userNumber,totalCPU,actionIndex)
            reward_equal = env.equalAllocate()
            reward_equal_record.append(reward_equal)
            x_t1, reward, done = env.takeAction(actionAarry)
            x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1)
            next_state = np.append(x_t1, state[:,:,:,:W-1], axis=3)
            reward_record.append(reward)
            agent.remember(state, actionIndex, reward, next_state, done)
            state = next_state
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
            if done:
                print("episode: {}/{}, time: {}, e: {:.2}".format(episode, EPISODES, time, agent.epsilon))
                break
            # if time > 100:
            #     # print("episode: {}/{}, time: {}, e: {:.2}".format(episode, EPISODES, time, agent.epsilon))
            #     print("ACTION", actionAarry, "/ REWARD", reward)
        time_record.append(time)
        # if episode % 10 == 0:
        #     agent.save("./save/MEC-dqn.h5")
    plt.figure()
    plt.plot(reward_equal_record)

    plt.figure()
    plt.plot(reward_record)

    plt.figure()
    plt.plot(time_record)

    plt.show()
    plt.close()
    # os.system("pause")