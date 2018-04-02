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
    def __init__(self, userNumber, totalCPU, channelModel):
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
        self.SNR_average = 10**(2)
        self.rho = 0.7
        self.punishment_delay = -userNumber
        self.channelModel = channelModel
        self.featureNumber = 4
        if userNumber == 2:
            self.action_size = totalCPU + 1
        if userNumber == 3:
            self.action_size = int(((totalCPU + 1) * (totalCPU + 2)) / 2)
        self.newTaskProbability = 0.99
        self.requiredErrorProbability = 10**(-3)
        self.taskSizeSpace = [500, 1000, 1500, 2000]
        self.taskCount = 0
        self.successTaskCount_best = 0
        self.successTaskCount_equal = 0
        self.delayTaskCount_best = 0

    def reset(self):
        self.userBuffer = []
        for i in range(self.userNumber):
            self.userBuffer.append([])
        self.channel_h = [0 for i in range(self.userNumber)]
        self.SNR = np.zeros(self.userNumber)
        self.taskCount = 0
        self.successTaskCount_best = 0
        self.successTaskCount_equal = 0
        self.delayTaskCount_best = 0

    def updateBuffer(self):
        z = [0.170279632305101, 0.903701776799380, 2.251086629866130, 4.266700170287658, 7.045905402393464,
             10.758516010180998, 15.740678641278004, 22.863131736889265]
        w = [0.369188589341638, 0.418786780814343, 0.175794986637172, 0.033343492261216, 0.002794536235226,
             9.076508773358205e-05, 8.485746716272531e-07, 1.048001174871507e-09]

        for i in range(self.userNumber):
            for j in range(len(self.userBuffer[i])):
                self.userBuffer[i][j]['waitTime'] += 1
            if np.random.rand() <= self.newTaskProbability:
                newTask = {'taskSize': np.random.choice(self.taskSizeSpace), 'waitTime': 0}
                self.userBuffer[i].append(newTask)
                self.taskCount += 1
            if self.channelModel == 1:
                self.SNR[i] = np.random.choice(z, p=w)
        if self.channelModel == 2:
            self.updateSNR()

    def updateSNR(self):
        for i in range(self.userNumber):
            h_bar = complex(np.random.randn(), np.random.randn()) / np.sqrt(2)
            if not self.SNR[i]:
                self.channel_h[i] = h_bar
            else:
                self.channel_h[i] = self.rho*self.channel_h[i]+np.sqrt(1-self.rho**2)*h_bar
            self.SNR[i] = abs(self.channel_h[i])**2

    def getState(self):
        State = np.zeros((self.userNumber, self.featureNumber))
        for i in range(self.userNumber):
            if not self.userBuffer[i]:
                State[i, 0] = 0
                State[i, 1] = 0
                State[i, 2] = 0
            else:
                State[i, 0] = self.userBuffer[i][0]['taskSize']
                State[i, 1] = self.userBuffer[i][0]['waitTime']
                State[i, 2] = len(self.userBuffer[i])
            State[i, 3] = self.SNR[i]
        return State

    def takeAction(self, action):
        successCount = 0
        reward = 0
        for i in range(self.userNumber):
            if (action[i] == 0) or (not self.userBuffer[i]):
                continue
            else:
                Dm = self.userBuffer[i][0]['taskSize']
                cpu = action[i]
                mc = int(np.ceil(Dm * self.L / (cpu * self.f0 * self.Ts)))
                m_MT = self.T_f - self.m_UM - mc
                if m_MT <= 0:
                    reward = reward - 5  # punishment for long delay of calculation
                    continue
                else:
                    Qfunction = lambda x: 0.5 - 0.5 * scipy.special.erf(x / np.sqrt(2))
                    gamma = self.SNR[i] * self.SNR_average
                    QfunctionInput = (np.log2(1 + gamma) - (self.beta * Dm / m_MT)) / (
                                np.log2(np.e) * np.sqrt((1 - (1 + gamma) ** (-2)) / m_MT))
                    decodeError = Qfunction(QfunctionInput)
                    if decodeError < self.requiredErrorProbability:
                        successCount += 1
                        reward = reward + 1
                    else:
                        reward = reward - 1
        return reward, successCount

    def action_equal(self):
        actionAarry = [int(np.floor(self.totalCPU / self.userNumber)) for i in range(self.userNumber)]
        actionAarry[np.argmin(self.SNR)] += (totalCPU - np.sum(actionAarry))
        return actionAarry

    def step_bestAllocate(self):
        reward_best = -10000
        success_best = 0
        action_best = actionTransfer(self.userNumber,self.totalCPU,0)
        for i in range(self.action_size):
            action_temp = actionTransfer(self.userNumber,self.totalCPU,i)
            reward_temp, success_temp = self.takeAction(action_temp)
            if reward_temp > reward_best:
                reward_best = reward_temp
                action_best = action_temp
                success_best = success_temp
        self.successTaskCount_best += success_best

        # for i in range(self.userNumber):
        #     if not self.userBuffer[i]:
        #         break
        #     if self.userBuffer[i][0]['waitTime'] == (self.alpha - 1):
        #         action_best = action_AllForOne(self.userNumber,self.totalCPU,i)
        #         reward_best = self.takeAction(action_best)
        #         break

        for i in range(self.userNumber):
            if (action_best[i] != 0) and (self.userBuffer[i]):
                self.userBuffer[i].pop(0)
        self.updateBuffer()
        done, count = self.checkDelayViolation()
        if done:
            self.delayTaskCount_best += count
            reward_best = self.punishment_delay
        return reward_best, action_best

    def equalAllocate(self):
        action = self.action_equal()
        reward, count = self.takeAction(action)
        self.successTaskCount_equal += count
        return reward, action

    def checkDelayViolation(self):
        done = False
        count = 0
        for i in range(self.userNumber):
            if self.userBuffer[i]:
                if self.userBuffer[i][0]['waitTime'] == self.alpha:
                    self.userBuffer[i].pop(0)
                    count += 1
                    done = True
        return done, count

    def step_network(self, action):
        reward = 0
        done = False
        for i in range(self.userNumber):
            if action[i] == 0:
                continue
            else:
                Dm = self.userBuffer[i][0]['taskSize']
                cpu = action[i]
                mc = int(np.ceil(Dm * self.L / (cpu * self.f0 * self.Ts)))
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
        for i in range(self.userNumber):
            if self.userBuffer[i][0]['waitTime'] == self.alpha:
                reward = self.punishment_done  # punishment for long waiting time
                done = True
        return newState, reward, done

class DQNAgent:
    def __init__(self, userNumber, totalCPU, W, networkModel):
        self.featureNumber = 4
        self.state_size = self.featureNumber * userNumber * W
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
        if networkModel == 1:
            self.model = self._build_FNN_model()
        if networkModel == 2:
            self.model = self._build_Con_model()

    def _build_FNN_model(self):
        # FNN Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(InputLayer(input_shape=(self.userNumber, self.featureNumber, W)))
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
        model.add(Conv2D(20, (2, 2), input_shape=(self.userNumber, self.featureNumber, W)))
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



def action_AllForOne(userNumber, totalCPU, userIndex):
    actionAarry = []
    if userNumber == 2:
        if userIndex == 0:
            actionAarry = [totalCPU,0]
        else:
            actionAarry = [0,totalCPU]
    if userNumber == 3:
        if userIndex == 0:
            actionAarry = [totalCPU,0,0]
        elif userIndex == 1:
            actionAarry = [0,totalCPU,0]
        else:
            actionAarry = [0,0,totalCPU]
    return actionAarry

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
    userNumber = 3
    totalCPU = 4
    W = 4

    batch_size = 32
    EPISODES = 1
    done = False

    networkModel = 1    # 1: normal 2: convolution
    agent = DQNAgent(userNumber, totalCPU, W, networkModel)
    channelModel = 2   # 1: discrete 2: correlation
    env = Environment(userNumber, totalCPU, channelModel)


    reward_record = []
    reward_equal_record = []
    reward_best_record = []

    time_record = []

    bestActionModel = True

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
            if bestActionModel:
                x_t = env.getState()
                reward_equal, actionEqual = env.equalAllocate()
                reward_best, actionBest = env.step_bestAllocate()
                reward_best_record.append(reward_best)
                reward_equal_record.append(reward_equal)
                # print(time)
                # print(x_t)
                # print(actionBest)
                # print(actionEqual)
                if time > 1000:
                    break
            else:
                actionIndex = agent.act(state)
                actionAarry = actionTransfer(userNumber, totalCPU, actionIndex)
                reward_equal = env.equalAllocate()
                reward_equal_record.append(reward_equal)
                x_t1, reward, done = env.step_network(actionAarry)
                x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1)
                next_state = np.append(x_t1, state[:, :, :, :W - 1], axis=3)
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
    bestSuccRate = 'best success rate: ' + repr(env.successTaskCount_best / env.taskCount) + ', success number:' + repr(env.successTaskCount_best)
    equalSuccRate = 'equal success rate: ' + repr(env.successTaskCount_equal / env.taskCount) + ', success number:' + repr(env.successTaskCount_equal)
    delayRate = 'delay rate: ' + repr(env.delayTaskCount_best / env.taskCount) + ', delay number:' + repr(env.delayTaskCount_best)
    totalTaskNumber = 'total number:' + repr(env.taskCount)
    print(bestSuccRate)
    print(equalSuccRate)
    print(delayRate)
    print(totalTaskNumber)
    plt.figure()
    plt.plot(reward_equal_record, label='equal', marker='+')
    plt.plot(reward_best_record, label='best', marker='.')
    plt.legend()
    # plt.figure()
    # plt.plot(reward_record)
    #
    # plt.figure()
    # plt.plot(time_record)
    plt.grid(True)
    plt.show()

    plt.close()
    # os.system("pause")