import random
import scipy.special
import numpy as np
from collections import deque
from keras.models import Sequential, load_model
from keras.layers import Dense, InputLayer, Flatten, Conv2D
from keras.optimizers import Adam
from matplotlib import pylab as plt

# No reset of environment


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
        self.alpha = 3
        self.T_f = 600   # time of one frame in symbols
        self.dr = self.alpha * self.T_f * self.Ts   # delay time limitation in us
        self.L = 2640   # required calculation cycle per bit
        self.SNR_average = 10**(2.5)
        self.rho = 0.7
        self.punishment_delay = -1
        self.channelModel = channelModel
        self.featureNumber = 4
        if userNumber == 2:
            self.action_size = totalCPU + 1
        if userNumber == 3:
            self.action_size = int(((totalCPU + 1) * (totalCPU + 2)) / 2)
        self.newTaskProbability = 1
        self.requiredErrorProbability = 10**(-4)
        self.taskSizeSpace = [500, 1000, 1500, 2000]
        self.taskCount = 0
        self.successTaskCount_best = 0
        self.successTaskCount_equal = 0
        self.successTaskCount_random = 0
        self.delayTaskCount_random = 0
        self.delayTaskCount_best = 0
        self.successTaskCount_network = 0
        self.delayTaskCount_network = 0

    def reset(self):
        self.userBuffer = []
        for i in range(self.userNumber):
            self.userBuffer.append([])
        self.channel_h = [0 for i in range(self.userNumber)]
        self.SNR = np.zeros(self.userNumber)
        self.taskCount = 0
        self.successTaskCount_best = 0
        self.successTaskCount_equal = 0
        self.successTaskCount_random = 0
        self.delayTaskCount_best = 0
        self.successTaskCount_network = 0
        self.delayTaskCount_network = 0
        self.delayTaskCount_random = 0


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
                        reward += 1
                    else:
                        reward -= 1
        return reward, successCount

    def action_equal(self):
        if self.userBuffer[0] and self.userBuffer[1]:
            actionAarry = [int(np.floor(self.totalCPU / self.userNumber)) for i in range(self.userNumber)]
            actionAarry[np.argmin(self.SNR)] += (totalCPU - np.sum(actionAarry))
        elif self.userBuffer[0]:
            actionAarry = [self.totalCPU,0]
        else:
            actionAarry = [0,self.totalCPU]
        return actionAarry

    def step_equal(self):
        action = self.action_equal()
        reward, count = self.takeAction(action)
        self.successTaskCount_equal += count

        for i in range(self.userNumber):
            if (action != 0) and (self.userBuffer[i]):
                self.userBuffer[i].pop(0)

        self.updateBuffer()

    def step_random(self):
        actionIndex = random.randrange(self.action_size)
        actionAarry = actionTransfer(self.userNumber, self.totalCPU, actionIndex)
        reward_random, success_random = self.takeAction(actionAarry)
        self.successTaskCount_random += success_random

        for i in range(self.userNumber):
            if (actionAarry[i] != 0) and (self.userBuffer[i]):
                self.userBuffer[i].pop(0)

        self.updateBuffer()
        done, countDelay = self.checkDelayViolation()
        if done:
            self.delayTaskCount_random += countDelay
            reward_random = self.punishment_delay

        # newState = self.getState()

        # return newState, reward_random, done

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
        done, countDelay = self.checkDelayViolation()
        if done:
            self.delayTaskCount_best += countDelay
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
        reward_network, success_network = self.takeAction(action)
        self.successTaskCount_network += success_network

        for i in range(self.userNumber):
            if (action[i] != 0) and (self.userBuffer[i]):
                self.userBuffer[i].pop(0)

        self.updateBuffer()
        done, countDelay = self.checkDelayViolation()
        if done:
            self.delayTaskCount_network += countDelay
            reward_network += self.punishment_delay  # punishment is negative, it should be added to the reward!!!!

        newState = self.getState()

        return newState, reward_network, done

class DQNAgent:
    def __init__(self, userNumber, totalCPU, W, networkModel):
        self.featureNumber = 4
        self.historyNumber = W
        self.state_size = self.featureNumber * userNumber * W
        self.userNumber = userNumber
        if userNumber == 2:
            self.action_size = totalCPU + 1
        if userNumber == 3:
            self.action_size = int(((totalCPU + 1) * (totalCPU + 2)) / 2)
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1  # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.999
        self.learning_rate = 0.001
        self.loss = 0
        self.loss_record = []
        self.exploration = True
        if networkModel == 1:
            self.model = self._build_FNN_model()
            self.model_temp = self.model
        if networkModel == 2:
            self.model = self._build_Con_model()
            self.model_temp = self.model
        if networkModel == 3:
            self.model = load_model(filepath="model/dqn2.h5")
            self.exploration = False

    def _build_FNN_model(self):
        # FNN Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(InputLayer(input_shape=(self.userNumber, self.featureNumber, self.historyNumber)))
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
        model.add(Conv2D(20, (2, 2), input_shape=(self.userNumber, self.featureNumber, self.historyNumber)))
        # model.add(Conv2D(40, (2, 2)))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if (np.random.rand() <= self.epsilon) and self.exploration:
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
        inputs = np.zeros((batch_size, self.userNumber, self.featureNumber, self.historyNumber))
        targets = np.zeros((batch_size, self.action_size))
        index = -1
        for state, action, reward, next_state, done in minibatch:
            # target_f = self.model.predict(state)
            # target_f[0][action] = reward + self.gamma * np.max(self.model.predict(next_state)[0])
            # self.model.fit(state, target_f, epochs=1, verbose=0)
            index += 1
            inputs[index: index+1] = state
            targets[index] = self.model.predict(state)[0]
            targets[index, action] = reward + self.gamma * np.max(self.model_temp.predict(next_state)[0])
        # self.model.fit(inputs, targets, epochs=1, verbose=0)
        self.loss = self.model.train_on_batch(inputs, targets)
        self.loss_record.append(self.loss)

    # def load(self):
    #     self.model = load_model(filepath="model/dqn_episode10_epsilon01.h5")

    def save(self):
        # self.model.save_weights(name)
        self.model.save(filepath="model/dqn2.h5")


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

def test_for_real_episode_rate(agent, userNumber, totalCPU, channelModel, testStep):
    testEnv = Environment(userNumber, totalCPU, channelModel)
    testEnv.reset()
    testEnv.updateBuffer()
    x_t = testEnv.getState()
    state = np.stack((x_t for i in range(W)), axis=2)
    state = state.reshape(1, state.shape[0], state.shape[1], state.shape[2])

    for i in range(testStep):
        actionIndex = agent.act(state)
        actionAarry = actionTransfer(userNumber, totalCPU, actionIndex)
        x_t1, reward, done = testEnv.step_network(actionAarry)
        x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1)
        next_state = np.append(x_t1, state[:, :, :, :W - 1], axis=3)
        state = next_state

    success_rate = testEnv.successTaskCount_network / testEnv.taskCount
    delay_rate = testEnv.delayTaskCount_network / testEnv.taskCount
    return success_rate, delay_rate


if __name__ == "__main__":
    userNumber = 2
    totalCPU = 3
    W = 3

    actionModel = 1  # 1: network, 2: best, 3: random, 4: equal

    if actionModel != 1:
        EPISODES = 10000
        total_step_number = 2 * EPISODES
    else:
        EPISODES = 500
        total_step_number = 100 * EPISODES

    batch_size = 32
    testStep = 10000
    update_model_temp = 200
    done = False

    networkModel = 1    # 1: normal 2: convolution 3: load trained model
    agent = DQNAgent(userNumber, totalCPU, W, networkModel)
    channelModel = 2   # 1: discrete 2: correlation
    env = Environment(userNumber, totalCPU, channelModel)

    reward_record = []
    reward_equal_record = []
    reward_best_record = []

    time_record = []

    network_episode_success_rate_record = []
    network_episode_delay_rate_record = []
    equal_episode_success_rate_record = []
    real_network_episode_success_rate_record = []
    real_network_episode_delay_rate_record = []


    env.reset()
    env.updateBuffer()
    x_t = env.getState()
    state = np.stack((x_t for i in range(W)), axis=2)
    state = state.reshape(1, state.shape[0], state.shape[1], state.shape[2])

    network_episode_success = 0
    network_episode_delay = 0
    episode_total_task = 0
    episode = 0

    for time in range(total_step_number):

        if actionModel == 2:
            # x_t = env.getState()
            # reward_equal, actionEqual = env.equalAllocate()
            reward_best, actionBest = env.step_bestAllocate()
            reward_best_record.append(reward_best)
            # reward_equal_record.append(reward_equal)
            # print(time)
            # print(x_t)
            # print(actionBest)
            # print(actionEqual)
        elif actionModel == 1:
            actionIndex = agent.act(state)
            actionAarry = actionTransfer(userNumber, totalCPU, actionIndex)
            # reward_equal, actionEqual = env.equalAllocate()
            # reward_equal_record.append(reward_equal)
            x_t1, reward, done = env.step_network(actionAarry)
            x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1)
            next_state = np.append(x_t1, state[:, :, :, :W - 1], axis=3)
            reward_record.append(reward)
            # print(actionAarry,reward, actionEqual,reward_equal)
            agent.remember(state, actionIndex, reward, next_state, done)
            state = next_state
            if ((time % update_model_temp) == 0):
                agent.model_temp = agent.model
            if (len(agent.memory) > batch_size) and agent.exploration:
                agent.replay(batch_size)
            # if time % 50 == 0:
            #     print(agent.loss)
        elif actionModel == 3:
            # reward_equal, actionEqual = env.equalAllocate()
            env.step_random()
            # reward_best_record.append(reward_best)
            # reward_equal_record.append(reward_equal)
        else:
            env.step_equal()

        if ((time % EPISODES) == 0) and (time != 0):
            episode += 1
            if actionModel == 2:
                bestSuccRate = 'best success rate: ' + repr(
                    env.successTaskCount_best / env.taskCount) + ', success number:' + repr(env.successTaskCount_best)
                equalSuccRate = 'equal success rate: ' + repr(
                    env.successTaskCount_equal / env.taskCount) + ', success number:' + repr(env.successTaskCount_equal)
                bestDelayRate = 'best delay rate: ' + repr(
                    env.delayTaskCount_best / env.taskCount) + ', delay number:' + repr(env.delayTaskCount_best)
                totalTaskNumber = 'total number:' + repr(env.taskCount)
                print(bestSuccRate)
                print(equalSuccRate)
                print(bestDelayRate)
                print(totalTaskNumber)
            elif actionModel == 1:
                network_episode_success = env.successTaskCount_network - network_episode_success
                network_episode_delay = env.delayTaskCount_network - network_episode_delay
                episode_total_task = env.taskCount - episode_total_task

                equal_episode_success_rate = env.successTaskCount_equal / env.taskCount
                equal_episode_success_rate_record.append(equal_episode_success_rate)
                network_episode_success_rate = network_episode_success / episode_total_task
                network_episode_delay_rate = network_episode_delay / episode_total_task
                network_episode_delay_rate_record.append(network_episode_delay_rate)
                network_episode_success_rate_record.append(network_episode_success_rate)

                network_episode_success = env.successTaskCount_network
                network_episode_delay = env.delayTaskCount_network
                episode_total_task = env.taskCount

                agent.exploration = False
                real_episode_success_rate, real_episode_delay_rate = test_for_real_episode_rate(agent, userNumber,
                                                                                                totalCPU, channelModel,
                                                                                                testStep)
                agent.exploration = True

                real_network_episode_success_rate_record.append(real_episode_success_rate)
                real_network_episode_delay_rate_record.append(real_episode_delay_rate)


                networkSuccRate = 'network success rate: ' + repr(
                    env.successTaskCount_network / env.taskCount) + ', success number: ' + repr(
                    env.successTaskCount_network) + ', total number: ' + repr(env.taskCount)
                networkSuccRate_episode = 'episode network success rate: ' + repr(
                    network_episode_success_rate) + ', success number: ' + repr(
                    network_episode_success) + ', episode total number: ' + repr(episode_total_task)
                equalSuccRate = 'equal success rate: ' + repr(equal_episode_success_rate) + ', success number: ' + repr(
                    env.successTaskCount_equal) + ', total number: ' + repr(env.taskCount)
                networkDelayRate = 'network delay rate: ' + repr(
                    env.delayTaskCount_network / env.taskCount) + ', delay number: ' + repr(
                    env.delayTaskCount_network) + ', total number: ' + repr(env.taskCount)
                networkDelayRate_episode = 'episode network delay rate: ' + repr(
                    network_episode_delay_rate) + ', delay number: ' + repr(
                    network_episode_delay) + ', episode total number: ' + repr(episode_total_task)
                network_loss = 'loss: ' + repr(agent.loss)

                print_real_episode_success_rate = 'real episode network success rate: ' + repr(real_episode_success_rate)
                print_real_episode_delay_rate = 'real episode network delay rate: ' + repr(real_episode_delay_rate)

                print(episode)
                print(networkSuccRate)
                print(networkSuccRate_episode)
                # print(equalSuccRate)
                print(networkDelayRate)
                print(networkDelayRate_episode)
                print(print_real_episode_success_rate)
                print(print_real_episode_delay_rate)


                print(network_loss)





            elif actionModel == 3:
                randomSuccRate = 'random success rate: ' + repr(
                    env.successTaskCount_random / env.taskCount) + ', success number:' + repr(
                    env.successTaskCount_random)
                equalSuccRate = 'equal success rate: ' + repr(
                    env.successTaskCount_equal / env.taskCount) + ', success number:' + repr(env.successTaskCount_equal)
                randomDelayRate = 'random delay rate: ' + repr(
                    env.delayTaskCount_random / env.taskCount) + ', delay number:' + repr(env.delayTaskCount_random)
                totalTaskNumber = 'total number:' + repr(env.taskCount)
                print(randomSuccRate)
                print(equalSuccRate)
                print(randomDelayRate)
                print(totalTaskNumber)

            else:
                equalSuccRate = 'equal success rate: ' + repr(
                    env.successTaskCount_equal / env.taskCount) + ', success number:' + repr(env.successTaskCount_equal)
                totalTaskNumber = 'total number:' + repr(env.taskCount)
                print(equalSuccRate)
                print(totalTaskNumber)







    if actionModel == 2:
        plt.figure()
        plt.plot(reward_equal_record, label='equal', marker='+')
        plt.plot(reward_best_record, label='best', marker='.')
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.close()
    elif actionModel == 1:
        agent.save()
        # np.savez('result/p06w2DiscreteChannel_negative05punishment', loss=agent.loss_record, network_episode_success_rate_record=network_episode_success_rate_record, network_episode_delay_rate_record=network_episode_delay_rate_record,real_network_episode_success_rate_record=real_network_episode_success_rate_record,real_network_episode_delay_rate_record=real_network_episode_delay_rate_record)
        plt.figure()
        plt.plot(network_episode_success_rate_record, label='DQN success', marker='+')
        plt.plot(network_episode_delay_rate_record, label='DQN drop', marker='.')
        plt.plot(real_network_episode_success_rate_record, label='real DQN success', marker='*')
        plt.plot(real_network_episode_delay_rate_record, label='real DQN drop', marker='^')
        # plt.plot(equal_episode_success_rate_record, label='equal_success_rate', marker='*')
        # plt.title('Accuracy related to layer numbers and historical time tags')
        plt.xlabel('Episode')
        plt.ylabel('Average task success and drop ratio')

        plt.legend()
        plt.grid(True)
        plt.figure()
        plt.plot(agent.loss_record)
        plt.grid(True)
        plt.show()
        plt.close()