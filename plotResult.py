import numpy as np
import matplotlib
from matplotlib import pylab as plt

load_data = np.load('result/p10w3.npz')

network_episode_success_rate_record = load_data['network_episode_success_rate_record']
network_episode_delay_rate_record = load_data['network_episode_delay_rate_record']
real_network_episode_success_rate_record = load_data['real_network_episode_success_rate_record']
real_network_episode_delay_rate_record = load_data['real_network_episode_delay_rate_record']

randomSucc=0.754*np.ones(100)
randomDrop=0.044*np.ones(100)
equalSucc=0.81*np.ones(100)


font = {'family' : 'Times New Roman',
        'size'   : 14}

matplotlib.rc('font', **font)
# plot accuracy related to historical time tags and hidden layer numbers
plt.figure()
plt.plot(real_network_episode_success_rate_record,label='Q-network success')
plt.plot(real_network_episode_delay_rate_record,label='Q-network drop')
plt.plot(equalSucc,'k',label='Equal success')
plt.plot(randomSucc,'r',label='Random success')
plt.plot(randomDrop,'g',label='Random drop')

# plt.title('Accuracy related to layer numbers and historical time tags')
plt.xlabel('Episode')
plt.ylabel('Average task success and drop ratio')
# plt.legend(loc='right')
# plt.annotate('Binary', xy=(4, 0.94), xytext=(2, 0.86),
#             arrowprops=dict(facecolor='black', shrink=0.05),
#             )
# plt.annotate('Non-zero', xy=(4, 0.43), xytext=(2, 0.53),
#             arrowprops=dict(facecolor='black', shrink=0.05),
#             )
plt.legend(loc='right')
plt.grid(True)
plt.show()
# plt.savefig('p05w3.pdf')
plt.close()
