import numpy as np
import matplotlib
from matplotlib import pyplot as plt

load_data = np.load('result/3db_p04w3_negative1punishment.npz')

network_episode_success_rate_record = load_data['network_episode_success_rate_record']
network_episode_delay_rate_record = load_data['network_episode_delay_rate_record']
real_network_episode_success_rate_record = load_data['real_network_episode_success_rate_record']
real_network_episode_delay_rate_record = load_data['real_network_episode_delay_rate_record']

# print(np.max(real_network_episode_success_rate_record))

randomSucc=0.863765*np.ones(100)
randomDrop=0.03456*np.ones(100)
equalSucc=0.938955*np.ones(100)


font = {'family' : 'Times New Roman',
        'size'   : 14}
linewidth_network=2
linewidth=2.0
matplotlib.rc('font', **font)
# plot accuracy related to historical time tags and hidden layer numbers
plt.figure()
plt.plot(real_network_episode_success_rate_record,label='Q-network success',linewidth=linewidth_network)
plt.plot(real_network_episode_delay_rate_record,label='Q-network drop',linewidth=linewidth_network)
plt.plot(equalSucc,'k',label='Equal success',linewidth=linewidth,linestyle='-')
plt.plot(randomSucc,'r',label='Random success',linewidth=linewidth,linestyle='--')
plt.plot(randomDrop,'g',label='Random drop',linewidth=linewidth,linestyle='-.')
plt.xlim(xmin=0)
plt.xlim(xmax=100)
plt.ylim(ymin=0)
plt.ylim(ymax=1)
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
# plt.show()
plt.savefig('p04w3_new1.pdf',bbox_inches='tight')
plt.close()
