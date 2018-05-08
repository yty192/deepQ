import numpy as np
import matplotlib
from matplotlib import pyplot as plt


# load_data = np.load('result/p05w3.npz')

# network_episode_success_rate_record = load_data['network_episode_success_rate_record']
# network_episode_delay_rate_record = load_data['network_episode_delay_rate_record']
# real_network_episode_success_rate_record = load_data['real_network_episode_success_rate_record']
# real_network_episode_delay_rate_record = load_data['real_network_episode_delay_rate_record']

p=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
randomSucc=[0.878571,0.873959,0.869687,0.863765,0.852792,0.840633,0.819104,0.788262,0.73732,0.672516]
# randomDrop=[0.044,0.059,0.084,0.117,0.172,0.249]
equalSucc=[0.977881,0.966687,0.953947,0.938955,0.926775,0.913218,0.90076,0.885669,0.872681,0.859597]
networkSucc=[0.99168,0.9906,0.9895,0.983058,0.9824,0.974288,0.973704,0.971639,0.964912,0.941515]
# networkDrop=[0.011,0.014,0.0178,0.0187,0.0193,0.02]

font = {'family' : 'Times New Roman',
        'size'   : 14}

matplotlib.rc('font', **font)
# plot accuracy related to historical time tags and hidden layer numbers
plt.figure()
# plt.style.use('ggplot')
plt.plot(p,networkSucc,label='Q-network',marker='^',clip_on=False)
plt.plot(p,equalSucc,'k',label='Equal',marker='o',clip_on=False)
plt.plot(p,randomSucc,'r',label='Random',marker='*',clip_on=False)

# ax.get_xaxis().set_tick_params(direction='out', width=1)
# plt.title('Accuracy related to layer numbers and historical time tags')
plt.xticks(p)
plt.xlabel('Task arrival probability')
plt.ylabel('Average task success ratio')
# plt.axis([0.1, 1])
plt.xlim(xmin=0.1)
plt.xlim(xmax=1)
plt.ylim(ymax=1)
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig('succ1.pdf',bbox_inches='tight')
plt.close()

# plt.figure()
# plt.plot(p,networkDrop,label='Q-network drop',marker='^')
# plt.plot(p,randomDrop,'r',label='Random drop',marker='*')
#
# # plt.title('Accuracy related to layer numbers and historical time tags')
# plt.xlabel('Task arravial probability')
# plt.ylabel('Average task drop ratio')
#
# plt.legend()
# plt.grid(True)
# # plt.show()
# plt.savefig('drop.pdf')
# plt.close()
