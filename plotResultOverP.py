import numpy as np
import matplotlib
from matplotlib import pylab as plt

# load_data = np.load('result/p05w3.npz')

# network_episode_success_rate_record = load_data['network_episode_success_rate_record']
# network_episode_delay_rate_record = load_data['network_episode_delay_rate_record']
# real_network_episode_success_rate_record = load_data['real_network_episode_success_rate_record']
# real_network_episode_delay_rate_record = load_data['real_network_episode_delay_rate_record']

p=[0.5,0.6,0.7,0.8,0.9,1]
randomSucc=[0.754,0.736,0.715,0.692,0.648,0.586]
randomDrop=[0.044,0.059,0.084,0.117,0.172,0.249]
equalSucc=[0.812,0.795,0.764,0.745,0.704,0.683]
networkSucc=[0.91,0.9,0.889,0.865,0.833,0.801]
networkDrop=[0.011,0.014,0.0178,0.0187,0.0193,0.02]

font = {'family' : 'Times New Roman',
        'size'   : 14}

matplotlib.rc('font', **font)
# plot accuracy related to historical time tags and hidden layer numbers
plt.figure()
plt.plot(p,networkSucc,label='Q-network success',marker='^')
plt.plot(p,equalSucc,'k',label='Equal success',marker='o')
plt.plot(p,randomSucc,'r',label='Random success',marker='*')


# plt.title('Accuracy related to layer numbers and historical time tags')
plt.xlabel('Task arravial probability')
plt.ylabel('Average task success ratio')

plt.legend()
plt.grid(True)
# plt.show()
plt.savefig('succ.pdf')
plt.close()

plt.figure()
plt.plot(p,networkDrop,label='Q-network drop',marker='^')
plt.plot(p,randomDrop,'r',label='Random drop',marker='*')

# plt.title('Accuracy related to layer numbers and historical time tags')
plt.xlabel('Task arravial probability')
plt.ylabel('Average task drop ratio')

plt.legend()
plt.grid(True)
# plt.show()
plt.savefig('drop.pdf')
plt.close()
