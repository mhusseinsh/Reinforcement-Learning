import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import csv
import numpy as np

q_random = []
q_av_random = []

with open('rew.txt','r') as csvfile:
    plots = csv.reader(csvfile)
    for row in plots:
        q_random.append(float(row[0]))
        cum_sum = np.cumsum(q_random)
        size = len(q_random)
        q_av_random.append(cum_sum[-1] / size)

fig1 = plt.figure()
random, = plt.plot(q_av_random)
plt.xlabel('Episode')
plt.ylabel('Rolling Average Q Values')
plt.title('Training - Episode Q Values')
plt.show()
#fig1.savefig('training_q_values_mean.png')
