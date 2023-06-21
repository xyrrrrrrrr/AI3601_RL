import numpy as np
import matplotlib.pyplot as plt


epsilon = [0.001, 0.01, 0.05, 0.1, 0.3]
value = [0.005, 0.004, 0.003, 0.002, 0.001]
policy = [0.003, 0.002, 0.002, 0.003, 0.002]

plt.plot(epsilon, value, label='value', marker='o', markersize=6)
plt.plot(epsilon, policy, label='policy', marker='o',  markersize=6)
plt.xlabel('epsilon')
plt.ylabel('time cost')
plt.legend()
plt.show()