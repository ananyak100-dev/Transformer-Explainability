
import numpy as np
from sklearn import metrics

x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

y_dino = [25.648, 11.498, 5.912, 3.224, 1.852, 1.056, 0.664, 0.35, 0.204]

print('Dino AUC: ')
print(metrics.auc(x, y_dino))

y_deit = [41.888, 25.612, 15.096, 8.238, 4.208, 2.046, 1.13,  0.574, 0.334]

print('DeiT AUC: ')
print(metrics.auc(x, y_deit))

import matplotlib.pyplot as plt
import os

plt.plot(x, y_dino)
plt.xlabel('Perturbation Levels')
plt.ylabel('Accuracy (%)')
plt.savefig(os.environ['AMLT_DATA_DIR'] + "/output/DINO/DINO_Small Perturbations/False" + '/plot.png')

plt.plot(x, y_deit)
plt.xlabel('Perturbation Levels')
plt.ylabel('Accuracy (%)')
plt.savefig(os.environ['AMLT_DATA_DIR'] + "/output/DeiT/DeiT_Small Perturbations/False" + '/plot.png')