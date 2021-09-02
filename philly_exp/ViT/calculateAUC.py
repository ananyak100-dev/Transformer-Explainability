
import numpy as np
from sklearn import metrics

x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

y_dino_small_pos = [73.93, 25.648, 11.498, 5.912, 3.224, 1.852, 1.056, 0.664, 0.35, 0.204]
print('DINO Small AUC Pos: ')
print(metrics.auc(x, y_dino_small_pos))
y_dino_small_neg = [73.93, 69.494, 68.1, 66.35, 63.696, 59.92, 54.398, 45.522, 32.692, 14.264]
print('DINO Small AUC Neg: ')
print(metrics.auc(x, y_dino_small_neg))

y_deit_small_pos = [77.14, 41.888, 25.612, 15.096, 8.238, 4.208, 2.046, 1.13,  0.574, 0.334]
print('DeiT Small AUC Pos: ')
print(metrics.auc(x, y_deit_small_pos))
y_deit_small_neg = [77.14, 69.494, 68.1, 66.35, 63.696, 59.92, 54.398, 45.522, 32.692, 14.264]
print('DeiT Small AUC Neg: ')
print(metrics.auc(x, y_deit_small_neg))

y_deit_base_pos = [80.21, 48.704, 30.344, 18.006, 10.408, 5.534, 2.808, 1.364, 0.642, 0.31]
print('DeiT Base AUC Pos: ')
print(metrics.auc(x, y_deit_base_pos))
y_deit_base_neg = [80.21, 77.126, 75.898, 73.982, 71.404, 67.448, 61.754, 51.806, 37.662, 16.952]
print('DeiT Base AUC Neg: ')
print(metrics.auc(x, y_deit_base_neg))

y_vit_base_pos = [0.8178, 0.50562, 0.3332, 0.21002, 0.1239, 0.06858, 0.03436, 0.01606, 0.00752, 0.0035 ]
print('ViT Base AUC Pos: ')
print(metrics.auc(x, y_vit_base_pos))

y_vit_base_neg = [0.8178, 0.7818,  0.76508, 0.73956, 0.70408, 0.65108, 0.5729,  0.45658, 0.2951,  0.1046 ]
print('ViT Base AUC Neg: ')
print(metrics.auc(x, y_vit_base_neg))

# import matplotlib
# plt.plot(x, y_dino_small_pos)
# plt.xlabel('Perturbation Levels')
# plt.ylabel('Accuracy (%)')
# plt.savefig(OUTPUT_PERTURB_DIR + '/plot.png')


# y_dino = [25.648, 11.498, 5.912, 3.224, 1.852, 1.056, 0.664, 0.35, 0.204]
#
# print('Dino AUC: ')
# print(metrics.auc(x, y_dino))
#
# y_deit = [41.888, 25.612, 15.096, 8.238, 4.208, 2.046, 1.13,  0.574, 0.334]
#
# print('DeiT AUC: ')
# print(metrics.auc(x, y_deit))

import matplotlib.pyplot as plt
import os

# plt.plot(x, y_dino)
# plt.xlabel('Perturbation Levels')
# plt.ylabel('Accuracy (%)')
# plt.savefig(os.environ['AMLT_DATA_DIR'] + "/output/DINO/DINO_Small Perturbations/False" + '/plot.png')
#
# plt.plot(x, y_deit)
# plt.xlabel('Perturbation Levels')
# plt.ylabel('Accuracy (%)')
# plt.savefig(os.environ['AMLT_DATA_DIR'] + "/output/DeiT/DeiT_Small Perturbations/False" + '/plot.png')