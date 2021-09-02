import os
# os.chdir(f'./Transformer-Explainability')
#
# !pip install -r requirements.txt

from PIL import Image
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
import numpy as np

models = ['DINO/DINO_Small Perturbations', 'DeiT/DeiT_Small Perturbations', 'DeiT/DeiT_Base Perturbations', 'ViT/ViT_Base Perturbations']
posNeg = ['False', 'True']

ROOT_DIR = os.environ['AMLT_DATA_DIR']

for model in models:
 for arg in posNeg:
  OUTPUT_DIR = ROOT_DIR + "/output/" + model + '/' + arg





print(np.mean(num_correct_model_full), np.std(num_correct_model_full))
print(np.mean(dissimilarity_model_full), np.std(dissimilarity_model_full))
print(np.mean(num_correct_perturb_full, axis=1), np.std(num_correct_perturb_full, axis=1))
print(np.mean(dissimilarity_perturb_full, axis=1), np.std(dissimilarity_perturb_full, axis=1))

means_num_correct_perturb = np.mean(num_correct_perturb_full, axis=1)
print('Means:')
for mean in means_num_correct_perturb:
 print(mean)

np.save(os.path.join(OUTPUT_PERTURB_DIR, 'means_num_correct_perturb_full.npy'), means_num_correct_perturb)

from numpy import savetxt
savetxt(os.path.join(OUTPUT_PERTURB_DIR, 'means_num_correct_perturb_full_csv.csv'), means_num_correct_perturb, delimiter=',')

# AUC

from sklearn import metrics
import matplotlib.pyplot as plt

x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
y = [100 * i for i in means_num_correct_perturb]

print('AUC: ')
print(metrics.auc(x, y))

plt.plot(x, y)
plt.xlabel('Perturbation Levels')
plt.ylabel('Accuracy (%)')
plt.savefig(OUTPUT_PERTURB_DIR + '/plot.png')