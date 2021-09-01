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

ROOT_DIR = os.environ['AMLT_DATA_DIR']
OUTPUT_VIS_DIR = ROOT_DIR + "/output/ViT/ViT_Base Visualizations"
OUTPUT_PERTURB_DIR = ROOT_DIR + "/output/ViT/ViT_Base Perturbations/False"

all = list(np.arange(0, 1000, 1))

num_correct_model_list = []
dissimilarity_model_list = []

num_correct_perturb_list = []
dissimilarity_perturb_list = []
logit_diff_perturb_list = []
prob_diff_perturb_list = []

for subdir in all:

 folder_perturb = OUTPUT_PERTURB_DIR + '/' + str(subdir)

 num_correct_model_arr = np.load(folder_perturb + '/model_hits.npy')
 num_correct_model_list.append(num_correct_model_arr)
 dissimilarity_model_arr = np.load(folder_perturb + '/model_dissimilarities.npy')
 dissimilarity_model_list.append(dissimilarity_model_arr)

 num_correct_perturb_arr = np.load(folder_perturb + '/perturbations_hits.npy')
 num_correct_perturb_list.append(num_correct_perturb_arr)
 dissimilarity_perturb_arr = np.load(folder_perturb + '/perturbations_dissimilarities.npy')
 dissimilarity_perturb_list.append(dissimilarity_perturb_arr)
 logit_diff_perturb_arr = np.load(folder_perturb + '/perturbations_logit_diff.npy')
 logit_diff_perturb_list.append(logit_diff_perturb_arr)
 prob_diff_perturb_arr = np.load(folder_perturb + '/perturbations_prob_diff.npy')
 prob_diff_perturb_list.append(prob_diff_perturb_arr)

num_correct_model_full = np.hstack((num_correct_model_list[i]) for i in range(1000))
dissimilarity_model_full = np.hstack((dissimilarity_model_list[i]) for i in range(1000))

num_correct_perturb_full = np.hstack((num_correct_perturb_list[i]) for i in range(1000))
dissimilarity_perturb_full = np.hstack((dissimilarity_perturb_list[i]) for i in range(1000))
logit_diff_perturb_full = np.hstack((logit_diff_perturb_list[i]) for i in range(1000))
prob_diff_perturb_full = np.hstack((prob_diff_perturb_list[i]) for i in range(1000))

np.save(os.path.join(OUTPUT_PERTURB_DIR, 'num_correct_model_full.npy'), num_correct_model_full)
np.save(os.path.join(OUTPUT_PERTURB_DIR, 'dissimilarity_model_full.npy'), dissimilarity_model_full)
np.save(os.path.join(OUTPUT_PERTURB_DIR, 'num_correct_perturb_full.npy'), num_correct_perturb_full)
np.save(os.path.join(OUTPUT_PERTURB_DIR, 'dissimilarity_perturb_full.npy'), dissimilarity_perturb_full)
np.save(os.path.join(OUTPUT_PERTURB_DIR, 'logit_diff_perturb_full.npy'), logit_diff_perturb_full)
np.save(os.path.join(OUTPUT_PERTURB_DIR, 'prob_diff_perturb_full.npy'), prob_diff_perturb_full)

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