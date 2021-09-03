To run on Philly, create a yaml file in the philly_exp folder, outside of the ViT folder.

To test new models, add the models to ViT/ViT_LRP.py and ViT/ViT_new.py. dino_full() in ViT/ViT_LRP is an example of how to add a function for a model that you finetune yourself using already-pretrained weights.

**Finetuning DINO Small**

Philly: finetune_dino2.py

GCR: python3 ViT/finetune_dino2_gcr.py

**Evaluation and Minor Perturbation to Visualizations**

To evaluate a model and save the predictions and accuracies per class, run:
python3 ViT/evaluateAndStore.py

To evaluate, visualize, and observe the effect of introducing RandomCrop on DeiT, run:

  DeiT Base: python3 generate_visualizations_deit2.py

  DeiT Small: python3 generate_visualizations_deit2_small.py
  
  The RandomCrop perturbation is introduced in the above file here:
  
    transform2 = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomCrop(size=224, padding=4),  # added for our experiments
        transforms.ToTensor(),
        normalize,
    ])

  
  This code looks at multiple visualization techniques, but this list can be edited:
    
    methods = ['transformer_attribution', 'attn_last_layer', 'rollout', 'lrp', 'attn_gradcam']


**Visualization and Perturbation**

To generate visualizations (split into three groups) and perturbations, run the following on Philly:
myFinetune_[pos/neg]_vis_perturbation2_Philly.py 
with start and end parameters depending on how many classes you can run per GPU on Philly (ex. start = 0 and end = 50 for running the first 50 classes only).
You'll also have to edit the following lines to change the model:

1) Import statements 

    from ViT_LRP import vit_base_patch16_224 as vit_LRP
    
    from ViT_new import vit_base_patch16_224
    
    from ViT_explanation_generator import Baselines, LRP
    
    from modules.layers_ours import *
    

2) Loading the model

    model_new = vit_base_patch16_224(pretrained=True).cuda()
    
    baselines = Baselines(model_new)
    

3) Saving directory:

    OUTPUT_VIS_DIR = ROOT_DIR + "/output/ViT/ViT_Base Visualizations"
   
    OUTPUT_PERTURB_DIR = ROOT_DIR + "/output/ViT/ViT_Base Perturbations"
    
    
To aggregate the results across multiple GPUs, each with a subset of the 1000 classes, and calculate the AUC, run:
ViT/perturbationMetrics.py
You'll have to manually include the unperturbed (0% perturbation level) datapoint for the AUC calculation.
An example of this manual incorporation can be found in ViT/calculateAUC.py

**Segmentation**

To run segmentation, run: python3 ViT/imagenet_seg_eval_dino_gcr.py.
You'll also have to edit the following lines to change the model:

1) Import statements 

  from ViT_new import vit_large_patch16_224

  from ViT_LRP import vit_large_patch16_224 as vit_LRP

2) Loading the model

  model = vit_large_patch16_224(pretrained=True).cuda()

  baselines = Baselines(model)

3) Saving directory

  OUTPUT_DIR = ROOT_DIR + "/output/Segmentation/ViT_Large"






