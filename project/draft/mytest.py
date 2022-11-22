import os
import imageio

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# /home/mdsamiul/github_project/fair_classifier_ml/visualization2/output_0.1_90
images = list()
for y in range(97):
    images.append(imageio.imread(f'/home/mdsamiul/github_project/fair_classifier_ml/visualization/output_0.1_90/{y+1:08d}.jpg'))
imageio.mimsave("/home/mdsamiul/github_project/fair_classifier_ml/visualization/adv.gif", images)

