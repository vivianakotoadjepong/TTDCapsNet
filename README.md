# TTDCapsNet
## Contents
1. [Introduction](#introduction)
2. [Usage](#usage)
3. [Results](#results)
4. [Contact Us](#contact-us)

   ## Introduction
We introduced a three hierachically layerd Capsule Network (CapsNet) structure named Tri Texton-Dense CapsNet (TTDCapsNet) with the aim of 
improving the classification of complex and medical images. The Texton layer contributes significantly to the better performance of the model.
Results from this study suggest that TTDCapsNet surpasses the baseline and demonstrates competitive performance when compared to state-of-the-art
CapsNet models. The state-of-the-art performance on fashion-MNIST, CIFAR-10, Breast Cancer, and Brain Tumor datasets, showcasing 94.90%, 89.09%, 95.01%, and 97.71% validation accuracies respectively, serves as evidence of the efficacy of TTDCapsNet. Below is the architecture of TTDCapsNet:
<p align="center">
<img src="figures/proposed architecture.png" width="700">
</p>

 ## Usage
Codes for training fashion-MNIST, CIFAR-10, Breast Cancer, and Brain Tumor Datasets is found in this repository. follow the procedure below:

**Step 1.
Install [Keras==2.1.2](https://github.com/fchollet/keras)
with [TensorFlow>=1.2](https://github.com/tensorflow/tensorflow) backend.**
```
pip install tensorflow-gpu
pip install keras==2.1.2

**Step 2. Clone the repository to local.**
```
https://github.com/vivianakotoadjepong/TTDCapsNet.git
```

**Step 3. Train the network.**  

Training TTDCapsNet on fashion-MNIST, and CIFAR-10 with default settings:
```
python T_MD_3LDCNet.py
```
You can check the well commented code for more settings.
```
For more settings, the code is well-commented and it should be easy to change the parameters looking at the comments. 

## Results

We perform various ablation experiments to prove the efficacy of TTDCapsNet. the following figures depicts  accuracy curves TTDCapsNet and the baseline model on CIFAR-10 and fashion-MNIST:
<p align="center">
<img src="figures/digitcapsAccuracy.png" width="350">
</p>
It can be seen that Level-3 and Merge-DigitCaps layer play a major role in the final performance. We also explore the relative effect of different levels of DigitCaps on the reconstruction outputs and experiment on the MNIST dataset by subtracting 0.2 from each digit one at a time in the 54D DigitCaps. It is observed (shown in the Figure below) that the effect on reconstructions decrease from the first level to the last level of capsules. This could be due to the fact that the first level DigitCaps activates very small area of an image and when perturbations are added to such smaller levels, it leads to an increased overall change in image (and vice versa). Also, it is observed that DigitCaps obtained from the concatenation of PrimaryCaps was most sensitive to this noise.
<p align="center">
<img src="figures/digitcaps_reconstructions.png" width="650">
</p>
