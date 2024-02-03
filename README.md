# TTDCapsNet

[![DOI](https://zenodo.org/badge/751786438.svg)](https://zenodo.org/doi/10.5281/zenodo.10613575)

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

We performed different ablation experiments to prove the efficiency of TTDCapsNet. the following figures depicts  accuracy curves for TTDCapsNet and the baseline model on CIFAR-10(first) and fashion-MNIST (second):
<p align="center">
<img src="figures/Cifar 10 Traning and Validation Accuracy.png" width="350">
</p>

<p align="center">
<img src="figures/fashion Mnist Traning and Validation Accuracy.png" width="350">
</p>
The precision recall (first) and Reciever Operating Characteristics Curve (second) values for CIFAR-10 using the proposed model can also be observed below with good values, representing better performance and robustness of the model.

<p align="center">
<img src="figures/cifar10 Precision_Recall plot.png" width="350">
</p>

<p align="center">
<img src="figures/cifar10 ROC plot.png" width="350">
</p>

## Contact Us
Please contact us on vivianakotoadjepong@gmail.com

