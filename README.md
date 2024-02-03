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
<img src="figures/dcnet.png" width="700">
</p>

 ## Usage
Codes for training fashion-MNIST, CIFAR-10, Breast Cancer, and Brain Tumor Datasets is found in this repository. follow the procedure below:

**Step 1.
Install [Keras==2.1.2](https://github.com/fchollet/keras)
with [TensorFlow>=1.2](https://github.com/tensorflow/tensorflow) backend.**
```
pip install tensorflow-gpu
pip install keras==2.1.2

