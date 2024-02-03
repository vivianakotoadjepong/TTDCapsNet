"""
This file contains code for some utility functions, such as code for saving CSV logs.
 vivian added codes for loading datasets
Code credits to Xifeng Guo: `https://github.com/XifengGuo/CapsNet-Keras`
"""

import numpy as np
from matplotlib import pyplot as plt
import csv
import math

import numpy as np
from tensorflow.keras.utils import to_categorical
import pandas
import tensorflow as tf

# ========================================================================================================================================
num_classes = 5

def plot_log(filename, show=True):
    # load data
    keys = []
    values = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if keys == []:
                for key, value in row.items():
                    keys.append(key)
                    values.append(float(value))
                continue

            for _, value in row.items():
                values.append(float(value))

        values = np.reshape(values, newshape=(-1, len(keys)))
        values[:,0] += 1

    fig = plt.figure(figsize=(4,6))
    fig.subplots_adjust(top=0.95, bottom=0.05, right=0.95)
    fig.add_subplot(211)
    for i, key in enumerate(keys):
        if key.find('loss') >= 0 and not key.find('val') >= 0:  # training loss
            plt.plot(values[:, 0], values[:, i], label=key)
    plt.legend()
    plt.title('Training loss')

    fig.add_subplot(212)
    for i, key in enumerate(keys):
        if key.find('acc') >= 0:  # acc
            plt.plot(values[:, 0], values[:, i], label=key)
    plt.legend()
    plt.title('Training and validation accuracy')

    fig.savefig('result/log.png')
    if show:
        plt.show()


def combine_images(generated_images, height=None, width=None):
    num = generated_images.shape[0]
    if width is None and height is None:
        width = int(math.sqrt(num))
        height = int(math.ceil(float(num)/width))
    elif width is not None and height is None:  # height not given
        height = int(math.ceil(float(num)/width))
    elif height is not None and width is None:  # width not given
        width = int(math.ceil(float(num)/height))

    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]
    return image

# ===================================================================================================================================
#                                                        DATASET                                   added by Vivian 
# ===================================================================================================================================

def load_mnist():
    # the data, shuffled and split between train and test sets
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))
    return (x_train, y_train), (x_test, y_test)

def load_fashion_mnist():
    # the data, shuffled and split between train and test sets
    fashion_mnist = tf.keras.datasets.fashion_mnist
#     from tf.keras.datasets import fashionmnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))
    return (x_train, y_train), (x_test, y_test)

def preprocess_input(x, data_format=None):
    """Preprocesses a tensor encoding a batch of images.
    # Arguments
        x: input Numpy tensor, 4D.
        data_format: data format of the image tensor.
    # Returns
        Preprocessed tensor.
    """
    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in {'channels_last', 'channels_first'}

    if data_format == 'channels_first':
        if x.ndim == 3:
            # 'RGB'->'BGR'
            x = x[::-1, ...]
            # Zero-center by mean pixel
            x[0, :, :] -= 103.939
            x[1, :, :] -= 116.779
            x[2, :, :] -= 123.68
        else:
            x = x[:, ::-1, ...]
            x[:, 0, :, :] -= 103.939
            x[:, 1, :, :] -= 116.779
            x[:, 2, :, :] -= 123.68
    else:
        # 'RGB'->'BGR'
        x = x[..., ::-1]
        # Zero-center by mean pixel
        x[..., 0] -= 103.939
        x[..., 1] -= 116.779
        x[..., 2] -= 123.68

    x *= 0.017 # scale values

    return x

def load_cifar10():
    # Load the dataset from Keras
    from keras.datasets import cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Preprocessing the dataset
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train= preprocess_input(x_train)
    x_test= preprocess_input(x_test)
    x_train = x_train.reshape(-1, 32, 32, 3).astype('float32') 
    x_test = x_test.reshape(-1, 32, 32, 3).astype('float32')
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))

    return (x_train, y_train), (x_test, y_test)

# def load_cifar100():
#     # the data, shuffled and split between train and test sets
#     cifar100 = tf.keras.datasets.cifar100
#     (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode="fine")
#     print('x_train shape:', x_train.shape)
#     print(x_train.shape[0], 'train samples')
#     print(x_test.shape[0], 'test samples')

#     # Convert class vectors to binary class matrices.
#     y_train = to_categorical(y_train, num_classes)
#     y_test = to_categorical(y_test, num_classes)
#     return (x_train, y_train), (x_test, y_test)


def load_cifar100():
    cifar100 = tf.keras.datasets.cifar100
#     from keras.datasets import cifar100
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()

    x_train = x_train.reshape(-1, 32, 32, 3).astype('float32') / 255.
    x_test = x_test.reshape(-1, 32, 32, 3).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))

    return (x_train, y_train), (x_test, y_test)


def load_dataset():
    # Load the dataset from Keras
    from keras.datasets import cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Preprocessing the dataset
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train= preprocess_input(x_train)
    x_test= preprocess_input(x_test)
    x_train = x_train.reshape(-1, 32, 32, 3).astype('float32') 
    x_test = x_test.reshape(-1, 32, 32, 3).astype('float32')
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))

    return (x_train, y_train), (x_test, y_test)

# =====================EYE DISEASE DATA LOADER===========================================
def load_eyedisease():
    import glob
    import numpy as np
    from matplotlib.pyplot import imread
    import os
    import cv2
    x_train1=[]
    y_train1=[]

    #eye disease  ##use this dataset for eye disease experiment
    data_dir='C:/Users/ASUS ROG/Desktop/MY DOCUMENTS/WORK Documents/DATASET/VIVIAN USE DATASET/EYE DISEASE used/train' 
    data_dir1='C:/Users/ASUS ROG/Desktop/MY DOCUMENTS/WORK Documents/DATASET/VIVIAN USE DATASET/EYE DISEASE used/val'
    for subdir, dirs, files in os.walk(data_dir):
        for file in files:
            hj=subdir
            hk=hj.split('\\')
            hl=int(hk[1])

            #print(hl)
            frame = cv2.imread(os.path.join(subdir, file))
            w=32
            h=32
            dim=(w,h)
            fr1=cv2.resize(frame,dim)
            y_train1.append(hl)
            x_train1.append(fr1)
    x_train=np.array(x_train1, dtype="float32") / 255.0    
    plo_1=np.array(y_train1, dtype="float32")
    y_train=to_categorical(plo_1.astype('float32'))
    x_test1=[]
    y_test1=[]
    for subdir, dirs, files in os.walk(data_dir1):
        for file in files:
            hjk=subdir
            hko=hjk.split('\\')
            hll=int(hko[1])
            w=32
            h=32
            dim=(w,h)
            #print(hl)
            framel = cv2.imread(os.path.join(subdir, file))
            fgl=cv2.resize(framel,dim)
            y_test1.append(hll)
            x_test1.append(fgl)
            #hj=subdi
    x_test=np.array(x_test1, dtype="float32") / 255.0
    jlo=np.array(y_test1, dtype="float32")
    y_test=to_categorical(jlo.astype('float32'))
    
    return (x_train, y_train), (x_test, y_test)

# =====================KVASIR-V2 DATA LOADER===========================================
def load_kvasir_v2():
    import glob
    import numpy as np
    from matplotlib.pyplot import imread
    import os
    import cv2
    x_train1=[]
    y_train1=[]

    #eye disease  ##use this dataset for eye disease experiment
    data_dir='C:/Users/ASUS ROG/Desktop/MY DOCUMENTS/WORK Documents/DATASET/VIVIAN USE DATASET/KVASIR-V2 used/train' 
    data_dir1='C:/Users/ASUS ROG/Desktop/MY DOCUMENTS/WORK Documents/DATASET/VIVIAN USE DATASET/KVASIR-V2 used/val'
    for subdir, dirs, files in os.walk(data_dir):
        for file in files:
            hj=subdir
            hk=hj.split('\\')
            hl=int(hk[1])

            #print(hl)
            frame = cv2.imread(os.path.join(subdir, file))
            w=64
            h=64
            dim=(w,h)
            fr1=cv2.resize(frame,dim)
            y_train1.append(hl)
            x_train1.append(fr1)
    x_train=np.array(x_train1, dtype="float32") / 255.0    
    plo_1=np.array(y_train1, dtype="float32")
    y_train=to_categorical(plo_1.astype('float32'))
    x_test1=[]
    y_test1=[]
    for subdir, dirs, files in os.walk(data_dir1):
        for file in files:
            hjk=subdir
            hko=hjk.split('\\')
            hll=int(hko[1])
            w=64
            h=64
            dim=(w,h)
            #print(hl)
            framel = cv2.imread(os.path.join(subdir, file))
            fgl=cv2.resize(framel,dim)
            y_test1.append(hll)
            x_test1.append(fgl)
            #hj=subdi
    x_test=np.array(x_test1, dtype="float32") / 255.0
    jlo=np.array(y_test1, dtype="float32")
    y_test=to_categorical(jlo.astype('float32'))
    
    return (x_train, y_train), (x_test, y_test)

# =====================covid 19 disease DATA LOADER===========================================
def load_covid19():
    import glob
    import numpy as np
    from matplotlib.pyplot import imread
    import os
    import cv2
    x_train1=[]
    y_train1=[]


    #covid19  ##use this dataset for covid19 disease experiment
    data_dir='C:/Users/ASUS ROG/Desktop/MY DOCUMENTS/WORK Documents/DATASET/VIVIAN USE DATASET/COVID 19 DATASET used/train' 
    data_dir1='C:/Users/ASUS ROG/Desktop/MY DOCUMENTS/WORK Documents/DATASET/VIVIAN USE DATASET/COVID 19 DATASET used/val'
    for subdir, dirs, files in os.walk(data_dir):
        for file in files:
            hj=subdir
            hk=hj.split('\\')
            hl=int(hk[1])

            #print(hl)
            frame = cv2.imread(os.path.join(subdir, file))
            w=32
            h=32
            dim=(w,h)
            fr1=cv2.resize(frame,dim)
            y_train1.append(hl)
            x_train1.append(fr1)
    x_train=np.array(x_train1, dtype="float32") / 255.0    
    plo_1=np.array(y_train1, dtype="float32")
    y_train=to_categorical(plo_1.astype('float32'))
    x_test1=[]
    y_test1=[]
    for subdir, dirs, files in os.walk(data_dir1):
        for file in files:
            hjk=subdir
            hko=hjk.split('\\')
            hll=int(hko[1])
            w=32
            h=32
            dim=(w,h)
            #print(hl)
            framel = cv2.imread(os.path.join(subdir, file))
            fgl=cv2.resize(framel,dim)
            y_test1.append(hll)
            x_test1.append(fgl)
            #hj=subdi
    x_test=np.array(x_test1, dtype="float32") / 255.0
    jlo=np.array(y_test1, dtype="float32")
    y_test=to_categorical(jlo.astype('float32'))
    
    return (x_train, y_train), (x_test, y_test)


# =====================covid 19 segmented disease DATA LOADER===========================================
def load_covid19_segmented():
    import glob
    import numpy as np
    from matplotlib.pyplot import imread
    import os
    import cv2
    x_train1=[]
    y_train1=[]


    #covid19  ##use this dataset for covid19 disease experiment
    data_dir='C:/Users/ASUS ROG/Desktop/MY DOCUMENTS/WORK Documents/DATASET/VIVIAN USE DATASET/COVID19 SEGMENTED used/train' 
    data_dir1='C:/Users/ASUS ROG/Desktop/MY DOCUMENTS/WORK Documents/DATASET/VIVIAN USE DATASET/COVID19 SEGMENTED used/val'
    for subdir, dirs, files in os.walk(data_dir):
        for file in files:
            hj=subdir
            hk=hj.split('\\')
            hl=int(hk[1])

            #print(hl)
            frame = cv2.imread(os.path.join(subdir, file))
            w=32
            h=32
            dim=(w,h)
            fr1=cv2.resize(frame,dim)
            y_train1.append(hl)
            x_train1.append(fr1)
    x_train=np.array(x_train1, dtype="float32") / 255.0    
    plo_1=np.array(y_train1, dtype="float32")
    y_train=to_categorical(plo_1.astype('float32'))
    x_test1=[]
    y_test1=[]
    for subdir, dirs, files in os.walk(data_dir1):
        for file in files:
            hjk=subdir
            hko=hjk.split('\\')
            hll=int(hko[1])
            w=32
            h=32
            dim=(w,h)
            #print(hl)
            framel = cv2.imread(os.path.join(subdir, file))
            fgl=cv2.resize(framel,dim)
            y_test1.append(hll)
            x_test1.append(fgl)
            #hj=subdi
    x_test=np.array(x_test1, dtype="float32") / 255.0
    jlo=np.array(y_test1, dtype="float32")
    y_test=to_categorical(jlo.astype('float32'))
    
    return (x_train, y_train), (x_test, y_test)



# ==============================monkeypox disease DATA LOADER=====================================
def load_monkeypox():
    import glob
    import numpy as np
    from matplotlib.pyplot import imread
    import os
    import cv2
    x_train1=[]
    y_train1=[]

    
    #monkeypox  ##use this dataset for monkeypox disease experiment
    data_dir='C:/Users/ASUS ROG/Desktop/MY DOCUMENTS/WORK Documents/DATASET/VIVIAN USE DATASET/MONKEYPOX DATASET used/train' 
    data_dir1='C:/Users/ASUS ROG/Desktop/MY DOCUMENTS/WORK Documents/DATASET/VIVIAN USE DATASET/MONKEYPOX DATASET used/val'
    for subdir, dirs, files in os.walk(data_dir):
        for file in files:
            hj=subdir
            hk=hj.split('\\')
            hl=int(hk[1])

            #print(hl)
            frame = cv2.imread(os.path.join(subdir, file))
            w=32
            h=32
            dim=(w,h)
            fr1=cv2.resize(frame,dim)
            y_train1.append(hl)
            x_train1.append(fr1)
    x_train=np.array(x_train1, dtype="float32") / 255.0    
    plo_1=np.array(y_train1, dtype="float32")
    y_train=to_categorical(plo_1.astype('float32'))
    x_test1=[]
    y_test1=[]
    for subdir, dirs, files in os.walk(data_dir1):
        for file in files:
            hjk=subdir
            hko=hjk.split('\\')
            hll=int(hko[1])
            w=32
            h=32
            dim=(w,h)
            #print(hl)
            framel = cv2.imread(os.path.join(subdir, file))
            fgl=cv2.resize(framel,dim)
            y_test1.append(hll)
            x_test1.append(fgl)
            #hj=subdi
    x_test=np.array(x_test1, dtype="float32") / 255.0
    jlo=np.array(y_test1, dtype="float32")
    y_test=to_categorical(jlo.astype('float32'))
    
    return (x_train, y_train), (x_test, y_test)


# ==============================monkeypox disease DATA LOADER=====================================
def load_braintumor():
    import glob
    import numpy as np
    from matplotlib.pyplot import imread
    import os
    import cv2
    x_train1=[]
    y_train1=[]

    
    #monkeypox  ##use this dataset for monkeypox disease experiment
    data_dir='C:/Users/ASUS ROG/Desktop/MY DOCUMENTS/WORK Documents/DATASET/VIVIAN USE DATASET/BRAIN TUMOR DATASET used/train' 
    data_dir1='C:/Users/ASUS ROG/Desktop/MY DOCUMENTS/WORK Documents/DATASET/VIVIAN USE DATASET/BRAIN TUMOR DATASET used/val'
    for subdir, dirs, files in os.walk(data_dir):
        for file in files:
            hj=subdir
            hk=hj.split('\\')
            hl=int(hk[1])

            #print(hl)
            frame = cv2.imread(os.path.join(subdir, file))
            w=32
            h=32
            dim=(w,h)
            fr1=cv2.resize(frame,dim)
            y_train1.append(hl)
            x_train1.append(fr1)
    x_train=np.array(x_train1, dtype="float32") / 255.0    
    plo_1=np.array(y_train1, dtype="float32")
    y_train=to_categorical(plo_1.astype('float32'))
    x_test1=[]
    y_test1=[]
    for subdir, dirs, files in os.walk(data_dir1):
        for file in files:
            hjk=subdir
            hko=hjk.split('\\')
            hll=int(hko[1])
            w=32
            h=32
            dim=(w,h)
            #print(hl)
            framel = cv2.imread(os.path.join(subdir, file))
            fgl=cv2.resize(framel,dim)
            y_test1.append(hll)
            x_test1.append(fgl)
            #hj=subdi
    x_test=np.array(x_test1, dtype="float32") / 255.0
    jlo=np.array(y_test1, dtype="float32")
    y_test=to_categorical(jlo.astype('float32'))
    
    return (x_train, y_train), (x_test, y_test)
# ===================================================================================================================================

# ==============================ODIR disease DATA LOADER=====================================
def load_ODIR():
    import glob
    import numpy as np
    from matplotlib.pyplot import imread
    import os
    import cv2
    x_train1=[]
    y_train1=[]

    
    #ODIR  ##use this dataset for monkeypox disease experiment
    
    data_dir='C:/Users/ASUS ROG/Desktop/MY DOCUMENTS/WORK Documents/DATASET/VIVIAN USE DATASET/ODIR used/train' 
    data_dir1='C:/Users/ASUS ROG/Desktop/MY DOCUMENTS/WORK Documents/DATASET/VIVIAN USE DATASET/ODIR used/val'
    for subdir, dirs, files in os.walk(data_dir):
        for file in files:
            hj=subdir
            hk=hj.split('\\')
            hl=int(hk[1])

            #print(hl)
            frame = cv2.imread(os.path.join(subdir, file))
            w=32
            h=32
            dim=(w,h)
            fr1=cv2.resize(frame,dim)
            y_train1.append(hl)
            x_train1.append(fr1)
    x_train=np.array(x_train1, dtype="float32") / 255.0    
    plo_1=np.array(y_train1, dtype="float32")
    y_train=to_categorical(plo_1.astype('float32'))
    x_test1=[]
    y_test1=[]
    for subdir, dirs, files in os.walk(data_dir1):
        for file in files:
            hjk=subdir
            hko=hjk.split('\\')
            hll=int(hko[1])
            w=32
            h=32
            dim=(w,h)
            #print(hl)
            framel = cv2.imread(os.path.join(subdir, file))
            fgl=cv2.resize(framel,dim)
            y_test1.append(hll)
            x_test1.append(fgl)
            #hj=subdi
    x_test=np.array(x_test1, dtype="float32") / 255.0
    jlo=np.array(y_test1, dtype="float32")
    y_test=to_categorical(jlo.astype('float32'))
    
    return (x_train, y_train), (x_test, y_test)
# ===================================================================================================================================

# ==============================OT disease DATA LOADER=====================================
def load_OCT():
    import glob
    import numpy as np
    from matplotlib.pyplot import imread
    import os
    import cv2
    x_train1=[]
    y_train1=[]

    
    #OCT  ##use this dataset for monkeypox disease experiment
    
    data_dir='C:/Users/ASUS ROG/Desktop/MY DOCUMENTS/WORK Documents/DATASET/VIVIAN USE DATASET/OCT DATASET used/train' 
    data_dir1='C:/Users/ASUS ROG/Desktop/MY DOCUMENTS/WORK Documents/DATASET/VIVIAN USE DATASET/OCT DATASET used/val'
    for subdir, dirs, files in os.walk(data_dir):
        for file in files:
            hj=subdir
            hk=hj.split('\\')
            hl=int(hk[1])

            #print(hl)
            frame = cv2.imread(os.path.join(subdir, file))
            w=32
            h=32
            dim=(w,h)
            fr1=cv2.resize(frame,dim)
            y_train1.append(hl)
            x_train1.append(fr1)
    x_train=np.array(x_train1, dtype="float32") / 255.0    
    plo_1=np.array(y_train1, dtype="float32")
    y_train=to_categorical(plo_1.astype('float32'))
    x_test1=[]
    y_test1=[]
    for subdir, dirs, files in os.walk(data_dir1):
        for file in files:
            hjk=subdir
            hko=hjk.split('\\')
            hll=int(hko[1])
            w=32
            h=32
            dim=(w,h)
            #print(hl)
            framel = cv2.imread(os.path.join(subdir, file))
            fgl=cv2.resize(framel,dim)
            y_test1.append(hll)
            x_test1.append(fgl)
            #hj=subdi
    x_test=np.array(x_test1, dtype="float32") / 255.0
    jlo=np.array(y_test1, dtype="float32")
    y_test=to_categorical(jlo.astype('float32'))
    
    return (x_train, y_train), (x_test, y_test)
# ===================================================================================================================================

# ============================== breast cancer DATA LOADER=====================================
def load_breastcancer():
    import glob
    import numpy as np
    from matplotlib.pyplot import imread
    import os
    import cv2
    x_train1=[]
    y_train1=[]

    
    #load_breastcancer  ##use this dataset for monkeypox disease experiment
    
    data_dir='C:/Users/ASUS ROG/Desktop/MY DOCUMENTS/WORK Documents/DATASET/VIVIAN USE DATASET/BREAST CANCER DATASET used/train' 
    data_dir1='C:/Users/ASUS ROG/Desktop/MY DOCUMENTS/WORK Documents/DATASET/VIVIAN USE DATASET/BREAST CANCER DATASET used/val'
    for subdir, dirs, files in os.walk(data_dir):
        for file in files:
            hj=subdir
            hk=hj.split('\\')
            hl=int(hk[1])

            #print(hl)
            frame = cv2.imread(os.path.join(subdir, file))
            w=32
            h=32
            dim=(w,h)
            fr1=cv2.resize(frame,dim)
            y_train1.append(hl)
            x_train1.append(fr1)
    x_train=np.array(x_train1, dtype="float32") / 255.0    
    plo_1=np.array(y_train1, dtype="float32")
    y_train=to_categorical(plo_1.astype('float32'))
    x_test1=[]
    y_test1=[]
    for subdir, dirs, files in os.walk(data_dir1):
        for file in files:
            hjk=subdir
            hko=hjk.split('\\')
            hll=int(hko[1])
            w=32
            h=32
            dim=(w,h)
            #print(hl)
            framel = cv2.imread(os.path.join(subdir, file))
            fgl=cv2.resize(framel,dim)
            y_test1.append(hll)
            x_test1.append(fgl)
            #hj=subdi
    x_test=np.array(x_test1, dtype="float32") / 255.0
    jlo=np.array(y_test1, dtype="float32")
    y_test=to_categorical(jlo.astype('float32'))
    
    return (x_train, y_train), (x_test, y_test)
# ===================================================================================================================================



# ============================== multi breast cancer DATA LOADER=====================================
def load_Breastcancer_Multi():
    import glob
    import numpy as np
    from matplotlib.pyplot import imread
    import os
    import cv2
    x_train1=[]
    y_train1=[]

    
    #load_breastcancer  ##use this dataset for monkeypox disease experiment
    
    data_dir='C:/Users/ASUS ROG/Desktop/MY DOCUMENTS/WORK Documents/DATASET/VIVIAN USE DATASET/BREAST CANCER DATASET MULTI used/train' 
    data_dir1='C:/Users/ASUS ROG/Desktop/MY DOCUMENTS/WORK Documents/DATASET/VIVIAN USE DATASET/BREAST CANCER DATASET MULTI used/val'
    for subdir, dirs, files in os.walk(data_dir):
        for file in files:
            hj=subdir
            hk=hj.split('\\')
            hl=int(hk[1])

            #print(hl)
            frame = cv2.imread(os.path.join(subdir, file))
            w=32
            h=32
            dim=(w,h)
            fr1=cv2.resize(frame,dim)
            y_train1.append(hl)
            x_train1.append(fr1)
    x_train=np.array(x_train1, dtype="float32") / 255.0    
    plo_1=np.array(y_train1, dtype="float32")
    y_train=to_categorical(plo_1.astype('float32'))
    x_test1=[]
    y_test1=[]
    for subdir, dirs, files in os.walk(data_dir1):
        for file in files:
            hjk=subdir
            hko=hjk.split('\\')
            hll=int(hko[1])
            w=32
            h=32
            dim=(w,h)
            #print(hl)
            framel = cv2.imread(os.path.join(subdir, file))
            fgl=cv2.resize(framel,dim)
            y_test1.append(hll)
            x_test1.append(fgl)
            #hj=subdi
    x_test=np.array(x_test1, dtype="float32") / 255.0
    jlo=np.array(y_test1, dtype="float32")
    y_test=to_categorical(jlo.astype('float32'))
    
    return (x_train, y_train), (x_test, y_test)
# ===================================================================================================================================



                                                    # DATASET loaders                            Added by Steve
# ===================================================================================================================================

# =====================corn pest DATA LOADER===========================================
def load_cornpest():
    import glob
    import numpy as np
    from matplotlib.pyplot import imread
    import os
    import cv2
    x_train1=[]
    y_train1=[]


    #ecorn pest  ##use this dataset for corn pest experiment
    data_dir="C:/Users/ASUS ROG/Desktop/MY DOCUMENTS/WORK Documents/DATASET/STEVE USE DATASET/dataset/plant pest dataset/CORN PEST DATASET used/train" 
    data_dir1="C:/Users/ASUS ROG/Desktop/MY DOCUMENTS/WORK Documents/DATASET/STEVE USE DATASET/dataset/plant pest dataset/CORN PEST DATASET used/val"
    for subdir, dirs, files in os.walk(data_dir):
        for file in files:
            hj=subdir
            hk=hj.split('\\')
            hl=int(hk[1])

            #print(hl)
            frame = cv2.imread(os.path.join(subdir, file))
            w=68
            h=68
            dim=(w,h)
            fr1=cv2.resize(frame,dim)
            y_train1.append(hl)
            x_train1.append(fr1)
    x_train=np.array(x_train1, dtype="float32") / 255.0    
    plo_1=np.array(y_train1, dtype="float32")
    y_train=to_categorical(plo_1.astype('float32'))
    x_test1=[]
    y_test1=[]
    for subdir, dirs, files in os.walk(data_dir1):
        for file in files:
            hjk=subdir
            hko=hjk.split('\\')
            hll=int(hko[1])
            w=68
            h=68
            dim=(w,h)
            #print(hl)
            framel = cv2.imread(os.path.join(subdir, file))
            fgl=cv2.resize(framel,dim)
            y_test1.append(hll)
            x_test1.append(fgl)
            #hj=subdi
    x_test=np.array(x_test1, dtype="float32") / 255.0
    jlo=np.array(y_test1, dtype="float32")
    y_test=to_categorical(jlo.astype('float32'))
    
    return (x_train, y_train), (x_test, y_test)

# ==============================tomato pest DATA LOADER=====================================
def load_tomatopest():
    import glob
    import numpy as np
    from matplotlib.pyplot import imread
    import os
    import cv2
    x_train1=[]
    y_train1=[]

    
    #covid  ##use this dataset for tomato pest experiment
    data_dir='C:/Users/ASUS ROG/Desktop/MY DOCUMENTS/WORK Documents/DATASET/STEVE USE DATASET/dataset/plant pest dataset/TOMATO PEST DATASET used/train' 
    data_dir1='C:/Users/ASUS ROG/Desktop/MY DOCUMENTS/WORK Documents/DATASET/STEVE USE DATASET/dataset/plant pest dataset/TOMATO PEST DATASET used/val'
    for subdir, dirs, files in os.walk(data_dir):
        for file in files:
            hj=subdir
            hk=hj.split('\\')
            hl=int(hk[1])

            #print(hl)
            frame = cv2.imread(os.path.join(subdir, file))
            w=68
            h=68
            dim=(w,h)
            fr1=cv2.resize(frame,dim)
            y_train1.append(hl)
            x_train1.append(fr1)
    x_train=np.array(x_train1, dtype="float32") / 255.0    
    plo_1=np.array(y_train1, dtype="float32")
    y_train=to_categorical(plo_1.astype('float32'))
    x_test1=[]
    y_test1=[]
    for subdir, dirs, files in os.walk(data_dir1):
        for file in files:
            hjk=subdir
            hko=hjk.split('\\')
            hll=int(hko[1])
            w=68
            h=68
            dim=(w,h)
            #print(hl)
            framel = cv2.imread(os.path.join(subdir, file))
            fgl=cv2.resize(framel,dim)
            y_test1.append(hll)
            x_test1.append(fgl)
            #hj=subdi
    x_test=np.array(x_test1, dtype="float32") / 255.0
    jlo=np.array(y_test1, dtype="float32")
    y_test=to_categorical(jlo.astype('float32'))
    
    return (x_train, y_train), (x_test, y_test)

# =====================corn disease DATA LOADER===========================================
def load_corndisease():
    import glob
    import numpy as np
    from matplotlib.pyplot import imread
    import os
    import cv2
    x_train1=[]
    y_train1=[]


    #corn disease  ##use this dataset for corn disease experiment
    data_dir='C:/Users/ASUS ROG/Desktop/MY DOCUMENTS/WORK Documents/DATASET/STEVE USE DATASET/dataset/plant disease dataset/CORN DISEASE DATASET used/train' 
    data_dir1='C:/Users/ASUS ROG/Desktop/MY DOCUMENTS/WORK Documents/DATASET/STEVE USE DATASET/dataset/plant disease dataset/CORN DISEASE DATASET used/val'
    for subdir, dirs, files in os.walk(data_dir):
        for file in files:
            hj=subdir
            hk=hj.split('\\')
            hl=int(hk[1])

            #print(hl)
            frame = cv2.imread(os.path.join(subdir, file))
            w=68
            h=68
            dim=(w,h)
            fr1=cv2.resize(frame,dim)
            y_train1.append(hl)
            x_train1.append(fr1)
    x_train=np.array(x_train1, dtype="float32") / 255.0    
    plo_1=np.array(y_train1, dtype="float32")
    y_train=to_categorical(plo_1.astype('float32'))
    x_test1=[]
    y_test1=[]
    for subdir, dirs, files in os.walk(data_dir1):
        for file in files:
            hjk=subdir
            hko=hjk.split('\\')
            hll=int(hko[1])
            w=68
            h=68
            dim=(w,h)
            #print(hl)
            framel = cv2.imread(os.path.join(subdir, file))
            fgl=cv2.resize(framel,dim)
            y_test1.append(hll)
            x_test1.append(fgl)
            #hj=subdi
    x_test=np.array(x_test1, dtype="float32") / 255.0
    jlo=np.array(y_test1, dtype="float32")
    y_test=to_categorical(jlo.astype('float32'))
    
    return (x_train, y_train), (x_test, y_test)

# ==============================tomato disease DATA LOADER=====================================
def load_tomatodisease():
    import glob
    import numpy as np
    from matplotlib.pyplot import imread
    import os
    import cv2
    x_train1=[]
    y_train1=[]

    
    #covid  ##use this dataset for tomato disease experiment
    data_dir='C:/Users/ASUS ROG/Desktop/MY DOCUMENTS/WORK Documents/DATASET/STEVE USE DATASET/dataset/plant disease dataset/TOMATO DISEASE DATASET used/train' 
    data_dir1='C:/Users/ASUS ROG/Desktop/MY DOCUMENTS/WORK Documents/DATASET/STEVE USE DATASET/dataset/plant disease dataset/TOMATO DISEASE DATASET used/val'
    for subdir, dirs, files in os.walk(data_dir):
        for file in files:
            hj=subdir
            hk=hj.split('\\')
            hl=int(hk[1])

            #print(hl)
            frame = cv2.imread(os.path.join(subdir, file))
            w=32
            h=32
            dim=(w,h)
            fr1=cv2.resize(frame,dim)
            y_train1.append(hl)
            x_train1.append(fr1)
    x_train=np.array(x_train1, dtype="float32") / 255.0    
    plo_1=np.array(y_train1, dtype="float32")
    y_train=to_categorical(plo_1.astype('float32'))
    x_test1=[]
    y_test1=[]
    for subdir, dirs, files in os.walk(data_dir1):
        for file in files:
            hjk=subdir
            hko=hjk.split('\\')
            hll=int(hko[1])
            w=32
            h=32
            dim=(w,h)
            #print(hl)
            framel = cv2.imread(os.path.join(subdir, file))
            fgl=cv2.resize(framel,dim)
            y_test1.append(hll)
            x_test1.append(fgl)
            #hj=subdi
    x_test=np.array(x_test1, dtype="float32") / 255.0
    jlo=np.array(y_test1, dtype="float32")
    y_test=to_categorical(jlo.astype('float32'))
    
    return (x_train, y_train), (x_test, y_test)
# ===================================================================================================================================

# ==============================rice disease DATA LOADER=====================================
def load_ricedisease():
    import glob
    import numpy as np
    from matplotlib.pyplot import imread
    import os
    import cv2
    x_train1=[]
    y_train1=[]

    
    #covid  ##use this dataset for tomato disease experiment
    data_dir='C:/Users/ASUS ROG/Desktop/MY DOCUMENTS/WORK Documents/DATASET/STEVE USE DATASET/dataset/plant disease dataset/RICE DISEASE DATASET used/train' 
    data_dir1='C:/Users/ASUS ROG/Desktop/MY DOCUMENTS/WORK Documents/DATASET/STEVE USE DATASET/dataset/plant disease dataset/RICE DISEASE DATASET used/val'
    for subdir, dirs, files in os.walk(data_dir):
        for file in files:
            hj=subdir
            hk=hj.split('\\')
            hl=int(hk[1])

            #print(hl)
            frame = cv2.imread(os.path.join(subdir, file))
            w=32
            h=32
            dim=(w,h)
            fr1=cv2.resize(frame,dim)
            y_train1.append(hl)
            x_train1.append(fr1)
    x_train=np.array(x_train1, dtype="float32") / 255.0    
    plo_1=np.array(y_train1, dtype="float32")
    y_train=to_categorical(plo_1.astype('float32'))
    x_test1=[]
    y_test1=[]
    for subdir, dirs, files in os.walk(data_dir1):
        for file in files:
            hjk=subdir
            hko=hjk.split('\\')
            hll=int(hko[1])
            w=32
            h=32
            dim=(w,h)
            #print(hl)
            framel = cv2.imread(os.path.join(subdir, file))
            fgl=cv2.resize(framel,dim)
            y_test1.append(hll)
            x_test1.append(fgl)
            #hj=subdi
    x_test=np.array(x_test1, dtype="float32") / 255.0
    jlo=np.array(y_test1, dtype="float32")
    y_test=to_categorical(jlo.astype('float32'))
    
    return (x_train, y_train), (x_test, y_test)


# =====================Cashew DATA LOADER===========================================
def load_cashew():
    import glob
    import numpy as np
    from matplotlib.pyplot import imread
    import os
    import cv2
    x_train1=[]
    y_train1=[]

    #eye disease  ##use this dataset for eye disease experiment
    data_dir='C:/Users/ASUS ROG/Desktop/MY DOCUMENTS/WORK Documents/DATASET/VIVIAN USE DATASET/CCMT Dataset/Cashew/training_set' 
    data_dir1='C:/Users/ASUS ROG/Desktop/MY DOCUMENTS/WORK Documents/DATASET/VIVIAN USE DATASET/CCMT Dataset/Cashew/test_set'
    for subdir, dirs, files in os.walk(data_dir):
        for file in files:
            hj=subdir
            hk=hj.split('\\')
            hl=int(hk[1])

            #print(hl)
            frame = cv2.imread(os.path.join(subdir, file))
            w=32
            h=32
            dim=(w,h)
            fr1=cv2.resize(frame,dim)
            y_train1.append(hl)
            x_train1.append(fr1)
    x_train=np.array(x_train1, dtype="float32") / 255.0    
    plo_1=np.array(y_train1, dtype="float32")
    y_train=to_categorical(plo_1.astype('float32'))
    x_test1=[]
    y_test1=[]
    for subdir, dirs, files in os.walk(data_dir1):
        for file in files:
            hjk=subdir
            hko=hjk.split('\\')
            hll=int(hko[1])
            w=32
            h=32
            dim=(w,h)
            #print(hl)
            framel = cv2.imread(os.path.join(subdir, file))
            fgl=cv2.resize(framel,dim)
            y_test1.append(hll)
            x_test1.append(fgl)
            #hj=subdi
    x_test=np.array(x_test1, dtype="float32") / 255.0
    jlo=np.array(y_test1, dtype="float32")
    y_test=to_categorical(jlo.astype('float32'))
    
    return (x_train, y_train), (x_test, y_test)

# =====================cassava DATA LOADER===========================================
def load_cassava():
    import glob
    import numpy as np
    from matplotlib.pyplot import imread
    import os
    import cv2
    x_train1=[]
    y_train1=[]


    #covid19  ##use this dataset for covid19 disease experiment
    data_dir='C:/Users/ASUS ROG/Desktop/MY DOCUMENTS/WORK Documents/DATASET/VIVIAN USE DATASET/CCMT Dataset/Cassava/training_set' 
    data_dir1='C:/Users/ASUS ROG/Desktop/MY DOCUMENTS/WORK Documents/DATASET/VIVIAN USE DATASET/CCMT Dataset/Cassava/test_set'
    for subdir, dirs, files in os.walk(data_dir):
        for file in files:
            hj=subdir
            hk=hj.split('\\')
            hl=int(hk[1])

            #print(hl)
            frame = cv2.imread(os.path.join(subdir, file))
            w=32
            h=32
            dim=(w,h)
            fr1=cv2.resize(frame,dim)
            y_train1.append(hl)
            x_train1.append(fr1)
    x_train=np.array(x_train1, dtype="float32") / 255.0    
    plo_1=np.array(y_train1, dtype="float32")
    y_train=to_categorical(plo_1.astype('float32'))
    x_test1=[]
    y_test1=[]
    for subdir, dirs, files in os.walk(data_dir1):
        for file in files:
            hjk=subdir
            hko=hjk.split('\\')
            hll=int(hko[1])
            w=32
            h=32
            dim=(w,h)
            #print(hl)
            framel = cv2.imread(os.path.join(subdir, file))
            fgl=cv2.resize(framel,dim)
            y_test1.append(hll)
            x_test1.append(fgl)
            #hj=subdi
    x_test=np.array(x_test1, dtype="float32") / 255.0
    jlo=np.array(y_test1, dtype="float32")
    y_test=to_categorical(jlo.astype('float32'))
    
    return (x_train, y_train), (x_test, y_test)

# ==============================maize DATA LOADER=====================================
def load_maize():
    import glob
    import numpy as np
    from matplotlib.pyplot import imread
    import os
    import cv2
    x_train1=[]
    y_train1=[]

    
    #monkeypox  ##use this dataset for monkeypox disease experiment
    data_dir='C:/Users/ASUS ROG/Desktop/MY DOCUMENTS/WORK Documents/DATASET/VIVIAN USE DATASET/CCMT Dataset/Maize/training_set' 
    data_dir1='C:/Users/ASUS ROG/Desktop/MY DOCUMENTS/WORK Documents/DATASET/VIVIAN USE DATASET/CCMT Dataset/Maize/test_set'
    for subdir, dirs, files in os.walk(data_dir):
        for file in files:
            hj=subdir
            hk=hj.split('\\')
            hl=int(hk[1])

            #print(hl)
            frame = cv2.imread(os.path.join(subdir, file))
            w=32
            h=32
            dim=(w,h)
            fr1=cv2.resize(frame,dim)
            y_train1.append(hl)
            x_train1.append(fr1)
    x_train=np.array(x_train1, dtype="float32") / 255.0    
    plo_1=np.array(y_train1, dtype="float32")
    y_train=to_categorical(plo_1.astype('float32'))
    x_test1=[]
    y_test1=[]
    for subdir, dirs, files in os.walk(data_dir1):
        for file in files:
            hjk=subdir
            hko=hjk.split('\\')
            hll=int(hko[1])
            w=32
            h=32
            dim=(w,h)
            #print(hl)
            framel = cv2.imread(os.path.join(subdir, file))
            fgl=cv2.resize(framel,dim)
            y_test1.append(hll)
            x_test1.append(fgl)
            #hj=subdi
    x_test=np.array(x_test1, dtype="float32") / 255.0
    jlo=np.array(y_test1, dtype="float32")
    y_test=to_categorical(jlo.astype('float32'))
    
    return (x_train, y_train), (x_test, y_test)


# ==============================tomato DATA LOADER=====================================
def load_tomato():
    import glob
    import numpy as np
    from matplotlib.pyplot import imread
    import os
    import cv2
    x_train1=[]
    y_train1=[]

    
    #monkeypox  ##use this dataset for monkeypox disease experiment
    data_dir='C:/Users/ASUS ROG/Desktop/MY DOCUMENTS/WORK Documents/DATASET/VIVIAN USE DATASET/CCMT Dataset/Tomato/training_set' 
    data_dir1='C:/Users/ASUS ROG/Desktop/MY DOCUMENTS/WORK Documents/DATASET/VIVIAN USE DATASET/CCMT Dataset/Tomato/test_set'
    for subdir, dirs, files in os.walk(data_dir):
        for file in files:
            hj=subdir
            hk=hj.split('\\')
            hl=int(hk[1])

            #print(hl)
            frame = cv2.imread(os.path.join(subdir, file))
            w=32
            h=32
            dim=(w,h)
            fr1=cv2.resize(frame,dim)
            y_train1.append(hl)
            x_train1.append(fr1)
    x_train=np.array(x_train1, dtype="float32") / 255.0    
    plo_1=np.array(y_train1, dtype="float32")
    y_train=to_categorical(plo_1.astype('float32'))
    x_test1=[]
    y_test1=[]
    for subdir, dirs, files in os.walk(data_dir1):
        for file in files:
            hjk=subdir
            hko=hjk.split('\\')
            hll=int(hko[1])
            w=32
            h=32
            dim=(w,h)
            #print(hl)
            framel = cv2.imread(os.path.join(subdir, file))
            fgl=cv2.resize(framel,dim)
            y_test1.append(hll)
            x_test1.append(fgl)
            #hj=subdi
    x_test=np.array(x_test1, dtype="float32") / 255.0
    jlo=np.array(y_test1, dtype="float32")
    y_test=to_categorical(jlo.astype('float32'))
    
    return (x_train, y_train), (x_test, y_test)
# ==================================================================================================================

#======================================================================================================================================
# if __name__=="__main__":
#     plot_log('result/log.csv')