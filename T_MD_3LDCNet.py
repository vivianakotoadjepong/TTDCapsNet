"""
Keras implementation 
Code was developed using the following GitHub repositories:
- Xifeng Guo's CapsNet code (https://github.com/XifengGuo/CapsNet-Keras) 
- titu1994's DenseNet code (https://github.com/titu1994/DenseNet)
-Sai Samarth R Phay; Github: `https://github.com/ssrp/Multi-level-DCNet


       ... ...

Author: Vivian Akoto-Adjepong, E-mail: `vivianakotoadjepong@gmail.com`
"""

import numpy as np
import random as rn
import os
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
import tensorflow as tf
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
session_conf.gpu_options.allow_growth=True
from keras import backend as K
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
K.set_image_data_format('channels_last')

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from keras import layers, models, optimizers
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from utils import *
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask,TextonLayer3
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
import densenet
import numpy as np

import keras.models as km
from sklearn.metrics import classification_report, confusion_matrix,precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import time
from datetime import datetime
from sklearn.preprocessing import label_binarize
from scipy import interp
from scipy.interpolate import interp1d
from itertools import cycle
from keras import callbacks, optimizers

# from codecarbon import OfflineEmissionsTracker

# tracker = OfflineEmissionsTracker(country_iso_code="CAN")
# tracker.start()


def MultiLevelDCNet(input_shape, n_class, routings):
    """

    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    :param routings: number of routing iterations
    :return: Two Keras Models, the first one used for training, and the second one for evaluation.
    """

    x = layers.Input(shape=input_shape)
    
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1
      ########################### Texton layer3 ###########################

#     concat_axis = 1 if K.image_data_format() == 'channels_first' else -1
        
    ########################### Level 1 Capsules ###########################
    # Incorporating DenseNets - Creating a dense block with 8 layers having 32 filters and 32 growth rate.
    texton_output1 = TextonLayer3()(x)
    conv, nb_filter = densenet.DenseBlock(texton_output1, growth_rate=32, nb_layers=8, nb_filter=32)
    # Batch Normalization
    DenseBlockOutput = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(conv)


    # Creating Primary Capsules (Level 1)
    # Here PrimaryCapsConv2D is the Conv2D output which is used as the primary capsules by reshaping and squashing (squash activation).
    # primarycaps_1 (size: [None, num_capsule, dim_capsule]) is the "reshaped and sqashed output" which will be further passed to the dynamic routing protocol.
    primarycaps_1, PrimaryCapsConv2D = PrimaryCap(DenseBlockOutput, dim_capsule=8, n_channels=12, kernel_size=5, strides=2, padding='valid')

    # Applying ReLU Activation to primary capsules 
    conv = layers.Activation('relu')(PrimaryCapsConv2D)

    ########################### Level 2 Capsules ###########################
    # Incorporating DenseNets - Creating a dense block with 8 layers having 32 filters and 32 growth rate.
    texton_output2 = TextonLayer3()(conv)
    conv, nb_filter = densenet.DenseBlock(texton_output2, growth_rate=32, nb_layers=8, nb_filter=32)
    # Batch Normalization
    DenseBlockOutput = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(conv)

    # Creating Primary Capsules (Level 2)
    primarycaps_2, PrimaryCapsConv2D = PrimaryCap(DenseBlockOutput, dim_capsule=8, n_channels=12, kernel_size=5, strides=2, padding='valid')

    # Applying ReLU Activation to primary capsules 
    conv = layers.Activation('relu')(PrimaryCapsConv2D)

    ########################### Level 3 Capsules ###########################
    # Incorporating DenseNets - Creating a dense block with 8 layers having 32 filters and 32 growth rate.
    texton_output3 = TextonLayer3()(conv)
    conv, nb_filter = densenet.DenseBlock(texton_output3, growth_rate=32, nb_layers=8, nb_filter=32)
    # Batch Normalization
    DenseBlockOutput = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(conv)

    # Creating Primary Capsules (Level 3)
    primarycaps_3, PrimaryCapsConv2D = PrimaryCap(DenseBlockOutput, dim_capsule=8, n_channels=12, kernel_size=3, strides=2, padding='valid')

    # Merging Primary Capsules for the Merged DigitCaps (CapsuleLayer formed by combining all levels of primary capsules)
    mergedLayer = layers.merge([primarycaps_1,primarycaps_2,primarycaps_3], mode='concat', concat_axis=1)

    
    ########################### Separate DigitCaps Outputs (used for training) ###########################
    # Merged DigitCaps
    digitcaps_0 = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings,
                             name='digitcaps0')(mergedLayer)
    out_caps_0 = Length(name='capsnet_0')(digitcaps_0)

    # First Level DigitCaps
    digitcaps_1 = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings,
                             name='digitcaps1')(primarycaps_1)
    out_caps_1 = Length(name='capsnet_1')(digitcaps_1)

    # Second Level DigitCaps
    digitcaps_2 = CapsuleLayer(num_capsule=n_class, dim_capsule=12, routings=routings,
                             name='digitcaps2')(primarycaps_2)
    out_caps_2 = Length(name='capsnet_2')(digitcaps_2)

    # Third Level DigitCaps
    digitcaps_3 = CapsuleLayer(num_capsule=n_class, dim_capsule=10, routings=routings,
                             name='digitcaps3')(primarycaps_3)
    out_caps_3 = Length(name='capsnet_3')(digitcaps_3)

    ########################### Combined DigitCaps Output (used for evaluation) ###########################
    digitcaps = layers.merge([digitcaps_1,digitcaps_2,digitcaps_3, digitcaps_0], mode='concat', concat_axis=2,
                             name='digitcaps')
    out_caps = Length(name='capsnet')(digitcaps)

    # Reconstruction (decoder) network
    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer. For training
    masked = Mask()(digitcaps)  # Mask using the capsule with maximal length. For prediction

    # Shared Decoder model in training and prediction
    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(512, activation='relu', input_dim=int(digitcaps.shape[2]*n_class), name='zero_layer'))
    decoder.add(layers.Dense(512, activation='relu', name='one_layer'))
    decoderFinal = models.Sequential(name='decoderFinal')
    # Concatenating two layers
    decoderFinal.add(layers.Merge([decoder.get_layer('zero_layer'), decoder.get_layer('one_layer')], mode='concat'))
    decoderFinal.add(layers.Dense(1024, activation='relu'))
    decoderFinal.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoderFinal.add(layers.Reshape(input_shape, name='out_recon'))
    
    
    
    # Model for training
    train_model = models.Model([x, y], [out_caps, decoderFinal(masked_by_y)])

    # Model for evaluation (prediction)
    eval_model = models.Model(x, [out_caps, decoderFinal(masked)])

    return train_model, eval_model


def margin_loss(y_true, y_pred):
    """
    Margin loss, as introduced for Capsule Networks.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))

start = datetime.today()  #**********Start tracking the training time****  t0 = time.time()
def train(model, data, args):
    """
    Training 
    :param model: the 3-level DCNet model
    :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: The trained model
    """

    # unpacking the data
    (x_train, y_train), (x_test, y_test) = data
    row = x_train.shape[1]
    col = x_train.shape[2]
    channel = x_train.shape[3]

    # callbacks
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs', histogram_freq=int(args.debug))
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/trained_model.h5', monitor='val_capsnet_acc', save_best_only=True, 
                                           save_weights_only=True, 
                                           mode='max',
                                           verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))
    earlystopper = callbacks.EarlyStopping(monitor='val_capsnet_acc',patience=160, verbose=0)
    # compile the model
    # Notice the four separate losses (for separate backpropagations)
    
#     start = datetime.today()  #**********Start tracking the training time****  t0 = time.time()
    
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., args.lam_recon],
                  metrics={'capsnet': 'accuracy'})
#     MODEL_PATH = 'models/model-2d.h5'
    #model.load_weights('result/weights.h5')

    """
    # Training without data augmentation:
    model.fit([x_train, y_train], [y_train, x_train], batch_size=args.batch_size, epochs=args.epochs,
              validation_data=[[x_test, y_test], [y_test, x_test]], callbacks=[log, tb, checkpoint, lr_decay])
    """
   
    # Training with data augmentation
    def train_generator(x, y, batch_size, shift_fraction=0.):
        train_datagen = ImageDataGenerator(width_shift_range=shift_fraction,
                                           height_shift_range=shift_fraction)  # shift up to 2 pixel for MNIST
        generator = train_datagen.flow(x, y, batch_size=batch_size)
        while 1:
            x_batch, y_batch = generator.next()
            yield ([x_batch, y_batch], [y_batch, x_batch])

    # Training with data augmentation. If shift_fraction=0., also no augmentation.
    model.fit_generator(generator=train_generator(x_train, y_train, args.batch_size, args.shift_fraction),
                        steps_per_epoch=int(y_train.shape[0] / args.batch_size),
                        epochs=args.epochs,
                        validation_data=[[x_test, y_test], [y_test, x_test]],
                        callbacks=[log, tb, checkpoint, earlystopper, lr_decay])

#     # Save model weights
#     model.save_weights(args.save_dir + '/trained_model.h5')
#     print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

    plot_log(args.save_dir + '/log.csv', show=True)

    return model


def test(model, data, args):
    x_test, y_test = data
    print('Testing the model...')
    y_pred, x_recon = model.predict(x_test,batch_size=100)#===========================

    print('Test Accuracy: ', 100.0*np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/(1.0*y_test.shape[0]))

    img = combine_images(np.concatenate([x_test[:50],x_recon[:50]]))
    image = img * 255
    Image.fromarray(image.astype(np.uint8)).save(args.save_dir + "/real_and_recon.png")
    print()
    print('Reconstructed images are saved to %s/real_and_recon.png' % args.save_dir)
#     plt.savefig('real_and_recon.png', dpi=1200)
    plt.imshow(plt.imread(args.save_dir + "/real_and_recon.png"))
    plt.show()
    return image 

# def load_dataset():
#     # Load the dataset from Keras
#     # the data, shuffled and split between train and test sets
#     from keras.datasets import mnist
#     (x_train, y_train), (x_test, y_test) = mnist.load_data()

#     x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
#     x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
#     y_train = to_categorical(y_train.astype('float32'))
#     y_test = to_categorical(y_test.astype('float32'))
#     return (x_train, y_train), (x_test, y_test)

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

if __name__ == "__main__":
    import argparse
    from keras import callbacks

# ========================================================================================================================================    
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="DCNets on MNIST.")
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--n_class', default=10, type=int)
    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.9, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--lam_recon', default=0.512, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('--shift_fraction', default=0.1, type=float,
                        help="Fraction of pixels to shift at most in each direction.")
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('--digit', default=5, type=int,
                        help="Digit to manipulate")
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load data===========================================================================================================================
    (x_train, y_train), (x_test, y_test) = load_braintumor()
                                    #     load_breastcancer 2
                                    #     load_braintumor 4
                                     #  load_kvasir_v2 5
                                        #  load_covid19 4
#                     load_fashion_mnist 10

    # define model
    model, eval_model = MultiLevelDCNet(input_shape=x_train.shape[1:],
                                                  n_class=len(np.unique(np.argmax(y_train, 1))),
                                                  routings=args.routings)
    model.summary()
    
    # train or test
    if args.weights is not None: # init the model weights with provided one
        model.load_weights(args.weights)
    if not args.testing:
        train(model=model, data=((x_train, y_train), (x_test, y_test)), args=args)
    else:  # as long as weights are given, will run testing
        if args.weights is None:
            print('No weights are provided. Will test using random initialized weights.')
        test(model=eval_model, data=(x_test, y_test), args=args)
        
        
        
# ==================================================vivian========================================================

          #######################################Added#############################################
    #Code for ROC CUrve and Confusion Matrix.
    # Add a channel dimension.
    ###################################################SNNTOOLBOX##############################################

    ####################################################SNNTOOLBOX##################################################
#     train_model, eval_model , _ , _, _ = base_model(input_shape = x_shape, output_shape = y_shape)
    model, eval_model = MultiLevelDCNet(input_shape=x_train.shape[1:],
                                                  n_class=len(np.unique(np.argmax(y_train, 1))),
                                                  routings=args.routings)
    model.load_weights('result/trained_model.h5')
    

    ## Evaluating the Model
    y_pred = eval_model.predict(x_test)

    # Print prediction
    #print('pred_y',y_pred[:15])
    #NUM_CLASSES=[0,1,2,3,4]

    #print("y_test", y_test[:15])


    y_pred = np.array(y_pred[0])
    #y_pred=np.array(y_pred)


    
    round_pred = y_pred.argmax(axis=1)
    print("y_pred", y_pred)

    print("y_test", y_test)
    y_test=np.argmax(y_test, axis=1)
    print("y_test.axis=1", y_test)
    print("round_pred", round_pred)
    cnf_matrix = confusion_matrix(y_test, round_pred)
    #print(cnf_matrix)
# ====================================================================================================================
    n_class = 4
    kk = pd.DataFrame(cnf_matrix, range(n_class), range(n_class))
    sn.heatmap(kk, annot=True, annot_kws={'size':12}, fmt='g', cmap='Blues')
    plt.savefig('IConfusion_Matrix plot.png', dpi=1200)
    plt.show()
    

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    '''
    
    
    
    #fpr, tpr, _ = roc_curve(y_test, np.argmax(round_pred, axis=0))
    fpr, tpr, _ = roc_curve(y_test, round_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(11,8))
    lw = 10
    plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (auc = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('roc2.png')
    '''
    if n_class == 2:
        
        y_test_1=label_binarize(y_test, classes=[0,1,2])
        #print("in if")
    else:
        y_test_1=label_binarize(y_test, classes=[i for i in range(n_class)])
        #print("in else")
    # First aggregate all false positive rates
    #print("y_test",y_test_1.shape)
    #print(y_test_1)
    
    #print("y_pred.argmax(axis=1)",y_pred.argmax(axis=1))
    #hj=y_pred.argmax(axis=1)
    #hj=label_binarize(round_pred , classes=[i for i in range(NUM_CLASSES)])
    #print("y_pred_linarize",hj)
    
    for i in range(n_class):
        fpr[i], tpr[i],threshold = roc_curve(y_test_1[:,i],  y_pred[:,i])
        
        #print("fpr[i]",fpr[i])
        #print("tpr[i]",tpr[i])
        #print("threshold", threshold)
        roc_auc[i] = auc(fpr[i], tpr[i])

    #print("y_test.shape", y_test.shape)
    #print("hj.shape",hj.shape)
    #print("hj1.shape",hj1.shape)
    #print("y_test[:,1].shape", y_test[:,1].shape)
    
    #Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_1[:,1].ravel(), y_pred[:, 1].ravel())
    
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_class)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_class):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /=n_class

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    lw=1.5
    # Plot all ROC curves
    plt.figure()
    '''
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)
    '''
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue','purple', 'navy', 'teal','turquoise', 'yellow', 'green','red'])

    for i, color in zip(range(n_class), colors):
        #f=interp1d(fpr[i], tpr[i], kind='cubic')
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right",prop=dict(size=10), fontsize=10)
    plt.tight_layout()
    plt.savefig('IROC plot.png', dpi=1200)
    plt.show()
    
    
    ##################################PRECISION RECALL CURVE###############################################################  
    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_class):
        precision[i], recall[i], _ = precision_recall_curve(y_test_1[:, i],  y_pred[:,i])
        
        average_precision[i] = average_precision_score(y_test_1[:, i], y_pred[:,i])
      
    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ =precision_recall_curve(y_test_1[:,1].ravel(),y_pred[:, 1].ravel())

    average_precision["micro"] = average_precision_score(y_test_1[:,i],y_pred[:,i], average="micro")

    print('Average precision score, micro-averaged over all classes: {0:0.2f}'
          .format(average_precision["micro"]))

    # setup plot details
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal','red', 'green', 'purple', 'brown', 'cyan'])

    plt.figure(figsize=(7, 8))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    #lines.append(l)
    #labels.append('iso-f1 curves')
    #l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
    #lines.append(l)
    #labels.append('micro-average Precision-recall (area = {0:0.2f})'
    #              ''.format(average_precision["micro"]))

    for i, color in zip(range(n_class), colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                      ''.format(i, average_precision[i]))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(lines, labels, loc="lower left", prop=dict(size=10), fontsize=20)
    #(0, -.38)
    plt.savefig('IPrecision_Recall plot.png', dpi=1200)
    plt.show()

 
    
    #=========================ADDED BY FOR CSV FILE and others======================
    end = datetime.today()  #**********Stop tracking the training time****
    print("*"*10)
    print("Training Time Took: {}".format(end-start))
    print("*"*10)
    
#     tracker.stop()
#     print("Training time:", time.time()-t0)
#     from utils import plot_log
#     plot_log(save_dir + '/log.csv', show=True)
   #=========================ADDED FOR CSV FILE and others======================