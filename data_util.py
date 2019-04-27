import os
import soundfile as sf
import random
import numpy as np
import librosa

def to_np_array( data, dim ):
    ret = []
    for x in data:
        x,y = dim
        a = np.array(x, y - data.shape[1])
        b = np.concatenate(x, a, axis=1)
        ret.append(b)
    return np.array(ret)

def generate_class_dict( class_list ):
    class_dict = {}
    class_list = set(class_list)
    num_classes = len(class_list)

    for num, c in enumerate(class_list):
        d = class_dict
        if not c in d:
            d[c] = [0] * num_classes;
            d[c][num-1] = 1

    return (class_dict, num_classes)

def load_data( path, fraction=0.1, data_length=0, n_mfcc=20 ):
    train_data, train_labels = [],[]
    test_data, test_labels = [],[]

    for (dirpath, dirnames, filenames) in os.walk(path):
        for num, file in enumerate(filenames):
            y, sr = librosa.load( dirpath + '/' + file)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
            if( num <= fraction * len(filenames) ):
                test_data.append(mfcc)
                test_labels.append( int(os.path.basename(dirpath)) )
            else:
                train_data.append(mfcc)
                train_labels.append( int(os.path.basename(dirpath)) )

    return (train_data, train_labels), (test_data, test_labels)

def generate_categorical_list( class_list ):
    class_dict,_ = generate_class_dict(class_list)
    return np.array(list(map(lambda x: class_dict[x], class_list)))

def build_dirs():
    if not os.path.isdir('model'):
        print("Creating directory model.")
        os.mkdir('model')
    if not os.path.isdir('./model/checkpoints'):
        print("Creating directory model/checkpoints.")
        os.mkdir('model/checkpoints')
    if not os.path.isdir('data_cache'):
        print("Creating directory data_cache.")
        os.mkdir('data_cache')
