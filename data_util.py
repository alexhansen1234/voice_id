import os
import soundfile as sf
import random
import numpy as np
import librosa

def to_np_array( data ):
    a = np.resize(data[0], (50,400))
    print(a.shape)
    return
    maxm = 0
    for x in data:
        maxm = max(maxm, x.shape[1])
    ret = np.array((len(data), data[0].shape[0], maxm))
    print(data[0], data[0].shape)
    for num, x in enumerate(data):
        ret[num] = np.resize(x, (data[0].shape[0], maxm))
    return ret

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
        os.path.mkdir('model')
    if not os.path.isdir('./model/checkpoints'):
        print("Creating directory model/checkpoints.")
        os.path.mkdir('model/checkpoints')
    if not os.path.isdir('data_cache'):
        print("Creating directory data_cache.")
        os.path.mkdir('data_cache')
