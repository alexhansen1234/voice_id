import os
import soundfile as sf
import random
import numpy as np
import librosa
import wget
import tarfile
import shutil

def to_np_array( data, dim ):
    ret = []
    for x in data:
        a = np.zeros((x.shape[0], dim[1] - x.shape[1]))
        b = np.concatenate((x, a), axis=1)
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
            y, sr = librosa.load( os.path.join(dirpath,file) )
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
            if( num <= fraction * len(filenames) ):
                test_data.append( np.array(mfcc) )
                test_labels.append( int(os.path.basename(dirpath)) )
            else:
                train_data.append( np.array(mfcc) )
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

def fetch_LibriSpeech(save_as='train-clean',
                      dir='.',
                      url_path='http://www.openslr.org/resources/12/train-clean-100.tar.gz'):
    if not os.path.isfile( os.path.join(dir,save_as) ):
        print('Fetching ' + url_path)
        wget.download(url_path, out=dir)
        print('')
    if not os.path.isfile(dir + '/' + save_as):
        tar_path = dir + '/' + url_path.rsplit('/', 1)[-1]
        print('Extracting ' + tar_path + ' to ' + dir + '/' + save_as)
        file = tarfile.open(tar_path)
        file.extractall(path=dir)
        file.close()
        print('Deleting ' + tar_path)
        os.remove(tar_path)

def get_size(path = '.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for file in filenames:
            fp = os.path.join(dirpath,file)
            total_size += os.path.getsize(fp)
    return total_size

def extract_data(src, dest, n_classes=5):
    dir_list = []
    for dirpath, dirnames, filenames in os.walk(src):
        for dir in dirnames:
            dir_list.append( (dirpath, dir, get_size( os.path.join(dirpath,dir) ) ) )
        break
    dir_list.sort(key=lambda x: x[2])
    if not os.path.isdir(dest):
        os.mkdir(dest)
    for x in range(0, n_classes):
        if not os.path.isdir( os.path.join(dest,dir_list[x][1]) ):
            os.mkdir( os.path.join(dest, dir_list[x][1] ))
        for dirpath, dirnames, filenames in os.walk( os.path.join(dir_list[x][0],dir_list[x][1]) ):
            for file in filenames:
                if not file.endswith('.txt'):
                    shutil.copyfile( os.path.join(dirpath,file), os.path.join(dest, dir_list[x][1], file) )
