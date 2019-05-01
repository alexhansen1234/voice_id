import tflearn
import data_util
import librosa
import librosa.display
import numpy as np
import os
import matplotlib.pyplot as plt
from tflearn.data_utils import pad_sequences

learning_rate = 0.001
n_mfcc=50
batch_size = 32

data_util.build_dirs()

if not os.path.isdir( os.path.join('LibriSpeech', 'train-clean') ):
    data_util.fetch_LibriSpeech(dir='.',
                                url_path='http://www.openslr.org/resources/12/train-clean-100.tar.gz')

    data_util.extract_data(src= os.path.join('LibriSpeech', 'train-clean-100'),
                           dest=os.path.join('LibriSpeech', 'train-clean'),
                           n_classes=5)

if not os.path.isfile( os.path.join('data_cache','train_data.npy') ) or not os.path.isfile( os.path.join('data_cache','test_data.npy') ):
    train, test = data_util.load_data(path=os.path.join('LibriSpeech','train-clean'), fraction=0.1, n_mfcc=n_mfcc)

    train_data, train_labels = train
    test_data, test_labels = test

    maxm = 0

    for x in train_data + test_data:
            maxm = max(maxm, x.shape[1])

    train_data = data_util.to_np_array(train_data, dim=(n_mfcc, maxm))
    test_data = data_util.to_np_array(test_data, dim=(n_mfcc, maxm))

    train_labels = data_util.generate_categorical_list(train_labels)
    test_labels = data_util.generate_categorical_list(test_labels)

    np.save( os.path.join('data_cache','train_data.npy'), train_data )
    np.save( os.path.join('data_cache','train_labels.npy'), train_labels )
    np.save( os.path.join('data_cache','test_data.npy'), test_data )
    np.save( os.path.join('data_cache','test_labels.npy'), test_labels )

else:
    train_data = np.load( os.path.join('data_cache','train_data.npy') )
    train_labels = np.load( os.path.join('data_cache','train_labels.npy') )
    test_data = np.load( os.path.join('data_cache','test_data.npy') )
    test_labels = np.load( os.path.join('data_cache','test_labels.npy') )

print(train_data.shape)

tflearn.init_graph(num_cores=4)

net = tflearn.input_data(shape=[None,n_mfcc,train_data.shape[2]])
#net = tflearn.embedding(net, input_dim=10000, output_dim=128)
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, 5, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate,
                         loss='categorical_crossentropy')

model = tflearn.DNN(net, tensorboard_verbose=3, checkpoint_path='./model/checkpoints/voice_id.tfl.checkpoint')

model.fit(train_data, train_labels, validation_set=(test_data, test_labels), show_metric=True,
            snapshot_epoch=True, snapshot_step=1000, batch_size=batch_size, n_epoch=100)

model.save('./model/voice_id.tfl')
