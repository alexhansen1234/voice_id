import os
import tflearn
import data_util
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from tflearn.data_utils import pad_sequences

learning_rate = 0.001
n_mfcc=50
batch_size = 32

if not os.path.isfile('train_data.npy') or not os.path.isfile('test_labels.npy')
    train_data = np.load('train_data.npy')
    test_data = np.load('test_data.npy')
    test_labels = np.load('test_labels.npy')


tflearn.init_graph(num_cores=4)

tflearn.init_graph(num_cores=4)

net = tflearn.input_data(shape=[None,n_mfcc,train_data[0].shape[1]])
#net = tflearn.embedding(net, input_dim=10000, output_dim=128)
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, 5, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate,
                         loss='categorical_crossentropy')

model = tflearn.DNN(net, tensorboard_verbose=3, checkpoint_path='./model/checkpoints/voice_id.tfl.checkpoint')
model.load('./model/voice_id.tfl')
results = model.predict(test_data)

for i in range(0, len(results)):
    print(results[i], test_labels[i])
