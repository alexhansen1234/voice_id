import os
import tensorflow as tf
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

train_data = np.load('data_cache/train_data.npy')
test_data = np.load('data_cache/test_data.npy')
test_labels = np.load('data_cache/test_labels.npy')

tflearn.init_graph(num_cores=4)

net = tflearn.input_data(shape=[None,train_data[0].shape[0],train_data[0].shape[1]])
#net = tflearn.embedding(net, input_dim=10000, output_dim=128)
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, 5, activation='softmax')

net_weights = net.W
net_bias = net.b

net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate,
                         loss='categorical_crossentropy')

model = tflearn.DNN(net, tensorboard_verbose=3, checkpoint_path='./model/checkpoints/voice_id.tfl.checkpoint')
model.load('./model/voice_id.tfl')
results = model.predict(test_data)

# print('Network Weights and Biases')
#
# init = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     sess.run(init)
#     net_weights = net_weights.eval()
#     net_bias = net_bias.eval()
#     rec_weights = rec_weights.eval()
#     rec_bias = rec_bias.eval()
#
# print(net_weights.shape)
# print(net_weights)
# print(net_bias.shape)
# print(net_bias)
# print(rec_weights)

total_correct = 0
this_correct = 0
for i in range(0, len(results)):
    if np.argmax(results[i]) == np.argmax(test_labels[i]):
        total_correct += 1
        this_correct = 1
    else:
        this_correct = 0
    print(results[i], test_labels[i])

print("n=", len(results))
print("p=", total_correct/len(results))
