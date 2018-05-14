import tensorflow as tf
import numpy as np
import sys


batchsz = 20
IN = 10
hidden_1 = 15
hidden_2 = 5
OUT = 1


I = tf.placeholder(np.float64, shape=[batchsz, IN, 1], name="I")

H1 = tf.Variable(np.random.rand(hidden_1, IN))
B1 = tf.Variable(np.random.rand(hidden_1, 1))

H2 = tf.Variable(np.random.rand(hidden_2, hidden_1))
B2 = tf.Variable(np.random.rand(hidden_2, 1))

O  = tf.Variable(np.random.rand(OUT, hidden_2))
OB = tf.Variable(np.random.rand(OUT, 1))


""" Model """
predictions = []

tensors = tf.unstack(I)
for tensor in tensors:
    F1 = tf.nn.sigmoid(tf.add(tf.matmul(H1, tensor),  B1))
    F2 = tf.nn.sigmoid(tf.matmul(H2, F1) + B2)
    F3 = tf.nn.sigmoid(tf.matmul(O, F2)  + OB)

    predictions.append(F3)

OL = tf.placeholder(np.float64, shape=[batchsz, 1, 1], name="OL")
OL = tf.unstack(OL)



loss = tf.reduce_mean([tf.pow(pred - label, 2) for pred, label in zip(predictions, OL)])
optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

batches = 100
epochs  = 100

I_data = np.random.rand(batches, 20, 10, 1)
O_labels = np.random.rand(batches, 20, 1, 1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for e in range(epochs):
        mean_loss = 0
        for I_batch, O_batch in zip(I_data, O_labels):
            _, l = sess.run((optimizer, loss), {"I:0": I_batch, "OL:0": O_batch})

            mean_loss += l

        print("epoch: {}, loss: {} ".format(e, mean_loss / batches))

