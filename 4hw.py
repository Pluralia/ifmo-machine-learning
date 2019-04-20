from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt


# notMNIST
name = 'notMNIST'

size = 0
for dir in os.listdir(name):
    size += len(os.listdir(f'{name}/{dir}'))
label = np.empty((size,), dtype=np.uint8)
data = np.empty((size, 28, 28))

iter = 0
for dir in os.listdir(name):
    for file_name in os.listdir(f'{name}/{dir}'):
        label[iter] = ord(dir[0]) - ord('A')
        data[iter] = plt.imread(f'{name}/{dir}/{file_name}')
        iter += 1

labels = np.eye(10)[label]
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, train_size=0.8, test_size=0.2)


def conv_layer(input, size_in, size_out):
    w = tf.Variable(tf.truncated_normal([5, 5, size_in, size_out], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[size_out]))
    conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
    act = tf.nn.relu(conv + b)
    return tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def fc_layer(input, size_in, size_out):
    w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[size_out]))
    act = tf.matmul(input, w) + b
    return act


X = tf.placeholder(tf.float32, shape=[None, 28, 28])
x_image = tf.reshape(X, [-1, 28, 28, 1])
Y = tf.placeholder(tf.float32, shape=[None, 10])


conv1 = conv_layer(x_image, 1, 32)
conv_out = conv_layer(conv1, 32, 64)

flattened = tf.reshape(conv_out, [-1, 7 * 7 * 64])

fc1 = fc_layer(flattened, 7 * 7 * 64, 1024)
relu = tf.nn.relu(fc1)
logits = fc_layer(relu, 1024, 10)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))

optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


batch_size = 100
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    err_list = []
    acc_list = []
    for i in range(200 + 1):
        indices = np.random.choice(train_data.shape[0], batch_size)
        sess.run([optimizer], feed_dict={X: train_data[indices], Y: train_labels[indices]})

        if i % 5 == 0:
            indices = np.random.choice(test_data.shape[0], 1000)
            curr_err, = sess.run([loss], feed_dict={X: test_data[indices], Y: test_labels[indices]})
            # curr_err, = sess.run([loss], feed_dict={X: test_data, Y: np.expand_dims(test_labels, axis=1)})
            print("loss: %.2f, %i" % (curr_err, i))
            err_list.append([i, curr_err])

        if i % 20 == 0:
            acc0, = sess.run([accuracy], feed_dict={X: test_data[:1000], Y: test_labels[:1000]})
            acc1, = sess.run([accuracy], feed_dict={X: test_data[1000:2000], Y: test_labels[1000:2000]})
            acc2, = sess.run([accuracy], feed_dict={X: test_data[2000:3000], Y: test_labels[2000:3000]})
            acc = (acc0 + acc1 + acc2) / 3
            print("accuracy: %.2f, %i" % (acc, i))
            acc_list.append([i, acc])

plt.ion()
plt.figure(0)
err = np.array(err_list)
plt.plot(err[:, 0], err[:, 1])
plt.title(name + '_loss')
plt.xlabel('Номер шага')
plt.ylabel('loss')

plt.figure(1)
acc = np.array(acc_list)
plt.plot(acc[:, 0], acc[:, 1])
plt.title(name + '_acc')
plt.xlabel('Номер шага')
plt.ylabel('accuracy')
plt.ioff()

plt.show()
