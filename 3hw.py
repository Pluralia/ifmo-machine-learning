from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def get_activation(name):
    if name == 'sigmoid':
        act = tf.nn.sigmoid
    elif name == 'tanh':
        act = tf.nn.tanh
    elif name == 'ReLu':
        act = tf.nn.relu
    return act


# mnist
name = 'mnist'
df = pd.read_csv("mnist.csv")
labels = df['label'].values
data = df.values[:, 1:].reshape(10000, 28, 28)

labels = np.eye(10)[labels]
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, train_size=0.8, test_size=0.2)

acivation_name = 'sigmoid'
# acivation_name = 'tanh'
# acivation_name = 'ReLu'
activation = get_activation(acivation_name)


def conv_layer(input, size_in, size_out):
    w = tf.Variable(tf.truncated_normal([3, 3, size_in, size_out], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[size_out]))
    conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
    return conv + b


def fc_layer(input, size_in, size_out):
    w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[size_out]))
    return tf.matmul(input, w) + b


X = tf.placeholder(tf.float32, shape=[None, 28, 28])
x_image = tf.reshape(X, [-1, 28, 28, 1])
Y = tf.placeholder(tf.float32, shape=[None, 10])


conv1 = activation(conv_layer(x_image, 1, 8))
conv2 = activation(conv_layer(conv1, 8, 8))
conv_out = activation(conv_layer(conv2, 8, 8))

flattened = tf.reshape(conv_out, [-1, 28 * 28 * 8])

fc1 = activation(fc_layer(flattened, 28 * 28 * 8, 64))
fc2 = activation(fc_layer(fc1, 64, 64))
logits = fc_layer(fc2, 64, 10)

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
            acc = (acc0 + acc1) / 2
            print("accuracy: %.2f, %i" % (acc, i))
            acc_list.append([i, acc])


plt.ion()
plt.figure(0)
err = np.array(err_list)
plt.plot(err[:, 0], err[:, 1])
plt.title(name + '_' + acivation_name + '_loss')
plt.xlabel('Номер шага')
plt.ylabel('loss')

plt.figure(1)
acc = np.array(acc_list)
plt.plot(acc[:, 0], acc[:, 1])
plt.title(name + '_' + acivation_name + '_acc')
plt.xlabel('Номер шага')
plt.ylabel('accuracy')
plt.ioff()

plt.show()
