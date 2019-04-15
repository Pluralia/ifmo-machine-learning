from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# cancer
name = 'cancer'
df = pd.read_csv("cancer.csv")
labels = np.array([1 if l == 'M' else 0 for l in df['label'].values])
df = df.values
data = df[:, 1:]
data = data / np.max(data, axis=0)

# spam
# name = 'spam'
# df = pd.read_csv("spam.csv")
# df = df.values
# labels = df[:, -1]
# data = df[:, 0:-1]
# data = data / np.max(data, axis=0)


train_data, test_data, train_labels, test_labels = train_test_split(data, labels, train_size=0.8, test_size=0.2)
batch_size = 16
dims = data.shape[1]

X = tf.placeholder(tf.float32, shape=(None, dims))
Y = tf.placeholder(tf.float32, shape=(None, 1))
A = tf.Variable(tf.random_normal((dims, 1)), dtype=tf.float32, name='A')
b = tf.Variable(tf.random_normal((1,)), dtype=tf.float32, name='b')

f = 0.0001
Y_predict = tf.math.sigmoid(tf.matmul(X, A) + b) * (1 - 2 * f) + f
loss = -tf.reduce_sum((Y * tf.log(Y_predict) + (1 - Y) * tf.log(1 - Y_predict)), name='loss')

optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    curr_err = 10000
    err_list = []
    for i in range(100000):
        indices = np.random.choice(train_data.shape[0], batch_size)
        sess.run([optimizer], feed_dict={X: train_data[indices], Y: np.expand_dims(train_labels[indices], axis=1)})

        if i % 500 == 0:
            err = curr_err
            indices = np.random.choice(test_data.shape[0], 300)
            curr_err, = sess.run([loss], feed_dict={X: test_data[indices], Y: np.expand_dims(test_labels[indices], axis=1)})
            # curr_err, = sess.run([loss], feed_dict={X: test_data, Y: np.expand_dims(test_labels, axis=1)})
            print("loss: %.2f, %i" % (curr_err, i))
            err_list.append([i, curr_err])
            # if err < curr_err:
            #     break

err = np.array(err_list)
plt.plot(err[:, 0], err[:, 1])
plt.title(name)
plt.xlabel('Номер шага')
plt.ylabel('loss')
plt.show()
