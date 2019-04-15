import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


n_samples, batch_size, num_steps = 1000, 100, 1000
X_data = np.random.uniform(-1, 1, (n_samples, 1))
y_data = X_data * (X_data * (X_data * (X_data * 20 - 10) - 15) + 10) + 1 + np.random.normal(0, 1, (n_samples, 1))

X = tf.placeholder(tf.float32, shape=(batch_size, 1))
y = tf.placeholder(tf.float32, shape=(batch_size, 1))
a4 = tf.Variable(tf.random_normal((1, 1)), dtype=tf.float32, name='a4')
a3 = tf.Variable(tf.random_normal((1, 1)), dtype=tf.float32, name='a3')
a2 = tf.Variable(tf.random_normal((1, 1)), dtype=tf.float32, name='a2')
a1 = tf.Variable(tf.random_normal((1, 1)), dtype=tf.float32, name='a1')
a0 = tf.Variable(tf.zeros((1,)), dtype=tf.float32, name='a0')

y_predict = tf.multiply(X, tf.multiply(X, tf.multiply(X, tf.multiply(X, a4) + a3) + a2) + a1) + a0
loss = tf.reduce_sum((y - y_predict) ** 2, name='loss')
optimizer = tf.train.GradientDescentOptimizer(0.005).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(num_steps + 1):
        indices = np.random.choice(n_samples, batch_size)
        X_batch, y_batch = X_data[indices], y_data[indices]
        sess.run([optimizer], feed_dict={X: X_batch, y: y_batch})
    val_a4, val_a3, val_a2, val_a1, val_a0 = sess.run([a4, a3, a2, a1, a0])


plt.plot(X_data, y_data, ".")
plt.pause(0.001)
X_data = np.sort(X_data, axis=0)

plt.title('h=%.2f, k=%.2f, l=%.2f, p=%.2f, b=%.2f' % (val_a4, val_a3, val_a2, val_a1, val_a0))
plt.plot(X_data, X_data * (X_data * (X_data * (X_data * val_a4 + val_a3) + val_a2) + val_a1) + val_a0, "-", linewidth=3)
plt.pause(0.1)

plt.ioff()
plt.show()
