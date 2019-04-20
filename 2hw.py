import numpy as np
import matplotlib.pyplot as plt
import os


# mnist
# name = 'mnist'
# df = pd.read_csv("mnist.csv")
# labels = df['label'].values
# data = df.values[:, 1:].reshape(10000, 28, 28)
# print(data.shape)
# plt.imshow(data[0])
# plt.show()


# notMNIST
name = 'notMNIST'

size = 0
for dir in os.listdir(name):
    size += len(os.listdir(f'{name}/{dir}'))
label = np.empty((size, 1), dtype=np.uint8)
data = np.empty((size, 28, 28))

iter = 0
for dir in os.listdir(name):
    for file_name in os.listdir(f'{name}/{dir}'):
        label[iter] = ord(dir[0])
        data[iter] = plt.imread(f'{name}/{dir}/{file_name}')
        iter += 1

print(data.shape)

plt.ion()

plt.figure(1)
im = data[0]
plt.imshow(data[0])
print(chr(label[0]))
plt.pause(0.0001)

plt.figure(2)
plt.imshow(data[14000])
print(chr(label[14000]))
plt.ioff()
plt.show()
