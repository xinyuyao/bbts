import numpy as np
from numpy import asarray
from numpy import save
data = np.loadtxt("mnist_train.csv", delimiter=",")
print(data.shape)

print("***************************** X: matrix storing the input features **************************\n")
x_data = asarray(data[:600,1:71])
print(x_data.shape)
print(x_data.dtype)
print(x_data)
# save('x.npy', x_data)
# print(np.load("x.npy"))
print("***************************** Y: matrix of one-hot encoded labels **************************\n")
y_data_label = np.array(data[:600,:1],dtype=int)
y_data_label_reshape = np.reshape(y_data_label, (600))
y_data = np.zeros((600,10))
y_data[np.arange(y_data_label_reshape.size),y_data_label_reshape] = 1
print(y_data.shape)
print(y_data.dtype)
print(y_data)
# save('y.npy', y_data)
# print(np.load("y.npy"))

print("***************************** W1 **************************\n")
w1 = np.random.uniform(0,1,size=(70,40))
print(w1.shape)
print(w1.dtype)
print(w1)
# save('w1.npy', w1)
# print(np.load("w1.npy"))
print("***************************** W2 **************************\n")
w2 = np.random.uniform(0,1,size=(40,10))
print(w2.shape)
print(w2.dtype)
print(w2)
save('w2.npy', w2)
# print(np.load("w2.npy"))





