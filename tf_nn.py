import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("C:/Users/Dell/Desktop/Self Learning/R/lr.csv")

x_data = np.array(data[["x1","x2"]],np.float32)
y_data = np.array(data["y"],np.float32)

x = tf.placeholder('float', [None,2])
y = tf.placeholder('float')

n_data = 2
l1 = 3
l2 = 2
o = 1

tf.set_random_seed(1234)
w1 = tf.Variable(tf.random_normal([n_data,l1]),tf.float32)
b1 = tf.Variable(tf.random_normal([l1]),tf.float32)
w2 = tf.Variable(tf.random_normal([l1,l2]),tf.float32)
b2 = tf.Variable(tf.random_normal([l2]),tf.float32)
w3 = tf.Variable(tf.random_normal([l2,o]),tf.float32)
b3 = tf.Variable(tf.random_normal([o]),tf.float32)

z1 = tf.add(tf.matmul(x,w1),b1)
a1 = tf.nn.sigmoid(z1)
z2 = tf.add(tf.matmul(a1,w2),b2)
a2 = tf.nn.sigmoid(z2)
z3 = tf.add(tf.matmul(a2,w3),b3)
a3 = tf.nn.sigmoid(z3)

prediction = a3

loss = tf.reduce_mean(tf.square(prediction-y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    err = []
    for step in range(1,2000):
        _,c = sess.run([optimizer,loss], feed_dict = {x: x_data ,y: y_data})
        err.append(c)
        if step % 50 == 0:
            print("Error =", c)
    plt.plot(err)
    #yhat = np.where(sess.run(prediction) > 0.5 , 1, 0)




