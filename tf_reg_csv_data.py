import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("C:/Users/Dell/Desktop/Self Learning/python/mreg.csv")

x_data = np.array(data[["x1","x2"]],np.float32)
y_data = np.array(data["y"],np.float32)

x = tf.placeholder('float', [None,2])
y = tf.placeholder('float')
with tf.device("/gpu:0"):
    b = tf.Variable(tf.random_normal([1,1]),np.float32)
    w = tf.Variable(tf.random_normal([2,1]),np.float32)
    model = tf.add(tf.matmul(x,w),b)
    loss = tf.reduce_mean(tf.square(y-model))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001).minimize(loss)

init = tf.global_variables_initializer()
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.50)

with tf.Session() as sess:
    sess.run(init)
    err = []
    for step in range(1,20000):
        _,c = sess.run([optimizer,loss], feed_dict = {x: x_data ,y: y_data})
        err.append(c)
        if step % 50 == 0:
            print("b =",sess.run(b),"w =",sess.run(w),"loss=", c)
    plt.plot(err)
    plt.show()
    yhat = np.add(np.dot(x_data,sess.run(w)),sess.run(b))
    res = y_data.reshape(10,1)-yhat   
    plt.plot(res,'ro')



