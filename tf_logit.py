import tensorflow as tf
import numpy as np
import pandas as pd
sess = tf.Session()
df = pd.read_csv("C:/Users/Yongyut/Desktop/Self Learning/R/LR.csv")

x_data = df[["x1","x2"]]
x_data = np.array(x_data, dtype="float64")
y_data = np.array(df["y"],dtype="float64")

b = tf.Variable(tf.zeros([1], name="Bias",dtype="float64"))
w = np.random.uniform(-1,1,[2,1])
W = tf.Variable(w, name="Weights")

yhat = tf.sigmoid(tf.matmul(x_data,W)+b)

ls = -y_data*tf.log(yhat)-(1-y_data)*tf.log(yhat)
loss = tf.reduce_mean(ls)
optimizer = tf.train.GradientDescentOptimizer(0.001)
train = optimizer.minimize(loss)

sess.run(tf.global_variables_initializer())

for step in range(10000):
  sess.run(train)
  if (step % 20 == 0):
     print("iteration", step, "theta0", sess.run(b),"theta 1,2", sess.run(W),"\n")

pprint(yhat)
