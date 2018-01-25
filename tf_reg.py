import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_data = np.array([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0])
np.random.seed(1234)
noise = np.random.normal(0, 1, 10) #mean sd size
y_data = 2*x_data+5*noise

x = tf.placeholder('float')
y = tf.placeholder('float')

tf.set_random_seed(1234)
b = tf.Variable(tf.random_normal([1]),tf.float32)
w = tf.Variable(tf.random_normal([1]),tf.float32)

yhat = tf.multiply(x,w)+b
loss = tf.reduce_sum(tf.square(y-yhat))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    err = []
    for epoch in range(1,2000):
         _, c = sess.run([optimizer,loss],feed_dict={x:x_data,y:y_data})
         err.append(c)
         if epoch % 50 == 0:        
             print("b =",sess.run(b),"w =",sess.run(w),"error =", c)
    
    plt.plot(x_data, y_data, 'ro')
    y_pred = sess.run(w)*x_data+sess.run(b)
    plt.plot(x_data, y_pred)
    plt.show()
    plt.plot(err)
    plt.show()
    

