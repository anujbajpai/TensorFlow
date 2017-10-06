import tensorflow as tf

#Model Parameters
W = tf.Variable([.3], dtype = tf.float32)
b = tf.Variable([-.3], dtype = tf.float32)

#Model Input
x = tf.placeholder(tf.float32)

#Model Output
y = tf.placeholder(tf.float32)

#model
Linear_model = W*x+b

#loss function
loss = tf.reduce_sum(tf.square(Linear_model - y))

#optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

#training data
x_train = [1,2,3,4]
y_train = [0, -1, -2, -3]

#training loops; more the steps(loop) better is the training of model
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(1000):
    print(sess.run(train,{x:x_train, y:y_train}))

#evaluate training accuracy
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y:y_train})

print ("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
