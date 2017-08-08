import tensorflow as tf
import numpy as np

def generate_xor(length=1000):
    x = np.random.randint(0,2, size=(length,2))
    y = []
    for pair in x:
        y.append(int(np.logical_xor(pair[0],pair[1])))
    return x, np.array(y)

n_inputs = 2
n_hidden = n_inputs*4
n_outputs = 1


x = tf.placeholder(tf.float32, shape=[1,n_inputs])
y = tf.placeholder(tf.float32, [1, n_outputs])

W = tf.Variable(tf.random_uniform([n_inputs, n_hidden],-1,1))
b = tf.Variable(tf.zeros([n_hidden]))

W2 = tf.Variable(tf.random_uniform([n_hidden,n_outputs],-1,1))
b2 = tf.Variable(tf.zeros([n_outputs]))

def xor_model(data):
    x = data
    hidden_layer = tf.nn.sigmoid(tf.matmul(x,W)+b)
    output = tf.matmul(hidden_layer, W2)+b2
    return output

xor_nn = xor_model(x)
cost = tf.reduce_mean(tf.abs(xor_nn - y))
train_step = tf.train.AdagradOptimizer(0.05).minimize(cost)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

x_data,y_data = generate_xor(length=100000)
errors = []
count = 0
out_freq = 1000
for xor_in, xor_out in zip(x_data,y_data):
    _, err = sess.run([train_step, cost], feed_dict={x:xor_in.reshape(1,2), y:xor_out.reshape(1,n_outputs)})
    errors.append(err)
    count += 1

    if count == out_freq:
        tol = np.mean(errors[-out_freq:])
        print tol
        count = 0
        if tol < 0.005:
            break


n_tests = 100
correct = 0
count = 0
x_test, y_test = generate_xor(length=n_tests)
for xor_in, xor_out in zip(x_test, y_test):
    output = sess.run([xor_nn], feed_dict={x:xor_in.reshape(1,2)})[0]
    guess = int(output[0][0]+0.5)
    truth = int(xor_out)
    if guess == truth:
        correct += 1
    count += 1
    print "Model %d : Truth %d - Pass Rate %.2f" % (int(guess), int(xor_out), float(correct*100.0)/float(count))
