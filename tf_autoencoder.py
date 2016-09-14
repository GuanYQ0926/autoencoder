from __future__ import division, print_function,\
    absolute_import
import tensorflow as tf
import numpy as np
import json


def save(filename, n_layers, weights, biases):
    data = {'n_layers': n_layers,
            'weights': weights,
            'biases': biases}
    f = open(filename, "w")
    json.dump(data, f)
    f.close()


def load(filename):
    f = open(filename, "r")
    data = json.load(f)
    f.close()

    causility = data['causility']
    data_number = data['data_number']
    time_step = data['time_step']
    datas = data['datas']  # datas.shape = (3000,25,2)
    return causility, data_number, time_step, datas


def data_reshape(datas, test_data_num):
    reshaped_datas = np.reshape(datas, (test_data_num, -1))
    return reshaped_datas  # shape = (3000, 50)


def get_random_data(reshaped_datas, test_data_num, batch_size):
    indices = np.random.randint(0, test_data_num, batch_size)
    data = []
    for i in indices:
        data.append(reshaped_datas[i])
    return data

causility, test_data_num, time_step, datas = load('test_data')
reshaped_datas = data_reshape(datas, test_data_num)
print ("load finished, the num is {}".format(test_data_num))

# Parameters
# test_data_num = 5000
learning_rate = 0.01
training_epochs = 30
batch_size = 32
display_step = 1

# Network Parameters
n_hidden_1 = time_step
n_hidden_2 = int(time_step/2)
n_input = 2 * time_step

# input
X = tf.placeholder("float", [None, n_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input]))
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([n_input]))
}


def encoder(x):  # build encoder
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2


def decoder(x):  # build decoder
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2

# construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# prediction
y_pred = decoder_op  # encoder -> decoder
y_true = X  # real value

# cost(quadratic cost) and optimizer
cost = tf.reduce_mean(tf.pow((y_true - y_pred), 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

# initialize the variables
init = tf.initialize_all_variables()

# launch the graph
with tf.Session() as sess:
    sess.run(init)
    total_batch = int(test_data_num / batch_size)
    # training
    for epoch in range(training_epochs):
        # loop over all batches
        for i in range(total_batch):
            batch_xs = get_random_data(reshaped_datas, test_data_num,
                                       batch_size)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
        if epoch % display_step == 0:
            print("Epoch:", "%02d" % (epoch + 1),
                  "cost=", "{:.9f}".format(c))
    print("Optimization Finished!")
    model_weights = [weights['encoder_h1'].eval().tolist(),
                     weights['encoder_h2'].eval().tolist(),
                     weights['decoder_h1'].eval().tolist(),
                     weights['decoder_h2'].eval().tolist()]
    model_biases = [biases['encoder_b1'].eval().tolist(),
                    biases['encoder_b2'].eval().tolist(),
                    biases['decoder_b1'].eval().tolist(),
                    biases['decoder_b2'].eval().tolist()]

save("model_parameters", 5, model_weights, model_biases)
