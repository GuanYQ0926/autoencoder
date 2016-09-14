import matplotlib.pyplot as plt
import json
import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def load_model(filename):
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    n_layers = data['n_layers']
    weights = data['weights']
    biases = data['biases']
    return n_layers, weights, biases


def xy_function(xy, causility):
    x = xy[0] * (3.9 - 3.9 * xy[0] - 2 * causility * xy[1])
    y = xy[1] * (3.7 - 3.7 * xy[1])
    return [x, y]


def generate_test_data(n_sample, time_step, causility):
    # shape = (n_sample,32,2)
    samples = []
    i = 0
    while i < n_sample:
        flag = True
        xy = [np.random.random(), np.random.random()]
        sample = [xy]
        for step in xrange(1, time_step):
            xy = xy_function(xy, causility)
            if abs(xy[0]) > 100000 or abs(xy[1]) > 100000:
                flag = False
                break
            sample.append(xy)
        if flag:
            samples.append(sample)
            i += 1
    return samples


def reshape_calculate(samples):
    # (n,32,2)->(n,64) 64:(x0 y0...x31 y31)
    reshaped_samples = np.reshape(samples, (len(samples), -1))
    return reshaped_samples


def reshape_display(samples, time_step):
    samples = samples.tolist()
    # (n,64) 64:(x0 y0...x31 y31)->(n,64) 64:(x0...x31 y0...y32)
    reshaped_samples = []
    for i in range(len(samples)):
        sample = []
        for step in range(time_step):
            sample.append(samples[i][2 * step])
        for step in range(time_step):
            sample.append(samples[i][1 + 2 * step])
        reshaped_samples.append(sample)
    return reshaped_samples


def main():
    time_step = 32
    times = [i for i in range(time_step * 2)]
    raw_samples = generate_test_data(6, 32, 1)
    # load the model parameters
    n_layers, weights, biases = load_model("model_parameters")
    encoder_h1 = np.matrix(weights[0])
    encoder_h2 = np.matrix(weights[1])
    decoder_h1 = np.matrix(weights[2])
    decoder_h2 = np.matrix(weights[3])

    encoder_b1 = np.matrix(biases[0])
    encoder_b2 = np.matrix(biases[1])
    decoder_b1 = np.matrix(biases[2])
    decoder_b2 = np.matrix(biases[3])

    cal_samples = reshape_calculate(raw_samples)
    cal_samples = np.array(cal_samples)
    caled_samples = sigmoid(sigmoid(sigmoid(sigmoid(
        cal_samples * encoder_h1 + encoder_b1) *
        encoder_h2 + encoder_b2) * decoder_h1 + decoder_b1) *
        decoder_h2 + decoder_b2)
    dis_caled_samples = reshape_display(caled_samples, time_step)

    dis_samples = reshape_display(reshape_calculate(raw_samples), time_step)

    for i in range(6):
        plt.subplot(6, 1, i + 1)
        plt.plot(times, dis_samples[i], color='r', lw=2)
        plt.plot(times, dis_caled_samples[i], color='g', lw=1)
    plt.show()

if __name__ == '__main__':
    main()
