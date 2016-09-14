import json
import numpy as np


class test_data(object):

    def __init__(self, n_data, causility):
        self.n_data = n_data  # the number of data
        self.time_step = 8
        self.initial_xy = np.random.random_sample((n_data, 2)).tolist()
        # self.initial_xy = (np.random.randint(
        #    100, 1000, (n_data, 2)) / 100).tolist()
        self.datas = []  # shape = (n_data, time_step, 2)
        self.causility = causility

    def xy_function(self, xy):
        y = np.nan_to_num(xy[1] * (3 - 3 * xy[1]))
        x = np.nan_to_num(xy[0] * (4 - 4 * xy[0] - 2 * self.causility *
                                   xy[1]))
        return [x, y]

    def generate_data(self):
        for i in xrange(self.n_data):
            xy = self.initial_xy[i]
            data = [xy]
            for st in xrange(self.time_step - 1):
                xy = self.xy_function(xy)
                data.append(xy)
            self.datas.append(data)

    def save(self, filename):
        data = {'causility': self.causility,
                'data_number': self.n_data,
                'time_step': self.time_step,
                'datas': self.datas}
        f = open(filename, 'w')
        json.dump(data, f)
        f.close()


class sample_data(object):

    def __init__(self, n_data, time_step, causility):
        self.n_data = n_data
        self.time_step = time_step
        self.datas = []
        self.causility = causility

    def xy_function(self, xy):
        x = xy[0] * (3.9 - 3.9 * xy[0] - 2 * self.causility * xy[1])
        y = xy[1] * (3.7 - 3.7 * xy[1])
        return [x, y]

    def generate_data(self):
        i = 0
        while i < self.n_data:
            flag = True
            xy = [np.random.random(), np.random.random()]
            data = [xy]
            for step in xrange(1, self.time_step):
                xy = self.xy_function(xy)
                if abs(xy[0]) > 100000 or abs(xy[1]) > 100000:
                    flag = False
                    break
                data.append(xy)
            if flag:
                self.datas.append(data)
                i += 1

    def save(self, filename):
        data = {'causility': self.causility,
                'data_number': self.n_data,
                'time_step': self.time_step,
                'datas': self.datas}
        f = open(filename, 'w')
        json.dump(data, f)
        f.close()


def main():
    # data = test_data(3000, 1)
    # data.generate_data()
    # data.save("test_data")
    # print "the shape of datas {}".format(np.shape(data.datas))
    data = sample_data(6000, 32, 1)
    data.generate_data()
    data.save("test_data")
    print "the shape of datas {}".format(np.shape(data.datas))


if __name__ == '__main__':
    main()
