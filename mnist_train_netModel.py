import numpy as np

import saveNet
from BPNetwork import BPNetwork
from sklearn.preprocessing import StandardScaler


class Queue(object):
    def __init__(self):
        self.queue = []

    def put(self, index):
        self.queue.append(index)

    def get(self):
        return self.queue.pop()

    def empty(self):
        return True if len(self.queue) == 0 else False


def initialize():
    scaler = StandardScaler()
    train_x = scaler.fit_transform(np.genfromtxt(
        r'C:\Users\24034\Desktop\python-temp\datasets\mnist\train_img.csv', dtype=int, delimiter=','
    ))
    train_y = np.genfromtxt(
        r'C:\Users\24034\Desktop\python-temp\datasets\mnist\train_labels.csv', dtype=int, delimiter=','
    )
    test_x = scaler.fit_transform(np.genfromtxt(
        r'C:\Users\24034\Desktop\python-temp\datasets\mnist\test_img.csv', dtype=int, delimiter=','
    ))
    test_y = np.genfromtxt(
        r'C:\Users\24034\Desktop\python-temp\datasets\mnist\test_labels.csv', dtype=int, delimiter=','
    )
    # train_x, train_y, test_x, test_y = [None]*4
    return train_x, train_y, test_x, test_y


class NetModel(object):
    train_x, train_y, test_x, test_y = initialize()

    def __init__(self, queue):
        self.learningSpeed = None
        self.penaltyCoefficient = None
        self.activation = None
        self.levels = None
        self.hidden_dim = None
        self.output_dim = None
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None
        self.queue = queue
        self.train_x = NetModel.train_x
        self.train_y = NetModel.train_y
        self.test_x = NetModel.test_x
        self.test_y = NetModel.test_y

    def __update_net(self):
        self.net = BPNetwork(self.queue, self.learningSpeed, self.penaltyCoefficient, self.activation)
        self.net.set(self.levels, self.hidden_dim, self.output_dim)

    def net_setting(self, levels, hidden_dim, output_dim, learningSpeed=0.0000005, penaltyCoefficient=0.0001, activation="relu"):
        self.levels = levels
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learningSpeed = learningSpeed
        self.penaltyCoefficient = penaltyCoefficient
        self.activation = activation

    def fit(self, repeat=10):
        self.__update_net()
        self.net.fit(self.train_x, self.train_y, repeat)

    def predict(self):
        return self.net.predict(self.test_x)

    def score(self):
        return self.net.score(self.test_x, self.test_y)

    def save(self, path):
        saveNet.save_model(self.net, path)


if __name__ == '__main__':
    model = NetModel(Queue())
    model.net_setting(1, [300], 10, learningSpeed=0.00001, penaltyCoefficient=0.0001, activation="sigmoid")
    import time
    t = time.time()
    model.fit(repeat=10)
    print("the accuracy of test is:\n", model.score())
    print("time had run:\n", time.time() - t)
    model.save(r"C:\Users\HP\Desktop\Python深入学习\机器学习_Python\datasets\mnist\model")
