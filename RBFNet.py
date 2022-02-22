"""
    Module: RBF
    Author: ShaoHaozhou
    motto: Self-discipline, self-improvement, self-love
    Date: 2021/7/13
    Introduce: Regularization gaussian radial basis neural network
    介绍: 正则化高斯径向基神经网络(输入层-径向基层(隐藏层)-输出层)
"""

import numpy as np
from functools import partial
from sklearn.preprocessing import StandardScaler


class RBFNet(object):
    """
        args:
            alpha: 学习率
            delta: 正则率
            means: C的个数[隐藏层的神经元个数]
            dim: 维度
    """
    def __init__(self, alpha=0.01, delta=0.001, means=50, out=1):
        self.alpha = alpha
        self.delta = delta
        self.means = means
        self.w = self.w = np.random.randn(self.means + 1, out)
        self.c = np.random.randn(means, 1)
        self.d = np.random.randn(means, 1)
        self.dim = None

    def rbf(self, x, c, d):
        """
            径向基函数
        """
        return np.exp(-np.power((np.sum(x - c, axis=1) / d), 2))

    def __softmax(self, y):
        """
            Normalization of output called prob
        """
        prob = np.exp(y - np.max(y, axis=1, keepdims=True))
        return prob / np.sum(prob, axis=1, keepdims=True)


    def __forward(self, train_x):
        """
            向前传播
        """
        p = partial(self.rbf, train_x)
        z = np.array([p(c=self.c[i], d=self.d[i]) for i in range(self.c.shape[0])])
        z = np.c_[z.T, np.ones((train_x.shape[0], 1))]
        y = np.dot(z, self.w)
        prob = self.__softmax(y)
        return z, y, prob

    def forward(self, train_x, train_y):
        """
            向前传播
        """
        z, y, prob = self.__forward(train_x)
        e = -np.sum(np.log(prob[np.arange(train_y.shape[0]), train_y])) / train_y.shape[0]
        return z, y, prob, e

    def backward(self, z, y, prob, train_x, train_y):
        """
            向后传播
            əe/əs
            əe/əw
            əe/əc
            əe/əd
        """

        prob[np.arange(train_y.shape[0]), train_y] -= 1
        df = prob
        dw = np.dot(z.T, df)

        self.c = np.array([self.c[i]
                           - self.alpha * (1 / np.power(self.d[i], 2)
                                           * np.dot(np.dot(z[i], np.dot(df, self.w.T).T), train_x - self.c[i]))
                           + self.delta * self.c[i]
                           for i in range(self.c.shape[0])])

        self.d = np.array([self.d[i]
                           - self.alpha * (1 / self.d[i]
                                           * np.dot(np.dot(z[i], np.dot(df, self.w.T).T), z[:, i]))
                           + self.delta * self.d[i]
                           for i in range(self.d.shape[0])])

        self.w -= self.alpha * dw + self.delta * self.w

    def fit(self, train_x, train_y, repeat=1000):
        """
            训练函数(train)
        """

        for _ in range(1, repeat+1):
            z, y, prob, e = self.forward(train_x, train_y)
            self.backward(z, y, prob, train_x, train_y)
            print("iteration[{i}]: loss({loss}), accuracy:({accuracy})".format
                  (i=_, loss=e, accuracy=self.score(train_x, train_y)))

    def predict(self, test_x):
        """
            预测函数
        """
        _, __, prob = self.__forward(test_x)

        predicts = np.array([], dtype=int)

        for i in range(test_x.shape[0]):
            predicts = np.append(predicts, np.argmax(prob[i, :]))

        return predicts

    def score(self, test_x, test_y):
        """
            准确率函数
        """
        y = self.predict(test_x)
        return sum(y == test_y) / y.shape[0]

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


if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
	
    stdscaler = StandardScaler(with_mean=False)
	
    iris = load_iris()
    train_x, test_x, train_y, test_y = train_test_split(iris.data, iris.target)
#    train_x, test_x = [stdscaler.fit_transform(x) for x in [train_x, test_x]]
    
    import time
    t = time.time()
    net = RBFNet(alpha=0.0002, delta=0.01, means=40, out=3)
    net.fit(train_x, train_y, 1000)
    print("the test y_data:\n", test_y)
    print("the predict of test_y is :\n", net.predict(test_x))
    print("the accuracy of test is:\n", net.score(test_x, test_y))
    print("time had run:\n", time.time() - t)
    # train_x, train_y, test_x, test_y = initialize()
    # net = RBFNet(alpha=0.001, delta=0.001, means=20, out=10)
    # import time
    # t = time.time()
    # net.fit(train_x, train_y, 1000)
    # print("the accuracy of test is:\n", net.score(test_x, test_y))
    # print("time had run:\n", time.time() - t)
