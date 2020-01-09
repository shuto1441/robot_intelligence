# coding: utf-8
import matplotlib.pyplot as plt
import numpy as np
import layers
from net import Net


# グラフ化
def graph(self,noise_rate,hidden_num,hidden_size):
    plt.figure()
    x2 = np.arange(len(self.acc))
    plt.plot(x2, self.acc)
    plt.xlabel("try(1/100)")
    plt.ylabel("accuracy")
    plt.xlim(0, len(self.acc)-1)
    dirname="result/隠れ層{}/ノイズ{}%/".format(hidden_num,int(noise_rate*100))
    plt.savefig(dirname+"ノイズ:{}% 隠れ層:{}層ノード数:{}.png".format(noise_rate*100,hidden_num,hidden_size))
    #plt.show()


# 学習
def learning(self, x, t):
    self.forward(x, t)
    self.backward()
    self.optimizer.update(self.params, self.grads, self.lr)

# 分類
def divide(self, x, t):
    y = self.predict(x)
    y = np.argmax(y, axis=1)
    if t.ndim != 1:
        t = np.argmax(t,axis=1)
    # t_batch = layers.onehot2label(x_batch, t_batch)

    self.acc.append(np.sum(y == t) / float(x.shape[0]))
