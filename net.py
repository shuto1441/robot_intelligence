# coding: utf-8
import numpy as np
import layers

class Net:

    def __init__(self, input_size, hidden_size, hidden_num, output_size, lr):

        self.lr = lr
        self.loss_list  = []
        self.acc = [0.0]

        # 重みとバイアスの初期化
        W1 = 0.01 * np.random.randn(input_size, hidden_size)
        b1 = np.zeros(hidden_size)
        W2 = 0.01 * np.random.randn(hidden_size, output_size)
        b2 = np.zeros(output_size)

        # レイヤの生成
        self.layer_list = [
            layers.Affine(W1, b1),
            layers.Sigmoid(),
            layers.Affine(W2, b2),
        ]

        # 中間層の追加
        for _ in range(hidden_num-1):
            W = 0.01 * np.random.randn(hidden_size, hidden_size)
            b = np.zeros(hidden_size)

            self.layer_list.insert(-1, layers.Affine(W, b))
            self.layer_list.insert(-1, layers.Sigmoid())

        # 出力層
        self.loss_layer = layers.SoftmaxWithLoss()
        self.optimizer = layers.SGD()

        # 全ての重みと勾配をリストにまとめる
        self.params = []
        self.grads  = []
        for layer in self.layer_list:
            self.params += layer.params
            self.grads  += layer.grads

    def predict(self, x):
        for layer in self.layer_list:
            x = layer.forward(x)
        return x

    # 損失関数を算出
    def forward(self, x, t):
        score = self.predict(x)
        loss  = self.loss_layer.forward(score, t)
        #return loss

    # 損失関数を小さくするよう、勾配に基づいて重みを更新
    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layer_list):
            dout = layer.backward(dout)
        return dout

    # 学習
    def learn(self, x_batch, t_batch):
        self.forward(x_batch, t_batch)
        self.backward()
        self.optimizer.update(self.params, self.grads, self.lr)

    # 分類
    def classify(self, x_batch, t_batch):
        y = self.predict(x_batch)
        y = np.argmax(y, axis=1)
        if t_batch.ndim != 1:
            t_batch = np.argmax(t_batch, axis=1)
        # t_batch = layers.onehot2label(x_batch, t_batch)

        self.acc.append(np.sum(y == t_batch) / float(x_batch.shape[0]))
        # self.corr_num = 0
        # score = self.predict(x_batch)
        # batch_size = score.shape[0]

        # for i in range(batch_size):
        #     corr_cls = t_batch[i].argmax()
        #     pred_cls = score[i].argmax()

        #     if pred_cls == corr_cls:
        #         self.corr_num += 1

        # self.acc_list.append(self.corr_num/batch_size)
