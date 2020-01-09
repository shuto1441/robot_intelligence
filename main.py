# coding: utf-8
import os
import sys
import numpy as np
from dataset.mnist import load_mnist
from net import Net
from noise import make_noise
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from learning import *

INPUT_SIZE = 784  # 画像サイズ 28*28
OUTPUT_SIZE = 10  # 0~9ß
noise_rate = [0.15,0.2,0.25] # ノイズ(0.0~1.0)
hidden_size = [30,60,90,120,150,180,210,240,270,300]  # デフォルト 50
hidden_num = [1,2,3]  # デフォルト 1
iters_num = 200000  # 繰り返しの回数 デフォルト 10000
batch_size = 100  # デフォルト 100
learning_rate = 0.1  # デフォルト 0.1


def main():

    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

    # 訓練データを絞ってみる
    # select_mask = np.random.choice(x_train.shape[0], 10000)
    # x_train = x_train[select_mask]
    # t_train = t_train[select_mask]
    for i in range(len(noise_rate)):
        # ランダムにノイズ処理
        if noise_rate[i] != 0:
            make_noise(x_train, noise_rate[i])
        noise=noise_rate[i]
        for j in range(len(hidden_size)):
            for k in range(len(hidden_num)):
                model = Net(INPUT_SIZE, hidden_size[j], hidden_num[k], OUTPUT_SIZE, learning_rate)
                train_size = x_train.shape[0]

                for l in range(1, iters_num+1):
                    # ランダムにバッチ処理
                    batch_mask = np.random.choice(train_size, batch_size)
                    x_batch = x_train[batch_mask]
                    t_batch = t_train[batch_mask]

                    # 学習の実行
                    learning(model,x_batch, t_batch)

                    # 途中の処理
                    if l % 100 == 0:
                        # 分類
                        divide(model,x_test, t_test)
                        print("\r  精度: {:.3f}, {:.1f} %   ".format(model.acc[-1], l/iters_num*100), end='')
                    if model.acc[-1]>=0.7 and np.var(model.acc[-150:]) < 0.0001 and l>=10000:
                        break


                # 損失関数の遷移をグラフ化
                graph(model,noise,hidden_num[k],hidden_size[j])

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n  [中断]")
