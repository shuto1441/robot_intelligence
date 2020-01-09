# coding: utf-8
import numpy as np

# 教師データがone-hotベクトルの場合、正解ラベルのインデックスに変換
def acurate_label(y, t):
    if t.size == y.size:
        return t.argmax(axis=1)
    else:
        return t

# softmax関数
def softmax(x):
    if x.ndim == 2:
        x -= x.max(axis=1, keepdims=True)
        x = np.exp(x)
        out = x / x.sum(axis=1, keepdims=True)
    elif x.ndim == 1:
        x -= np.max(x)
        out = np.exp(x) / np.sum(np.exp(x))
    return out

# 交差エントロピー誤差
def cross_entropy_error(y, t):
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)

        t = acurate_label(y, t)

        batch_size = y.shape[0]
        out = -1*np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
        return out

def sigmoid(x):
        out = 1 / (1 + np.exp(-x))
        return out
