# coding: utf-8
import numpy as np
import os
import sys
# rate(0.0~1.0)で指定した割合だけランダムに書き換え
def make_noise(x_batch, noise_rate):
    print("  ノイズ処理中・・・", end='')
    sys.stdout.flush()
    noise_num = int(len(x_batch[0])*noise_rate)
    idx_list = list(range(noise_num))

    for vec in x_batch:
        rand_int = np.random.choice(idx_list, noise_num, replace=False)  # 重複なし
        for i in rand_int:
            vec[i] = np.random.rand()
    print("終了")
