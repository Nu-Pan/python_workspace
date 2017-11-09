import os
import sys

import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import *
from scipy import signal

# ------------------------------------------------------------------------------
# constants
# ------------------------------------------------------------------------------

NUMBER_OF_HARMONICS = 8
NUMBER_OF_SAMPLES_UNIT = 5
OVERSAMPLES_MULTIPLIER = 16

# ------------------------------------------------------------------------------
# internal functions
# ------------------------------------------------------------------------------

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    return a * b // gcd(a, b)

# ------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------

def print_usage():
    'このプログラムの使い方を表示'
    print('Usage1 : python extract_bass_lower.py <sample>.wav <sample_1>.wav ... <sample_N>.wav')
    print('Usage2 : python extract_bass_lower.py <direcyory path>')

if __name__ == '__main__':
    for counter in range(0, 100):
        # 倍音数から必要サンプル数を計算
        HAMONICS_LCM = 1
        for i in range(2, NUMBER_OF_HARMONICS):
            HAMONICS_LCM = lcm(HAMONICS_LCM, i)
        NUMBER_OF_SAMPLES = NUMBER_OF_SAMPLES_UNIT * NUMBER_OF_HARMONICS * OVERSAMPLES_MULTIPLIER
        # 波形を生成
        samples = np.zeros(NUMBER_OF_SAMPLES)
        for i in range(1, NUMBER_OF_HARMONICS+1):
            offset = rand() * 2 * np.pi
            #offset = 0
            x = np.linspace(offset , offset + 2 * np.pi * i, NUMBER_OF_SAMPLES)
            samples += np.sin(x)
        # '情報量'を計算して表示
        criteria = signal.argrelmax(samples)[0].size + signal.argrelmin(samples)[0].size
        #RMS = np.sqrt(np.mean(samples * samples))
        print('%3d, RMS = %f' % (counter, criteria,))
        plt.plot(0, criteria, "o")

    # 波形をファイル出力
    #sf.write('doutput.wav', samples, 48000, 'FLOAT')

    # 波形をプロット
    #plt.plot(samples)
    plt.show()

    # 正常終了
    exit(0)
