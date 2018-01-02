import os
import sys
import glob
from fractions import Fraction   
import configparser
import math

import numpy
import scipy.fftpack 

from details import *

import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# constants
# ------------------------------------------------------------------------------

# 出力ファイルにつけるプリフィックス
OUTPUT_FILE_PREFIX = 'plot_amplitude_response\\'

# 計測に使うバッファのサンプル数
IMPULSE_LENGTH = 2 ** 12

# 仮想的なサンプルレート
SAMPLE_RATE = 48000 

# ------------------------------------------------------------------------------
# implementation
# ------------------------------------------------------------------------------

def apply_fft(samples):
    '''
    samples を FFT した結果の振幅特性と位相特性を得る。
    振幅特性の単位は dB で、位相特性の単位は degree 。
    '''
    SAMPLES_LENGTH = samples.size
    SAMPLES_LENGTH_HALF = math.ceil(SAMPLES_LENGTH/2)
    F_SAMPLES = scipy.fftpack.fft(samples)[:SAMPLES_LENGTH_HALF]
    F_SAMPLES_AMPLITUDE = 20.0 * numpy.log10(numpy.abs(F_SAMPLES))
    F_SAMPLES_PHASE = numpy.angle(F_SAMPLES)
    return F_SAMPLES_AMPLITUDE, F_SAMPLES_PHASE

def query_freq_list(samples):
    '''
    samples を FFT した結果の各バンドの周波数リストを得る。
    単位は hZ 。
    '''
    SAMPLES_LENGTH = samples.size
    SAMPLES_LENGTH_HALF = math.ceil(samples.size/2)
    return scipy.fftpack.fftfreq(SAMPLES_LENGTH, 1.0/SAMPLE_RATE)[:SAMPLES_LENGTH_HALF]

def wrap_phase(phases):
    return numpy.arctan2(numpy.sin(phases), numpy.cos(phases))

def apply_butter_worth(samples, frequency, order):
    

def apply_linkwitz_riley(samples, frequency):
    pass

def apply_chebyshev_1st(samples, frequency, order):
    pass

def apply_chebyshev_2nd(samples, frequency, order):
    pass
    
# ------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    # インパルスとその FFT を取得
    INPULSE = numpy.zeros(IMPULSE_LENGTH)
    INPULSE[int(IMPULSE_LENGTH/2)] = 1.0
    INPULSE_FFT_AMPLITUDE, INPULSE_FFT_PHASE = apply_fft(INPULSE)

    # FFT 結果の周波数リストを取得
    FREQUENCY_LIST = query_freq_list(INPULSE)

    '''
    - チェビシフフィルタの使い方
        - scipy.signal.cheby1(order, ripple gain, cut-off freq.)
    - TODO 位相・振幅はフィルタなしの特性との差分にする
    '''

    # フィルタのインパルス応答とその FFT を取得
    RESPONSE = extract_high_band(INPULSE, SAMPLE_RATE, 1000)
    RESPONSE_FFT_AMPLITUDE, RESPONSE_FFT_PHASE = apply_fft(RESPONSE)

    # インパルス FFT とインパルス応答 FFT の差分を取る
    FFT_AMPLITUDE_DIFF = RESPONSE_FFT_AMPLITUDE - INPULSE_FFT_AMPLITUDE
    FFT_PHASE_DIFF = wrap_phase(RESPONSE_FFT_PHASE - INPULSE_FFT_PHASE)

    # 結果をプロット
    plt.subplot(2, 1, 1)
    plt.xscale('log')
    plt.grid(which='major',color='black',linestyle='-')
    plt.grid(which='minor',color='black',linestyle='-')
    plt.xlim(20, 20000)
    plt.ylim(-48, 3)
    plt.plot(FREQUENCY_LIST, FFT_AMPLITUDE_DIFF, linestyle='-')
    plt.subplot(2, 1, 2)
    plt.xscale('log')
    plt.grid(which='major',color='black',linestyle='-')
    plt.grid(which='minor',color='black',linestyle='-')
    plt.xlim(20, 20000)
    plt.ylim(-180, +180)
    plt.plot(FREQUENCY_LIST, numpy.degrees(FFT_PHASE_DIFF), linestyle='-')
    plt.show()

    # 正常終了
    exit(0)
