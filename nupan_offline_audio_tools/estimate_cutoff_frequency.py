import os
import sys
import glob
from fractions import Fraction   
import configparser
import math
from functools import partial

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
IMPULSE_LENGTH = 2 ** 15

# 仮想的なサンプルレート
SAMPLE_RATE = 48000 

# 使用するフィルタタイプ
FILTER_TYPE = 'high'

# 許容リップルゲイン
ALLOW_RIPPLE = 3

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

def apply_butter_worth(samples, frequency, order, is_zero_phase):
    b, a = signal.butter(order, normalize_frequency(frequency, SAMPLE_RATE), FILTER_TYPE)
    if is_zero_phase:
        return signal.filtfilt(b, a, samples)
    else:
        return signal.lfilter(b, a, samples)

def apply_chebyshev_1st(samples, frequency, order, is_zero_phase):
    b, a = signal.cheby1(order, ALLOW_RIPPLE, normalize_frequency(frequency, SAMPLE_RATE), FILTER_TYPE)
    if is_zero_phase:
        return signal.filtfilt(b, a, samples)
    else:
        return signal.lfilter(b, a, samples)
    
def create_impulse(length):
    INPULSE = numpy.zeros(length)
    INPULSE[int(length/2)] = 1.0
    return INPULSE

def calculate_frequency_response(filter_function, frequency, order):
    # インパルスとその FFT を取得
    INPULSE = create_impulse(IMPULSE_LENGTH)
    INPULSE_FFT_AMPLITUDE, INPULSE_FFT_PHASE = apply_fft(INPULSE)

    # FFT 結果の周波数リストを取得
    FREQUENCY_LIST = query_freq_list(INPULSE)

    # フィルタのインパルス応答とその FFT を取得
    RESPONSE = filter_function(INPULSE,  frequency, order)
    RESPONSE_FFT_AMPLITUDE, RESPONSE_FFT_PHASE = apply_fft(RESPONSE)

    # インパルス FFT とインパルス応答 FFT の差分を取る
    FFT_AMPLITUDE_DIFF = RESPONSE_FFT_AMPLITUDE - INPULSE_FFT_AMPLITUDE
    FFT_PHASE_DIFF = wrap_phase(RESPONSE_FFT_PHASE - INPULSE_FFT_PHASE)

    # 正常終了
    return FREQUENCY_LIST, FFT_AMPLITUDE_DIFF, FFT_PHASE_DIFF

def query_frequency_from_apmlitude_response(frequency_list, response_amplitude, query_gain):
    ARG = numpy.argmin(numpy.abs(response_amplitude - query_gain))
    return frequency_list[ARG], response_amplitude[ARG]

def query_amplitude_response_from_frequency(frequency_list, response_amplitude, query_frequency):
    ARG = numpy.argmin(numpy.abs(frequency_list - query_frequency))
    return frequency_list[ARG], response_amplitude[ARG]

def estimate_optimal_cutoff_freqency(filter_function, base_frequency, order, target_gain, target_frequency):
    FREQUENCY_LIST, RESPONSE_AMPLITUDE, RESPONSE_PHASE = calculate_frequency_response(filter_function, base_frequency, order)
    found_frequency, _ = query_frequency_from_apmlitude_response(FREQUENCY_LIST, RESPONSE_AMPLITUDE, target_gain)
    return base_frequency * (target_frequency / found_frequency)


# ------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------

if __name__ == '__main__':

    # 引数個数チェック
    if len(sys.argv)!=2:
        print('Invalid number of arguments.')
        exit(1)

    # ファイルから設定をロード
    config = configparser.ConfigParser()
    config.read(sys.argv[1])
    if config['behavior']['FILTER_TYPE'] == 'butter':
        FILTER_FUNCTION = apply_butter_worth
    elif config['behavior']['FILTER_TYPE'] == 'cheby':
        FILTER_FUNCTION = apply_chebyshev_1st
    else:
        print('invalid FILTER_TYPE.')
        exit(1)
    FILTER_FUNCTION = partial(FILTER_FUNCTION, is_zero_phase=bool( config['behavior']['IS_ZERO_PHASE'] ) )
    FILTER_ORDER = int(config['behavior']['FILTER_ORDER'])
    TARGET_GAIN = int(config['behavior']['TARGET_GAIN'])
    TARGET_FREQ = int(config['behavior']['TARGET_FREQ'])
    ITERATION_COUNT = int(config['behavior']['ITERATION_COUNT'])
    QUERY_FREQ = [int(s) for s in config['display']['QUERY_FREQ'].split(',')]
    PLOT_MIN_FREQ = int(config['display']['PLOT_MIN_FREQ'])
    PLOT_MAX_FREQ = int(config['display']['PLOT_MAX_FREQ'])
    PLOT_MIN_MAG = int(config['display']['PLOT_MIN_MAG'])
    PLOT_MAX_MAG = int(config['display']['PLOT_MAX_MAG'])
    PLOT_MIN_PHASE = int(config['display']['PLOT_MIN_PHASE'])
    PLOT_MAX_PHASE = int(config['display']['PLOT_MAX_PHASE'])

    # 最適なカットオフ周波数を推定
    WORKING_CUTOFF_FREQUENCY = TARGET_FREQ
    OPTIMAL_CUTOFF_FREQUENCY = TARGET_FREQ
    OPTIMAL_ERROR = float('inf')
    for i in range(1, 1000):
        WORKING_CUTOFF_FREQUENCY = estimate_optimal_cutoff_freqency(FILTER_FUNCTION, WORKING_CUTOFF_FREQUENCY, FILTER_ORDER, TARGET_GAIN, TARGET_FREQ)
        FREQUENCY_LIST, RESPONSE_AMPLITUDE, RESPONSE_PHASE = calculate_frequency_response(FILTER_FUNCTION, WORKING_CUTOFF_FREQUENCY, FILTER_ORDER)
        WORKING_ERROR = numpy.abs(query_amplitude_response_from_frequency(FREQUENCY_LIST, RESPONSE_AMPLITUDE, TARGET_FREQ)[1] - TARGET_GAIN)
        if WORKING_ERROR < OPTIMAL_ERROR:
            OPTIMAL_CUTOFF_FREQUENCY = WORKING_CUTOFF_FREQUENCY
            OPTIMAL_ERROR = WORKING_ERROR

    # イテレーション中で見つけた最適なカットオフ周波数で再計算
    FREQUENCY_LIST, RESPONSE_AMPLITUDE, RESPONSE_PHASE = calculate_frequency_response(FILTER_FUNCTION, OPTIMAL_CUTOFF_FREQUENCY, FILTER_ORDER)
    print('cutoff=%f' % (OPTIMAL_CUTOFF_FREQUENCY,))
    for f in QUERY_FREQ:
        RESULT_FREQ, RESULT_AMPLITUDE = query_amplitude_response_from_frequency(FREQUENCY_LIST, RESPONSE_AMPLITUDE, f)
        print('amplitude@%.2fHz=%.2f' % (RESULT_FREQ, RESULT_AMPLITUDE, ) )

    # 結果をプロット
    plt.subplot(2, 1, 1)
    plt.xscale('log')
    plt.grid(which='major',color='black',linestyle='-')
    plt.grid(which='minor',color='black',linestyle='-')
    plt.xlim(PLOT_MIN_FREQ, PLOT_MAX_FREQ)
    plt.ylim(PLOT_MIN_MAG, PLOT_MAX_MAG)
    plt.plot(FREQUENCY_LIST, RESPONSE_AMPLITUDE, linestyle='-', label = 'original')
    plt.subplot(2, 1, 2)
    plt.xscale('log')
    plt.grid(which='major',color='black',linestyle='-')
    plt.grid(which='minor',color='black',linestyle='-')
    plt.xlim(PLOT_MIN_FREQ, PLOT_MAX_FREQ)
    plt.ylim(PLOT_MIN_PHASE, PLOT_MAX_PHASE)
    plt.plot(FREQUENCY_LIST, numpy.degrees(RESPONSE_PHASE), linestyle='-', label = 'original')

    # 結果表示
    plt.show()

    # 正常終了
    exit(0)
