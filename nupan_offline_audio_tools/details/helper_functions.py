import numpy
from scipy import signal

# ------------------------------------------------------------------------------
# constants
# ------------------------------------------------------------------------------

from .default_constants import *

# ------------------------------------------------------------------------------
# function
# ------------------------------------------------------------------------------

def to_decibel(ratio):
    '比率からデシベルに変換'
    return 20 * numpy.log(ratio)

def to_ratio(decibel):
    'デシベルから比率に変換'
    return numpy.power(10, decibel / 20)

def time2sample(time_in_sec, samplerate):
    '秒数からサンプル数に変換'
    return int(time_in_sec * samplerate)

def nl2sl(length, bpm, sample_rate):
    '''
    Note Length To Sample Length.
    length は Fractions.Fraction を仮定。
    length.numerator / length.denominator 音符の長さをサンプル数で得る。
    計算には bpm と sample_rate も必要。
    '''
    unit_note_length_in_sec = 4.0 * 60.0 * (1.0 / float(bpm))
    length_in_sec = (float(length.numerator) / float(length.denominator)) * unit_note_length_in_sec
    length_in_sample = length_in_sec * sample_rate
    return int(length_in_sample)

def normalize_frequency(frequency, sample_rate):
    'frequency[Hz] を [0, sample_rate/2] -> [0.0, 1.0] の値域にマップする。'
    return frequency / (sample_rate / 2.0)

def estimate_amplitude_envelope(samples):
    '''
    与えられたサンプル列の振幅包絡線を得る。\n
    包絡線は入力と同要素数のサンプル列として返却される\n
    '''
    return numpy.abs(signal.hilbert(samples))

def argextrema(samples):
    '与えられたサンプル列の極値を得る'
    return numpy.sort(numpy.c_[signal.argrelmax(samples), signal.argrelmin(samples)]).flatten()

def shift_forward_and_padding(samples, offset):
    '''
    samples を offset だけ前方にずらす。\n
    前方にはみ出た分はトリムされる。\n
    入力と出力でサンプル数が同じになるように末尾にゼロがパディングされる。\n
    マルチチャンネルサンプル列可。\n
    入力サンプル列は x 軸（第０軸）が時間方向であると仮定する。\n
    '''
    return numpy.pad(samples[offset:,:], ((0, offset), (0, 0)), 'constant', constant_values=0)

def is_slient_samples(sample):
    '''
    無音のサンプル列であるか？
    全ての振幅が SILENT_THRESHOLD を下回る場合無音と判定される。
    '''
    return numpy.max(sample) < SILENT_THRESHOLD

def compose_samples(samples_1, samples_2):
    '''
    ２つのサンプル列を１つのサンプル列に結合する
    #TODO 色々チェック類
    '''
    return numpy.r_[samples_1, samples_2]

def where_nearest(array, query):
    '''
    array 中の query に最も近い要素のインデックスを得る
    '''
    return numpy.abs(array - query).argmin()

def value_nearest(array, query):
    '''
    array 中の query に最も近ち値を得る
    '''
    return array[where_nearest(array, query)]
