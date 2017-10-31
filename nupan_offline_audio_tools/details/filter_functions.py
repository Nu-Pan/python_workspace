import numpy
from scipy import signal

from .helper_functions import *

# クロスオーバーとして使用する LR フィルタのパラメータ
# FILTER_ORDER 次の butter-worth フィルタをカスケード接続する
# TODO 名前適切じゃないし定数は別ファイルに移動させるべき
FILTER_FREQUENCY = 200
FILTER_ORDER = 2

# TODO 関数の切り分け方が果てしなく微妙

def apply_zpf(samples, filter_order, cutoff_frequency, sample_rate, filter_type):
    '''
    入力サンプル列にゼロ位相の linkwitz-riley フィルタを適用する。\n
    言い換えれば、butter-worth フィルタを２回適用する。\n
    入力サンプル列は x 軸（第０軸）が時間方向であると仮定する。\n
    filter_order = 2 の時 -12dB/Oct を２回かける。\n
    '''
    # ゼロ位相フィルタリングを適用
    normalized_frequency = normalize_frequency(cutoff_frequency, sample_rate)
    lpf_b, lpf_a = signal.butter(filter_order, normalized_frequency, filter_type)
    return signal.filtfilt(lpf_b, lpf_a, samples, 0)

def extract_low_band(samples, sample_rate, frequency=FILTER_FREQUENCY, offset=0):
    '''
    入力サンプル列の低音成分を抽出したサンプル列を得る。\n
    注意点等については _apply_zpf() に準ずる。\n
    '''
    return apply_zpf(samples, FILTER_ORDER, frequency+offset, sample_rate, 'low')

def extract_high_band(samples, sample_rate, frequency=FILTER_FREQUENCY, offset=0):
    '''
    入力サンプル列の高音成分を抽出したサンプル列を得る。\n
    注意点等については _apply_zpf() に準ずる。\n
    '''
    return apply_zpf(samples, FILTER_ORDER, frequency+offset, sample_rate, 'high')

def cutoff_extreme_band(samples, sample_rate):
    '''
    入力サンプル列の 40Hz-20kHz 成分を抽出したサンプル列を得る。\n
    要するに極端な周波数帯を削ぎ落とす。\n
    注意点等については _apply_zpf() に準ずる。\n
    '''
    temp_samples = samples
    temp_samples = apply_zpf(temp_samples, FILTER_ORDER, 20, sample_rate, 'high')
    temp_samples = apply_zpf(temp_samples, FILTER_ORDER, 20 * 1000, sample_rate, 'low')
    return temp_samples
