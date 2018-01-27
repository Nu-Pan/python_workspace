import numpy
from scipy import signal

from .helper_functions import *

# クロスオーバーとして使用する LR フィルタのパラメータ
# FILTER_ORDER 次の butter-worth フィルタをカスケード接続する
# TODO 名前適切じゃないし定数は別ファイルに移動させるべき
FILTER_FREQUENCY = 200
FILTER_ORDER = 2

# TODO 関数の切り分け方が果てしなく微妙

def apply_filter(samples, filter_type, filter_mode, filter_order, cutoff_frequency, sample_rate, is_zero_phase):
    '''
    入力サンプル列にフィルタを適用する。\n
    入力サンプル列は x 軸（第０軸）が時間方向であると仮定する。\n
    - samples : 入力サンプル列
    - filter_type : フィルタアルゴリズム('butter', 'cheby')
    - filter_mode : フィルタモード('low', 'high')
    - filter_order : フィルタ次数
    - cutoff_freqency : カットオフ周波数
    - sample_rate : サンプルレート
    - is_zer_phase : True の時ゼロ位相フィルタリング。\n同一のフィルタが二回適用されるので注意。
    '''
    normalized_frequency = normalize_frequency(cutoff_frequency, sample_rate)
    if filter_type=='butter':
        b, a = signal.butter(filter_order, normalized_frequency, filter_mode)
    elif filter_type=='cheby1st':
        b, a = signal.butter(filter_order, normalized_frequency, filter_mode)
    else:
        print('In apply_filter(). Unknown filter_type=%s' %(filter_type,))
        return None
    if is_zero_phase:
        return signal.filtfilt(b, a, samples, 0)
    else:
        return signal.lfilter(b, a, samples, 0)

def apply_zplr(samples, filter_mode, cutoff_frequency, sample_rate):
    '''
    入力サンプル列にゼロ位相の linkwitz-riley フィルタを適用する。\n
    詳細は apply_filter() を参照。\n
    '''
    return apply_filter(samples, 'butter', filter_mode, 2, cutoff_frequency, sample_rate, True)
