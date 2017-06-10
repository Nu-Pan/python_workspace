import sys
import os
import numpy
from scipy import signal
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# 定数
# ------------------------------------------------------------------------------

# 無音判定しきい値
SILENT_THRESHOLD = 1.0 / (2 ** 2)

# クロスオーバーとして使用する LR フィルタのパラメータ
# FILTER_ORDER 次の butter-worth フィルタをカスケード接続する
FILTER_FREQUENCY = 200
FILTER_ORDER = 2

# 「ガケ」位置検出に使うパラメータ
CLICK_ESTIMATE_THRESHOLD = 0.95
CLICK_ENVELOPE_THREASHOLD_DECIBEL = -6

# ------------------------------------------------------------------------------
# 単純な機能の内部関数
# ------------------------------------------------------------------------------

def to_decibel(ratio):
    '比率からデシベルに変換'
    return 20 * numpy.log(ratio)

def to_ratio(decibel):
    'デシベルから比率に変換'
    return numpy.power(10, decibel / 20)

def estimate_amplitude_envelope(samples):
    '''
    与えられたサンプル列の振幅包絡線を得る。\n
    包絡線は入力と同要素数のサンプル列として返却される\n
    '''
    return numpy.abs(signal.hilbert(samples))

def argextrema(samples):
    '与えられたサンプル列の極値を得る'
    return numpy.sort(numpy.c_[signal.argrelmax(samples), signal.argrelmin(samples)]).flatten()

def normalize_frequency(frequency, sample_rate):
    'frequency[Hz] を [0, sample_rate/2] -> [0.0, 1.0] の値域にマップする。'
    return frequency / (sample_rate / 2.0)

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

def extract_low_band(samples, sample_rate):
    '''
    入力サンプル列の低音成分を抽出したサンプル列を得る。\n
    注意点等については _apply_zpf() に準ずる。\n
    '''
    return apply_zpf(samples, FILTER_ORDER, FILTER_FREQUENCY, sample_rate, 'low')

def extract_high_band(samples, sample_rate):
    '''
    入力サンプル列の高音成分を抽出したサンプル列を得る。\n
    注意点等については _apply_zpf() に準ずる。\n
    '''
    return apply_zpf(samples, FILTER_ORDER, FILTER_FREQUENCY, sample_rate, 'high')

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

# ------------------------------------------------------------------------------
# correct_bass サブ実装
# ------------------------------------------------------------------------------

def estimate_bass_click_offset_(samples, threshold, start_offset):
    '''
    与えられた波形 samples の最初の「ガケ」の位置を推定する。\n
    samples はモノラル音源であると仮定する。\n
    ガケの位置はサンプル数で返却される。\n
    threshold は経験的に決定するべきパラメータ。\n
    サンプル列の全範囲中のピーク差のうち最大のピーク差 * threshold までの範囲のクリックとして認識する。\n
    0.95程度の 1 に近いパラメータを推奨。\n
    start_offset は探索開始オフセットで、これよりも前のガケは検出されない。\n
    '''
    # 極値を全て列挙
    local_extrema_indices = argextrema(samples)
    local_extrema_indices = local_extrema_indices[start_offset <= local_extrema_indices]
    local_extrema_peak = samples[local_extrema_indices]
    # `隣接する極値間の差の絶対値の最大値' / '隣接する極値間の間隔'から`ガケしきい値'を計算
    local_extrema_amplitude_diff = numpy.absolute(numpy.diff(numpy.append(local_extrema_peak, local_extrema_peak[0])))
    local_extrema_distance_diff = numpy.diff(numpy.append(local_extrema_indices, local_extrema_indices[0] + samples.shape[0]))
    local_extrema_diff = local_extrema_amplitude_diff / local_extrema_distance_diff
    local_extrema_diff_max = numpy.max(local_extrema_diff)
    click_threshold = local_extrema_diff_max * threshold
    # ガケのうち最も過去（最初）ものを選択してそれのサンプル位置を返却
    click_index_in_extrema = numpy.where(click_threshold < local_extrema_diff)
    click_sample_offset = local_extrema_indices[click_index_in_extrema[0][0]]
    # 正常終了
    return click_sample_offset

def estimae_kick_click_offset(kick_lowband_samples, target_ratio):
    '''
    ローパス済みキックサンプル列から「初回クリック位置」として適切な位置を推定する。\n
    「初回クリック位置」はベースの「クリック」がちょうどくる位置の事。\n
    「クリック」は隣接する極値の落差が最も大きい位置の事。\n
    位置はオフセットをサンプル数で返す。\n
    元となるキックの極値のうち振幅が target_ratio と最も近いものの位置を適切な「初回クリック位置」とする。\n
    '''
    # 対象波形の最大ピーク値を元に目標比率を補正
    source_max_amplitude = numpy.max(kick_lowband_samples)
    corrected_target_ratio = target_ratio * source_max_amplitude
    # 先頭から振幅が最大に達するまでの区間で目標に最も近い振幅を持つ極値を計算
    source_max_offset = numpy.argmax(kick_lowband_samples)
    source_extrema_indices = argextrema(kick_lowband_samples[0:source_max_offset])
    optimal_click_index_in_extrema = numpy.argmin(numpy.abs(kick_lowband_samples[source_extrema_indices] - corrected_target_ratio))
    optimal_click_offset = source_extrema_indices[optimal_click_index_in_extrema]
    # 正常終了
    return optimal_click_offset

def collect_attack_only_bass(samples, sample_rate, envelope):
    '''
    samples のベース帯域のみのアタック感を補正する。\n
    cross_freq 以下の周波数をベース帯域とみなす。\n
    envelope を補正として適用する。\n
    '''
    # 帯域を分離
    lowband = extract_low_band(samples, sample_rate)
    highband = extract_high_band(samples, sample_rate)
    # 補正に使うエンベロープを用意（先頭から最高ピークまでの範囲を切り出して最大 1.0 に正規化）
    trim_length = numpy.argmax(envelope)
    trimmed_envelope = envelope[0:trim_length]
    trimmed_envelope = trimmed_envelope / numpy.max(trimmed_envelope)
    # 低音域エンベロープを適用して元に戻す
    for channel in range(0, lowband.shape[1]):
        lowband[0:trim_length, channel] = lowband[0:trim_length, channel] * trimmed_envelope
    result = lowband + highband
    # 正常終了
    return result

# ------------------------------------------------------------------------------
# correct_bass メイン実装
# ------------------------------------------------------------------------------

def correct_bass(inputs, inputs_samplerate):
    '''
    inputs に含まれるキック波形とベース波形に補正をかける。\n
    補正処理は in-place で行われる。\n
    \n
    inputs の形式については correct_bass.py の呼び出し箇所を参照。\n
    inputs_samplerate には inputs に含まれるサンプル列のサンプルレートを渡す。\n
    異なるサンプルレートのサンプル列を混ぜて渡すことはできない。\n
    '''

    # 無音が混じってないかチェック
    for i in inputs:
        if is_slient_samples(i['stereo_samples']):
            print('(error) : Silient sample was passed from callee.')
            print('(error) : File = %s' % i['path'])
            return True

    # モノラル波形とそのローパス波形を事前に生成
    for i in inputs:
        temp = i['stereo_samples']
        temp = (temp[:, 0] + temp[:, 1]) / 2.0
        i['monoral_sample'] = temp
        i['monoral_lowband_sample'] = extract_low_band(temp, inputs_samplerate)

    # キックのローパス波形から「ガケ」合わせ位置を決定
    optimal_click_offset = estimae_kick_click_offset(inputs[0]['monoral_lowband_sample'], to_ratio(CLICK_ENVELOPE_THREASHOLD_DECIBEL))
    # DEBUG TODO verbose モードを実装
    # print('optimal_click_offset = %d [samples]' % optimal_click_offset)

    # 最適な位置に「ガケ」を合わせる
    for i in inputs[1:]:
        actual_click_delay = estimate_bass_click_offset_(i['monoral_lowband_sample'], CLICK_ESTIMATE_THRESHOLD, optimal_click_offset)
        # DEBUG
        # print('<%s> actual_click_delay = %d [samples]' % (i['path'], actual_click_delay))
        i['click_collected'] = shift_forward_and_padding(i['stereo_samples'], actual_click_delay - optimal_click_offset)

    # キックの包絡線を求める
    inputs[0]['monoral_lowband_envelope'] = estimate_amplitude_envelope(inputs[0]['monoral_lowband_sample'])

    # キックの包絡線をベース(200Hz以下)の振幅にアタック限定で適用
    for i in inputs[1:]:
        i['attack_collected'] = collect_attack_only_bass(i['click_collected'], inputs_samplerate, inputs[0]['monoral_lowband_envelope'])

    # TODO キックのリリースも調整する

    # 正常終了
    return False