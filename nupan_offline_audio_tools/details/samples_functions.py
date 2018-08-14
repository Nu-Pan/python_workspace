import numpy

from .helper_functions import argextrema
from .helper_functions import compose_samples
from .filter_functions import apply_zplr

import matplotlib.pyplot as plt
from statistics import mean

# ------------------------------------------------------------------------------
# constants
# ------------------------------------------------------------------------------

from .default_constants import *
from .samples_functions import *

# ------------------------------------------------------------------------------
# functions
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
    ガケ位置としてはゼロ交差点付近が返却される。
    '''
    # 極値を全て列挙
    extrema_indices = argextrema(samples)
    extrema_indices = extrema_indices[start_offset <= extrema_indices]
    extrema_peak = samples[extrema_indices]
    # `隣接する極値間の差の絶対値の最大値' / '隣接する極値間の間隔'から`ガケしきい値'を計算
    extrema_amplitude_diff = numpy.absolute(numpy.diff(numpy.append(extrema_peak, extrema_peak[0])))
    extrema_distance_diff = numpy.diff(numpy.append(extrema_indices, extrema_indices[0] + samples.shape[0]))
    extrema_diff = extrema_amplitude_diff / extrema_distance_diff
    extrema_diff_max = numpy.max(extrema_diff)
    click_threshold = extrema_diff_max * threshold
    # ガケのうち最も過去（最初）ものを選択してそれのサンプル位置を返却
    click_index_in_extrema = numpy.where(click_threshold < extrema_diff)
    click_sample_offset = extrema_indices[click_index_in_extrema[0][0]] + int(extrema_distance_diff[click_index_in_extrema[0][0]] / 2)
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

def correct_attack_only_bass(samples, sample_rate, envelope):
    '''
    samples のベース帯域のみのアタック感を補正する。\n
    cross_freq 以下の周波数をベース帯域とみなす。\n
    envelope を補正として適用する。\n
    '''
    # 帯域を分離
    lowband = apply_zplr(samples, 'low', 200, sample_rate)
    highband = apply_zplr(samples, 'high', 200, sample_rate)
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

def _filter_extrema_zerocross(extrema_offset, extrema_amplitude):
    '''
    ゼロクロスする極値のみを残す。
    始点と終点を残す。
    '''
    # ゼロクロスする「クリック」の開始極値のインデックスを列挙
    # note 後続極値との積の符号がマイナスなら、後続極値までの間にゼロクロスしている
    extrema_amplitude_current = extrema_amplitude[0:-1]
    extrema_amplitude_next = extrema_amplitude[1:]
    extrema_amplitude_sign = extrema_amplitude_current * extrema_amplitude_next
    filter_arg = numpy.where(extrema_amplitude_sign <= 0)[0]
    # 開始極値の次の極値も残す
    filter_arg = numpy.unique(numpy.r_[filter_arg, filter_arg + 1])
    # 正常終了
    return extrema_offset[filter_arg], extrema_amplitude[filter_arg]

def _filter_extrema_velocity(extrema_offset, extrema_amplitude, threshold):
    '''
    速度ベースで極値をフィルタリングする。
    '''
    # 落下速度（`後続する極値との差の絶対値' / '後続する極値との間隔'）を計算
    extrema_distance_diff = numpy.diff(extrema_offset)
    extrema_amplitude_diff = numpy.absolute(numpy.diff(extrema_amplitude))
    extrema_velocity = extrema_amplitude_diff / extrema_distance_diff
    # クリックとみなすスレッショルドを計算
    extrema_velocity_max = numpy.max(extrema_velocity)
    click_threshold = extrema_velocity_max * threshold
    # フィルタリング
    filter_arg = numpy.where(click_threshold < extrema_velocity)[0]
    return extrema_offset[filter_arg], extrema_amplitude[filter_arg]

def _filter_zerocross_period(zerocross_offset, period, torelence_period_error_rate):
    '''
    周期を元にゼロクロス点をフィルタリングする。
    アウトライアが存在する場合、その直前のインライアも除去される。
    '''
    zerocross_distance = numpy.diff(zerocross_offset)
    period_error_rate = (zerocross_distance - period) / period
    filter_arg = numpy.where(period_error_rate < torelence_period_error_rate)
    return zerocross_offset[filter_arg]

def _complete_zerocross_period(source_zerocross_offset, candidate_zerocross_offset, period, torelence_period_error_rate):
    '''
    周期情報を元にゼロクロス点を補完する。
    _filter_extrema_period() によってアウトライアが除去された状態の極値が与えられる事を仮定する。
    '''
    result_offset = source_zerocross_offset
    # 欠損箇所がなくなるまで繰り返す(頭が悪い) 
    while True:
        # 欠損箇所を検出
        tolerance_max_distance = period * (1.0 + torelence_period_error_rate)
        extrema_distance = numpy.diff(result_offset)    
        corrupt_zerocross_arg = numpy.where( tolerance_max_distance < extrema_distance)
        corrupt_zerocross_offset = result_offset[corrupt_zerocross_arg]
        # 欠損がなければ終了
        if corrupt_zerocross_offset.shape[0] == 0:
            break
        # 付加するべきオフセット値を列挙
        augumented_offset = numpy.empty(corrupt_zerocross_offset.shape[0], int)
        for i in range(0, corrupt_zerocross_offset.shape[0]):
            expected_offset = corrupt_zerocross_offset[i] + period
            torelance_error = period * torelence_period_error_rate
            expected_offset_lower = expected_offset - torelance_error
            expected_offset_upper = expected_offset + torelance_error
            query_result = candidate_zerocross_offset[candidate_zerocross_offset <= expected_offset_upper]
            query_result = query_result[expected_offset_lower <= query_result]
            if query_result.size == 0:
                augumented_offset[i] = expected_offset
            else:
                augumented_offset[i] = query_result[int(query_result.size/2)]
        # 返却値を更新
        result_offset = numpy.unique(numpy.sort(numpy.append(result_offset, augumented_offset)))
    # 正常終了
    return  result_offset

def detect_click(samples, peak_amplitude_threshold, torelence_period_error_rate):
    '''
    samples 中の「クリック」を検出する。\n
    samples は変化のほぼない一定の波形であることを仮定する。\n
    「クリック」はノコギリ波の急激に変化する部分のようなものを指す。\n
    検出されたクリックのうち、 expective_click_offset に最も近い物を結果として返却する。\n
    expectied_click_offsets はリストで複数個渡すことができ、返却されるオフセットもリスト形式となる。\n
    expectied_click_offsets の N 個目の要素に対応する結果は N 個目に格納されている。
    '''
    ''' TODO なんとかする
    ある波形の周期を推定する\n
    引数として波形のピーク情報を受け取り、そのピーク間距離の中央値を周期とみなす。\n
    なお、元波形はほぼ変化のない一定の波形であることを仮定する。\n
    \n
    extrema_offset は元波形から検出されたピークの位置（オフセット、サンプル単位）。\n
    extrema_amplitude 元波形から検出されたピークの振幅。\n
    いずれも配列形式を仮定する。\n
    \n
    threshold は経験的に決定するべきパラメータ。\n
    サンプル列の全範囲中のピーク差のうち最大のピーク差 * threshold までの範囲のクリックとして認識する。\n
    0.90程度の 1 に近いパラメータを推奨。\n
    \n
    推定結果はサンプル単位の int で返却される。\n
    \n
    '''

    #plt.plot(samples)

    # 元波形中の全てのゼロクロスポイントを検出
    samples_next = samples[1:]
    samples_next_sign = samples_next * samples[:samples.shape[0]-1]
    samples_zerocross_offset = numpy.where(samples_next_sign <= 0)[0]

    #plt.plot((samples_zerocross_offset,), 0, "go")

    # 全ての極値の位置と振幅を列挙
    extrema_offset = argextrema(samples)
    extrema_amplitude = samples[extrema_offset]
    # ゼロクロスするピークを抽出
    extrema_offset, extrema_amplitude = _filter_extrema_zerocross(extrema_offset, extrema_amplitude)

    #plt.plot(extrema_offset, extrema_amplitude, "ro")

    # 速度ベースのフィルタリングでピークを絞る
    extrema_offset, extrema_amplitude = _filter_extrema_velocity(extrema_offset, extrema_amplitude, peak_amplitude_threshold)

    #plt.plot(extrema_offset, extrema_amplitude, "co")

    # ゼロクロスピークの開始オフセットを元にゼロクロス位置を検出、ピーク位置とする
    click_zerocross_offset = numpy.empty(extrema_offset.shape[0], int)
    for arg in range(0, extrema_offset.size): # この処理どう考えても重い
        click_zerocross_offset[arg] = samples_zerocross_offset[extrema_offset[arg] < samples_zerocross_offset][0]
    # ゼロクロスピークの間隔の中央値を波形の周期とみなす
    estimated_period = numpy.median(click_zerocross_offset[1:]-click_zerocross_offset[0:-1])

    #plt.plot((click_zerocross_offset,), 0, "bo")

    # 周期情報を元にアウトライアを除去
    click_zerocross_offset = _filter_zerocross_period(click_zerocross_offset, estimated_period, torelence_period_error_rate)

    #plt.plot((click_zerocross_offset,), 0, "mo")

    # 周期情報を元にゼロクロスクリックを補完
    click_zerocross_offset = _complete_zerocross_period(click_zerocross_offset, samples_zerocross_offset, estimated_period, torelence_period_error_rate)

    #plt.plot((click_zerocross_offset,), 0, "yo")

    # 正常終了
    return click_zerocross_offset

def detect_positive_zero_cross(samples):
    '''
    samples 中の「ゼロクロスポイント」を検出する。\n
    samples は変化のほぼない一定の波形であることを仮定する。\n
    「ゼロクロスポイント」は samples 先頭からのサンプル数で返却される。\n
    以下の条件を満たす「ゼロクロスポイント」１つを結果として返却する。\n
    - 先頭からのサンプル数が triming_offset 以上
    - 最も先頭に近い
    - 次のサンプルが正の値を取る
    '''
    # samples 中の全てのゼロクロスポイントを検出
    samples_next = samples[1:]
    samples_next_sign = samples_next * samples[:samples.shape[0]-1]
    samples_zerocross_offset = numpy.where(samples_next_sign <= 0)[0]

    # 正常終了
    return samples_zerocross_offset

def detect_positive_extrema(samples):
    '''
    samples 中の「極値」を検出する。\n
    samples は変化のほぼない一定の波形であることを仮定する。\n
    「極値」は samples 先頭からのサンプル数で返却される。\n
    以下の条件を満たす「極値」１つを結果として返却する。\n
    - 先頭からのサンプル数が triming_offset 以上
    - 最も先頭に近い
    - 正の値をとる
    - １つ前の極値が負である
    '''
    # samples 中のすべての極値を検出
    extremas = argextrema(samples)

    # 負の方向の極値→正の方向の極値になるものを探索
    '''
    @note:
        いい手が思いつかなかったのでナイーブに for で実装。            
    '''
    selected_extremas = numpy.empty((0,), numpy.int64)
    for i in range(1, extremas.shape[0]):
        if samples[extremas[i-1]] < 0 and 0 < samples[extremas[i]]:
            selected_extremas = numpy.append(selected_extremas, extremas[i])

    # 正常終了
    return selected_extremas

def convert_to_median_rms(samples, window_size):
    '''
    引数 samples を RMS に変換する。
    あるサンプル位置における RMS をその位置の前後 +- window_size / 2 サンプルの範囲で計算し
    その結果として得られた RMS 配列の中央値を samples の RMS とみなす。
    '''
    per_ch_rms = []
    window = numpy.ones(window_size) / window_size
    for i in range(0, samples.shape[1]):
        samples_ch = samples[:,i]
        samples_s = samples_ch * samples_ch
        samples_ms = numpy.convolve(samples_s, window, 'valid')
        samples_ms.sort()
        median_rms = numpy.sqrt(samples_ms[int(len(samples_ms)/2)])
        per_ch_rms.append(median_rms)
    return mean(per_ch_rms)

def convert_to_median_peak(samples, window_size):
    '''
    引数 samples をピークの配列に変換する。
    あるサンプル位置におけるピークをその位置の前後 +- window_size / 2 サンプルの範囲で計算されし
    その結果として得られたピーク配列の中央値を samples の RMS とみなす。
    '''
    samples_peak = scipy.ndimage.filters.maximum_filter1d(samples, window_size)
    samples_peak.sort()
    return samples_peak[int(len(samples_peak)/2)]

def detect_zerocross_points(samples):
    '''
    引数 samples 中のゼロクロス点の位置をすべて列挙する。\n
    位置はサンプル数単位。\n
    '''
    samples_next = samples[1:]
    samples_next_sign = samples_next * samples[:samples.shape[0]-1]
    return numpy.where(samples_next_sign <= 0)[0]
