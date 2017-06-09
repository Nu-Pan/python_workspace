import sys
import os
import numpy
from scipy import signal
import soundfile as sf
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 定数
# -----------------------------------------------------------------------------

# クロスオーバーとして使用する LR フィルタのパラメータ
# FILTER_ORDER 次の butter-worth フィルタをカスケード接続する
FILTER_FREQUENCY = 200
FILTER_ORDER = 2

# 「ガケ」位置検出に使うパラメータ
CLICK_ESTIMATE_THRESHOLD = 0.95
CLICK_ENVELOPE_THREASHOLD_DECIBEL = -3

# 出力ファイルにつけるプリフィックス
OUTPUT_FILE_PREFIX = 'output_'

# -----------------------------------------------------------------------------
# 単純な機能の関数
# -----------------------------------------------------------------------------

def decompose_path(path):
    'path を (directory, stem, extension) に分解'
    directory, base_name = os.path.split(path)
    stem, extension = os.path.splitext(base_name)
    return (directory, stem, extension)

def compose_path(directory, stem, extension):
    '(directory, stem, extension) からパスを合成'
    return os.path.join(directory, stem + extension)

def to_decibel(ratio):
    '比率からデシベルに変換'
    return 20 * numpy.log(ratio)

def to_ratio(decibel):
    'デシベルから比率に変換'
    return numpy.power(10, decibel / 20)

def estimate_amplitude_envelope(samples):
    '''
    与えられたサンプル列の振幅包絡線を得る。
    包絡線は入力と同要素数のサンプル列として返却される
    '''
    return numpy.abs(signal.hilbert(samples))

def argextrema(samples):
    '与えられたサンプル列の極値を得る'
    return numpy.sort(numpy.c_[signal.argrelmax(samples), signal.argrelmin(samples)]).flatten()

def normalize_frequency(frequency, sample_rate):
    '''
    frequency[Hz] を [0, sample_rate/2] -> [0.0, 1.0] の値域にマップする。
    '''
    return frequency / (sample_rate / 2.0)

def apply_zpf(samples, filter_order, cutoff_frequency, sample_rate, filter_type):
    '''
    入力サンプル列にゼロ位相の linkwitz-riley フィルタを適用する。
    言い換えれば、butter-worth フィルタを２回適用する。
    入力サンプル列は x 軸（第０軸）が時間方向であると仮定する。
    filter_order = 2 の時 -12dB/Oct を２回かける。
    '''
    # ゼロ位相フィルタリングを適用
    normalized_frequency = normalize_frequency(cutoff_frequency, sample_rate)
    lpf_b, lpf_a = signal.butter(filter_order, normalized_frequency, filter_type)
    return signal.filtfilt(lpf_b, lpf_a, samples, 0)

def extract_low_band(samples, sample_rate):
    '''
    入力サンプル列の低音成分を抽出したサンプル列を得る。
    注意点等については _apply_zpf() に準ずる。
    '''
    return apply_zpf(samples, FILTER_ORDER, FILTER_FREQUENCY, sample_rate, 'low')

def extract_high_band(samples, sample_rate):
    '''
    入力サンプル列の高音成分を抽出したサンプル列を得る。
    注意点等については _apply_zpf() に準ずる。
    '''
    return apply_zpf(samples, FILTER_ORDER, FILTER_FREQUENCY, sample_rate, 'high')

def shift_forward_and_padding(samples, offset):
    '''
    samples を offset だけ前方にずらす。
    前方にはみ出た分はトリムされる。
    入力と出力でサンプル数が同じになるように末尾にゼロがパディングされる。
    '''
    return numpy.pad(samples[offset:,:], ((0, offset), (0, 0)), 'constant', constant_values=0)

# -----------------------------------------------------------------------------
# 複雑な機能の関数
# -----------------------------------------------------------------------------

def estimate_click_delay(samples, threshold, start_offset):
    '''
    与えられた波形 samples の最初の「ガケ」の位置を推定する。
    samples はモノラル音源であると仮定する。
    ガケの位置はサンプル数で返却される。
    threshold は経験的に決定するべきパラメータ。
    サンプル列の全範囲中のピーク差のうち最大のピーク差 * threshold までの範囲のクリックとして認識する。
    0.95程度の 1 に近いパラメータを推奨。
    start_offset は探索開始オフセット。
    コレよりも前のガケは検出されない。
    '''
    # 極値を全て列挙
    local_extrema_indices = argextrema(samples)
    local_extrema_indices = local_extrema_indices[start_offset <= local_extrema_indices]
    local_extrema_peak = samples[local_extrema_indices]
    # `隣接する極値間の差の絶対値の最大値'から`ガケしきい値'を計算
    local_extrema_diff = numpy.absolute(numpy.diff(numpy.append(local_extrema_peak, local_extrema_peak[0])))
    local_extrema_diff_max = numpy.max(local_extrema_diff)
    click_threshold = local_extrema_diff_max * threshold
    # ガケのうち最も過去（最初）ものを選択してそれのサンプル位置を返却
    click_index_in_extrema = numpy.where(click_threshold < local_extrema_diff)
    click_sample_offset = local_extrema_indices[click_index_in_extrema[0][0]]
    # 正常終了
    return click_sample_offset

def estimate_optimal_click_offset(kick_lowband_samples, target_ratio):
    '''
    ローパス済みキックサンプル列から「初回クリック位置」として適切な位置を推定する。
    「初回クリック位置」はベースの「クリック」がちょうどくる位置の事。
    「クリック」は隣接する極値の落差が最も大きい位置の事。
    位置はオフセットをサンプル数で返す。
    元となるキックの極値のうち振幅が target_ratio と最も近いものの位置を適切な「初回クリック位置」とする。
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
    samples のベース帯域のみのアタック感を補正する。
    cross_freq 以下の周波数をベース帯域とみなす。
    envelope を補正として適用する。
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

# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    # 定数
    SAMPLE_FORMAT = 'float64'
    '''
    # 引数チェック
    if len(sys.argv) < 2:
        print('Too few number of arguments(up to 2)')
        print('Usage : python correct_bass.py <kick_sample>.wav <bass_sample_1>.wav ... <bass_sample_N>.wav')
        exit(1)
    '''

    DEBUG_INPUT_PATH = sys.argv

    # 指定ファイル全てメモリ上にロード
    SAMPLE_RATE = 0
    INPUTS = []
    for p in DEBUG_INPUT_PATH:
        # ロード
        TEMP_INPUT, TEMP_SAMPLE_RATE = sf.read(file=p, dtype=SAMPLE_FORMAT)

        # サンプルレートをチェック
        if SAMPLE_RATE == 0:
            SAMPLE_RATE = TEMP_SAMPLE_RATE
        elif SAMPLE_RATE != TEMP_SAMPLE_RATE:
            print('Wrong sample rate is detected in input files.')
            print('File = ' + p)
            print('Expected sample rate = ' + SAMPLE_RATE)
            print('Actual sample rate = ' + TEMP_SAMPLE_RATE)
            exit(1)

        # ロードした波形をリストに追加
        INPUTS.append({'stereo_sample': TEMP_INPUT, 'path': p})

    # モノラル波形とそのローパス波形を事前に生成
    for i in INPUTS:
        TEMP = i['stereo_sample']
        TEMP = (TEMP[:, 0] + TEMP[:, 1]) / 2.0
        i['monoral_sample'] = TEMP
        i['monoral_lowband_sample'] = extract_low_band(TEMP, SAMPLE_RATE)

    # キックのローパス波形から「ガケ」合わせ位置を決定
    optimal_click_offset = estimate_optimal_click_offset(INPUTS[0]['monoral_lowband_sample'], to_ratio(CLICK_ENVELOPE_THREASHOLD_DECIBEL))

    # 最適な位置に「ガケ」を合わせる
    for i in INPUTS[1:]:
        actual_click_delay = estimate_click_delay(i['monoral_lowband_sample'], CLICK_ESTIMATE_THRESHOLD, optimal_click_offset)
        i['click_collected'] = shift_forward_and_padding(i['stereo_sample'], actual_click_delay - optimal_click_offset)

    # キックの包絡線を求める
    INPUTS[0]['monoral_lowband_envelope'] = estimate_amplitude_envelope(INPUTS[0]['monoral_lowband_sample'])

    # キックの包絡線をベース(200Hz以下)の振幅にアタック限定で適用
    for i in INPUTS[1:]:
        i['attack_collected'] = collect_attack_only_bass(i['click_collected'], SAMPLE_RATE, INPUTS[0]['monoral_lowband_envelope'])

    # TODO キックのリリースも調整する

    # 補正をかけたベース波形を出力
    for i in INPUTS[1:]:
        directory, stem, extension = decompose_path(i['path'])
        output_path = compose_path(directory, OUTPUT_FILE_PREFIX + stem, extension)
        sf.write(file=output_path, data=i['attack_collected'], samplerate=SAMPLE_RATE, subtype='FLOAT')
