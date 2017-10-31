import os
import sys
import glob
from fractions import Fraction   

import numpy

from details import *

# import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# constants
# ------------------------------------------------------------------------------

# 出力ファイルにつけるプリフィックス
OUTPUT_FILE_PREFIX = 'output\\'

# ファイル名に付属する番号の桁数
FILESTEM_NUMBER_OF_DIGIT = 3

# 「ガケ」位置検出に使うパラメータ
CLICK_ESTIMATE_THRESHOLD = 0.95
CLICK_ENVELOPE_THREASHOLD_DECIBEL = -6

# ------------------------------------------------------------------------------
# correct_bass パラメータクラス
# ------------------------------------------------------------------------------

class correct_bass_parameters:
    def __init__(self):
        self.samplerate = 0
        self.bpm = 0
        self.bpm = 128
        self.head_click_offset = Fraction(0, 1)
        self.note_length = Fraction(1, 8)
        self.post_offset = Fraction(0, 1)
        self.detection_offset_in_samples = 2**13

# ------------------------------------------------------------------------------
# correct_bass メイン実装
# ------------------------------------------------------------------------------

def correct_bass(inputs, parameters):
    '''
    inputs に含まれるキック波形とベース波形に補正をかける。\n
    補正処理は in-place で行われる。\n
    \n
    inputs の形式については correct_bass.py の呼び出し箇所を参照。\n
    inputs_samplerate には inputs に含まれるサンプル列のサンプルレートを渡す。\n
    異なるサンプルレートのサンプル列を混ぜて渡すことはできない。\n
    '''
    # TODO verbose モードを実装

    # モノラル波形とそのローパス波形を事前に生成
    for i in inputs:
        temp = i['stereo']
        temp = (temp[:, 0] + temp[:, 1]) / 2.0
        i['monoral_sample'] = temp
        i['monoral_lowband_sample'] = extract_low_band(temp, parameters.samplerate)

    # TODO パラメータチェック
    # 渡された各パラメータをサンプル数に変換
    post_offset_in_samples = nl2sl(parameters.post_offset, parameters.bpm, parameters.samplerate)
    head_click_offset_in_samples = nl2sl(parameters.head_click_offset, parameters.bpm, parameters.samplerate)
    note_length_in_samples = nl2sl(parameters.note_length, parameters.bpm, parameters.samplerate)

    # 最適クリックオフセット（サンプル数単位）を定義
    optimal_head_click_offset_in_samples = head_click_offset_in_samples + post_offset_in_samples
    detection_head_click_offset_in_samples = optimal_head_click_offset_in_samples + parameters.detection_offset_in_samples
    
    # 波形のクリック位置が最適になるように処理する
    for i in inputs:
        # 波形から「クリック」位置を検出
        detected_click = detect_click(i['monoral_lowband_sample'], 0.9, 0.1)
        # 波形中の最適クリックオフセットに最も近いクリックを選択
        actual_head_click_offset = value_nearest(detected_click, detection_head_click_offset_in_samples)
        actual_tail_click_offset = value_nearest(detected_click, actual_head_click_offset + note_length_in_samples)
        # リサンプルを実行
        actual_note_length = actual_tail_click_offset - actual_head_click_offset
        round=lambda x:int((x*2+1)//2)
        scaled_sample_count = round(parameters.samplerate * note_length_in_samples / actual_note_length)
        scaled_actual_head_click_offset = round(actual_head_click_offset * note_length_in_samples / actual_note_length)

        print('resample scale = %f' % (note_length_in_samples / actual_note_length))

        #plt.plot(i['monoral_lowband_sample'])
        #plt.plog()

        i['click_scaled'] = signal.resample(i['stereo'], scaled_sample_count)
        # オフセットを実行
        i['click_corrected'] = shift_forward_and_padding(i['click_scaled'], scaled_actual_head_click_offset - optimal_head_click_offset_in_samples)

    # ローとハイに分離
    for i in inputs:
        i['click_corrected_low'] = extract_low_band(i['click_corrected'], parameters.samplerate)
        i['click_corrected_high'] = extract_high_band(i['click_corrected'], parameters.samplerate)

    # 結果に名前をつける
    for i in inputs:
        i['total_corrected_low'] = i['click_corrected_low']
        i['total_corrected_high'] = i['click_corrected_high']
        i['total_corrected_full'] = i['click_corrected_low'] + i['click_corrected_high']

    # 正常終了
    return False

# ------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------

def print_usage():
    'このプログラムの使い方を表示'
    print('Too few number of arguments(up to 1)')
    print('Usage1 : python correct_bass.py <bass_sample_1>.wav ... <bass_sample_N>.wav')
    print('Usage2 : python correct_bass.py <direcyory path>')
    print('details :')
    print('In usage2, <directory path> must be directory that contain bass wav file.')
if __name__ == '__main__':
    # 引数のエイリアスを作る
    INPUT_PATHS = sys.argv[1:]

    # 引数チェック
    if len(INPUT_PATHS) < 1:
        print_usage()
        exit(1)

    # ディレクトリ指定の場合は中身を探索して引数を作る
    if os.path.isdir(INPUT_PATHS[0]):
        # 指定ディレクトリ中のマッチする名前のファイルを列挙
        INPUT_DIR = INPUT_PATHS[0]
        FILES = glob.glob(os.path.join(INPUT_DIR, '*.wav'))
        BASS_FILES = [x for x in FILES if 'bass' in x.lower()]

        # 見つかったファイル数のチェック
        if len(BASS_FILES) == 0:
            print('(error) : In usage2, No bass wav file in directory "%s".' % INPUT_DIR)
            print_usage()
            exit(1)

        # 引数リストを生成
        INPUT_PATHS = BASS_FILES

    # 引数チェック
    if len(INPUT_PATHS) < 2:
        print_usage()
        exit(1)

    # 指定ファイル全てメモリ上にロード
    SAMPLE_RATE = 0
    INPUTS = []
    for p in INPUT_PATHS:
        # ロード
        try:
            TEMP_INPUT, TEMP_SAMPLE_RATE = load_samples(p, INTERNAL_SAMPLE_FORMAT)
        except Exception as err:
            print(err)
            print_usage()
            raise

        # サンプルレートをチェック
        if SAMPLE_RATE == 0:
            SAMPLE_RATE = TEMP_SAMPLE_RATE
        elif SAMPLE_RATE != TEMP_SAMPLE_RATE:
            print('Wrong sample rate is detected in input files.')
            print('File = ' + p)
            print('Expected sample rate = ' + SAMPLE_RATE)
            print('Actual sample rate = ' + TEMP_SAMPLE_RATE)
            exit(1)

        # 無音じゃないかチェック
        if is_slient_samples(TEMP_INPUT):
            continue

        # ロードした波形をリストに追加
        INPUTS.append({'stereo': TEMP_INPUT, 'path': p})

    # ファイル名末尾の番号でソート(末尾番号がついてなければソートしない)
    is_valid_order = False
    for i in INPUTS:
        _, stem, _ = decompose_path(i['path'])
        detected_order = detect_stem_tail_number(stem)
        if detected_order != 0:
            is_valid_order = True
        i['order'] = detected_order
    if is_valid_order:
        INPUTS = sorted(INPUTS, key=lambda input: input['order'])

    ''' TODO
    - 外部設定ファイル化したい
    - 末尾側ピーク位置を指定したい
    - 先頭側ピークと末尾側ピークの気の利いた名前を考えないと
    - 一定の波形が続くのだからピーク間間隔の中央値をとればピッチを検出できそう
    '''
    # 補正処理の挙動を記述
    parameters = correct_bass_parameters()
    parameters.samplerate = SAMPLE_RATE
    parameters.bpm = 170
    parameters.head_click_offset = Fraction(1, 128)
    parameters.note_length = Fraction(12, 128)
    parameters.post_offset = Fraction(-3, 256)
    parameters.detection_offset_in_samples = 1024

    # 補正処理呼び出し
    if correct_bass(INPUTS, parameters):
        print('(error) : Some error has occured.')
        exit(1)

    # 補正結果を１つの波形に結合
    composed_low = numpy.empty((0, 2), INTERNAL_SAMPLE_FORMAT)
    composed_high = numpy.empty((0, 2), INTERNAL_SAMPLE_FORMAT)
    composed_full = numpy.empty((0, 2), INTERNAL_SAMPLE_FORMAT)
    for i in INPUTS:
        composed_low = numpy.r_[composed_low, i['total_corrected_low']]
        composed_high = numpy.r_[composed_high, i['total_corrected_high']]
        composed_full = numpy.r_[composed_full, i['total_corrected_full']]

    # 補正をかけたベース波形を出力
    directory, _, extension = decompose_path(INPUTS[0]['path'])
    output_path_low = compose_path(directory, OUTPUT_FILE_PREFIX + 'output_low', extension)
    output_path_high = compose_path(directory, OUTPUT_FILE_PREFIX + 'output_high', extension)
    output_path_full = compose_path(directory, OUTPUT_FILE_PREFIX + 'output_full', extension)
    make_directory_exist(output_path_low)
    make_directory_exist(output_path_high)
    make_directory_exist(output_path_full)
    save_samples(output_path_low, composed_low, SAMPLE_RATE, EXPORT_SAMPLE_FORMAT)
    save_samples(output_path_high, composed_high, SAMPLE_RATE, EXPORT_SAMPLE_FORMAT)
    save_samples(output_path_full, composed_full, SAMPLE_RATE, EXPORT_SAMPLE_FORMAT)

    # 正常終了
    exit(0)
