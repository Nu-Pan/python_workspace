import os
import sys
import glob
from fractions import Fraction
import configparser

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
        self.mode = 'zero-cross'
        self.detection_offset_iden_samples = 2**13
        self.is_verbose = False

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
    if parameters.is_verbose:
        print('*** create monoral & lowpass samples ***')
    for i in inputs:
        if parameters.is_verbose:
            print('path=' + i['path'])
        temp = i['stereo']
        temp = (temp[:, 0] + temp[:, 1]) / 2.0
        i['monoral_sample'] = temp
        i['monoral_lowband_sample'] = apply_zplr(temp, 'low', 200, parameters.samplerate)

    # TODO パラメータチェック

    if parameters.is_verbose:
        print('*** convert parameters ***')

    # 渡された各パラメータをサンプル数に変換
    head_click_offset_in_samples = nl2sl(parameters.head_click_offset, parameters.bpm, parameters.samplerate)
    if parameters.is_verbose:
        print('head_click_offset_in_samples = %d' % head_click_offset_in_samples)

    # 検出クリックオフセット（サンプル数単位）を定義
    detection_head_click_offset_in_samples = head_click_offset_in_samples + parameters.detection_offset_in_samples
    if parameters.is_verbose:
        print('detection_head_click_offset_in_samples = %d' % detection_head_click_offset_in_samples)
    
    # 波形のクリック位置が最適になるように処理する
    if parameters.is_verbose:
        print('*** correct offsets ***')
    for i in inputs:
        if parameters.is_verbose:
            print('path=' + i['path'])
        # 波形から「クリック」位置を検出
        if parameters.mode == 'zero-cross':
            detected_click = detect_positive_zero_cross(i['monoral_lowband_sample'])
        elif parameters.mode == 'extrema':
            detected_click = detect_positive_extrema(i['monoral_lowband_sample'])
        else:
            raise RuntimeError('Unknown mode string : ' + parameters.mode)
        if parameters.is_verbose:
            print('detected_click.size=%d' % detected_click.size)
            print(detected_click)
        # 波形中の最適クリックオフセットに最も近いクリックを選択
        actual_head_click_offset = get_nearest_value(detected_click, detection_head_click_offset_in_samples)
        if parameters.is_verbose:
            print('detection_head_click_offset_in_samples=%d' % detection_head_click_offset_in_samples)        
            print('actual_head_click_offset=%d' % actual_head_click_offset)        
        # オフセットを実行
        i['click_corrected'] = shift_forward_and_padding(i['stereo'], actual_head_click_offset - head_click_offset_in_samples)

    # ローとハイに分離
    if parameters.is_verbose:
        print('*** split low / high ***')
    for i in inputs:
        i['click_corrected_low'] = apply_zplr(i['click_corrected'], 'low', 200, parameters.samplerate)
        i['click_corrected_high'] = apply_zplr(i['click_corrected'], 'high', 200, parameters.samplerate)

    # 結果に名前をつける
    if parameters.is_verbose:
        print('*** save results ***')
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
    print('Usage : python correct_bass.py <direcyory path>')
    print('<directory path> must be directory that contain ".wav" file and config ".ini" file.')
    print('Directory allow to contain multiple ".wav" files.')
    print('Directory allow to contain single ".ini" file.')

if __name__ == '__main__':
    # 引数のエイリアスを作る
    INPUT_PATHS = sys.argv[1:]
    # 引数の数のチェック
    if len(INPUT_PATHS)!=1:
        print_usage()
        exit(1)
    # 非ディレクトリ指定は NG
    if not os.path.isdir(INPUT_PATHS[0]):
        print('(error) : In usage2, specified path is not directory. "%s".' % INPUT_PATHS[0])
    # 更にエイリアス
    INPUT_DIR = INPUT_PATHS[0]

    # 指定ディレクトリ中の wav ファイルを列挙
    WAV_FILES = find_wav_files(INPUT_DIR)
    if WAV_FILES is None:
        exit(1)

    # 指定ファイル全てメモリ上にロード
    INPUTS, SAMPLERATE = load_wav_files(WAV_FILES, INTERNAL_SAMPLE_FORMAT)

    # 指定ディレクトリ中の ini ファイルを列挙
    INI_FILE = find_ini_file(INPUT_DIR)
    if INI_FILE is None:
        exit(1)

    # 補正処理の挙動を設定ファイルから読み込み
    config = configparser.ConfigParser()
    config.read(INI_FILE)
    parameters = correct_bass_parameters()
    parameters.samplerate = SAMPLERATE
    parameters.bpm = int(config['specific']['bpm'])
    parameters.head_click_offset = Fraction(config['specific']['head_click_offset'])
    parameters.mode = config['specific']['mode']
    parameters.detection_offset_in_samples = int(config['empirical']['detection_offset_in_samples'])
    parameters.is_verbose = bool(config['empirical']['is_verbose'])

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
    save_samples(output_path_low, composed_low, SAMPLERATE, EXPORT_SAMPLE_FORMAT)
    save_samples(output_path_high, composed_high, SAMPLERATE, EXPORT_SAMPLE_FORMAT)
    save_samples(output_path_full, composed_full, SAMPLERATE, EXPORT_SAMPLE_FORMAT)

    # 正常終了
    exit(0)
