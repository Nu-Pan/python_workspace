import os
import sys
import glob
from fractions import Fraction
import configparser

import numpy
from scipy import signal

from details import *

# import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# constants
# ------------------------------------------------------------------------------

# 出力ファイルにつけるプリフィックス
OUTPUT_FILE_PREFIX = 'output_'

# ------------------------------------------------------------------------------
# correct_kick パラメータクラス
# ------------------------------------------------------------------------------

class correct_kick_parameters:
    def __init__(self):
        self.samplerate = 0
        self.bpm = 128
        self.snap_offset = Fraction(0, 1)
        self.is_verbose = False

# ------------------------------------------------------------------------------
# correct_kick メイン実装
# ------------------------------------------------------------------------------

def correct_kick(input, parameters):
    '''
    パラメータだ指定された時点にゼロクロスポイントが来るように\n
    inputs に含まれるキック波形を補正する。\n
    補正処理は再生速度の変化（リサンプリング）として実装される。\n
    補正処理は in-place で行われる。\n
    \n
    inputs の形式については correct_bass の呼び出し箇所を参照。\n
    '''
    # TODO verbose モードを実装

    # モノラル波形とそのローパス波形を事前に生成
    if parameters.is_verbose:
        print('*** create monoral samples ***')
        print('path=' + input['path'])
    temp = input['stereo']
    temp = (temp[:, 0] + temp[:, 1]) / 2.0
    input['monoral_sample'] = temp

    # TODO パラメータチェック

    if parameters.is_verbose:
        print('*** convert parameters ***')

    # スナップタイミングオフセットをサンプル数単位に変換
    sanpe_offset_in_samples = nl2sl(parameters.snap_offset, parameters.bpm, parameters.samplerate)
    if parameters.is_verbose:
        print('sanpe_offset_in_samples = %d' % sanpe_offset_in_samples)

    # 入力キックサンプル列のゼロクロスポイントをすべて列挙
    zero_cross_points = detect_zerocross_points(input['monoral_sample'])

    # 検出されたゼロクロスポイントのうち「スナップタイミングよりも後でかつ最小」のものを選択
    source_offset_in_samples = zero_cross_points[sanpe_offset_in_samples < zero_cross_points][0]

    # リサンプル実行
    input['corrected'] = signal.resample(input['stereo'], int(len(input['monoral_sample']) * sanpe_offset_in_samples / source_offset_in_samples))

    # 正常終了
    return False

# ------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------

def print_usage():
    'このプログラムの使い方を表示'
    print('Usage : python correct_bass.py <ini file path> <wav file path>')

if __name__ == '__main__':
    # 引数のエイリアスを作る
    INPUT_PATHS = sys.argv[1:]
    # 引数の数のチェック
    if len(INPUT_PATHS)!=2:
        print_usage()
        exit(1)
    # ディレクトリ指定は NG
    if os.path.isdir(INPUT_PATHS[0]):
        print('(error) : specified path is directory. "%s".' % INPUT_PATHS[0])
    if os.path.isdir(INPUT_PATHS[1]):
        print('(error) : specified path is directory. "%s".' % INPUT_PATHS[1])
    # 更にエイリアス
    if decompose_path(INPUT_PATHS[0])[2] == ".wav":
        INPUT_WAV_FILES = INPUT_PATHS[0]
        INPUT_INI_FILE = INPUT_PATHS[1]
    else:
        INPUT_WAV_FILES = INPUT_PATHS[1]
        INPUT_INI_FILE = INPUT_PATHS[0]        

    # wav ファイルをメモリ上にロード
    INPUTS, SAMPLERATE = load_wav_files([INPUT_WAV_FILES], INTERNAL_SAMPLE_FORMAT)
    INPUT = INPUTS[0]

    # 補正処理の挙動を設定ファイルから読み込み
    config = configparser.ConfigParser()
    config.read(INPUT_INI_FILE)
    parameters = correct_kick_parameters()
    parameters.samplerate = SAMPLERATE
    parameters.bpm = int(config['specific']['bpm'])
    parameters.snap_offset = Fraction(config['specific']['snap_offset'])
    parameters.is_verbose = bool(config['empirical']['is_verbose'])

    # 補正処理呼び出し
    if correct_kick(INPUT, parameters):
        print('(error) : Some error has occured.')
        exit(1)

    # 補正をかけたキック波形を出力
    directory, name, extension = decompose_path(INPUT['path'])
    output_path = compose_path(directory, OUTPUT_FILE_PREFIX + name, extension)
    print('output_path = ' + output_path)
    save_samples(output_path, INPUT['corrected'], SAMPLERATE, EXPORT_SAMPLE_FORMAT)

    # 正常終了
    exit(0)
