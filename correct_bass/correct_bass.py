import sys
import os
import glob
import re

import numpy
from scipy import signal
import soundfile as sf

import correct_bass_impl

import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# constants
# ------------------------------------------------------------------------------

# 計算に使うサンプルのフォーマット
INTERNAL_SAMPLE_FORMAT = 'float64'

# 波形ファイルセーブ時のフォーマット
EXPORT_SAMPLE_FORMAT = 'FLOAT'

# 出力ファイルにつけるプリフィックス
OUTPUT_FILE_PREFIX = 'output\\'

# ファイル名に付属する番号の桁数
FILESTEM_NUMBER_OF_DIGIT = 3

# ------------------------------------------------------------------------------
# internal methods
# ------------------------------------------------------------------------------

def decompose_path(path):
    'path を (directory, stem, extension) に分解'
    directory, base_name = os.path.split(path)
    stem, extension = os.path.splitext(base_name)
    return (directory, stem, extension)

def compose_path(directory, stem, extension):
    '(directory, stem, extension) からパスを合成'
    return os.path.join(directory, stem + extension)

def print_usage():
    'このプログラムの使い方を表示'
    print('Too few number of arguments(up to 1)')
    print('Usage1 : python correct_bass.py <kick_sample>.wav <bass_sample_1>.wav ... <bass_sample_N>.wav')
    print('Usage2 : python correct_bass.py <direcyory path>')
    print('details :')
    print('In usage2, <directory path> must be directory that contain kick & bass wav file.')
    print('Also, Need to kick wav filename has "kick" included (In bass wav file, "bass" it is).')

def make_directory_exist(input_path):
    '''
    指定パスのディレクトリが存在しなければ作成する。
    ファイルパスを渡すこともできる
    '''
    if len(directory) == 0:
        return
    if not os.path.exists(directory):
        os.makedirs(directory)

def pad_stem_zero(stem, number_of_digit):
    '''
    ファイル名のステム stem の末尾に付いている番号を digit 桁にゼロパディングする。
    '''
    m = re.search('^(.+?)(\\d+)$', stem)
    groups = m.groups()
    if len(groups) == 1:
        return stem
    else:
        return groups[0] + groups[1].zfill(number_of_digit)

# ------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------

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
        KICK_FILE = [x for x in FILES if 'kick' in x.lower()]
        BASS_FILES = [x for x in FILES if 'bass' in x.lower()]

        # 見つかったファイル数のチェック
        if len(KICK_FILE) == 0:
            print('(error) : In usage2, No kick wav file in directory "%s".' % INPUT_DIR)
            print_usage()
            exit(1)
        elif 1 < len(KICK_FILE):
            print('(error) : In usage2, Multiple kick wav file in directory "%s".' % INPUT_DIR)
            print_usage()
            exit(1)
        if len(BASS_FILES) == 0:
            print('(error) : In usage2, No bass wav file in directory "%s".' % INPUT_DIR)
            print_usage()
            exit(1)

        # 引数リストを生成
        INPUT_PATHS = []
        INPUT_PATHS.extend(KICK_FILE)
        INPUT_PATHS.extend(BASS_FILES)

    # 引数チェック
    if len(INPUT_PATHS) < 2:
        print_usage()
        exit(1)

    # 指定ファイル全てメモリ上にロード
    SAMPLE_RATE = 0
    INPUTS = []
    for p in INPUT_PATHS:
        # 存在チェック
        if not os.path.exists(p):
            print('Specified file "%s" has not existed.' % p)
            print_usage()
            exit(1)

        # ロード
        TEMP_INPUT, TEMP_SAMPLE_RATE = sf.read(file=p, dtype=INTERNAL_SAMPLE_FORMAT)

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
        INPUTS.append({'stereo_samples': TEMP_INPUT, 'path': p})

    # TODO 引数で指定できるべき
    # 補正処理の挙動を記述
    parameters = correct_bass_impl.correct_bass_parameters()
    parameters.samplerate = SAMPLE_RATE
    parameters.is_explicit_click_offset_mode = True
    parameters.click_length_num = 1
    parameters.click_length_dom = 128
    parameters.bpm = 170
    parameters.force_offset_in_sec = 0.3

    # 補正処理呼び出し
    if correct_bass_impl.correct_bass(INPUTS, parameters):
        print('(error) : Some error has occured.')
        exit(1)
    
    # 補正結果を１つの波形に結合
    composed_low = numpy.empty((0, 2), INTERNAL_SAMPLE_FORMAT)
    composed_high = numpy.empty((0, 2), INTERNAL_SAMPLE_FORMAT)
    for i in INPUTS[1:]:
        composed_low = numpy.r_[composed_low, i['total_corrected_low']]
        composed_high = numpy.r_[composed_high, i['total_corrected_high']]

    # 補正をかけたベース波形を出力
    directory, stem, extension = decompose_path(INPUTS[1]['path'])
    output_path_low = compose_path(directory, OUTPUT_FILE_PREFIX + 'output_low', extension)
    output_path_high = compose_path(directory, OUTPUT_FILE_PREFIX + 'output_high', extension)
    make_directory_exist(output_path_low)
    make_directory_exist(output_path_high)
    sf.write(file=output_path_low, data=composed_low, samplerate=SAMPLE_RATE, subtype=EXPORT_SAMPLE_FORMAT)
    sf.write(file=output_path_high, data=composed_high, samplerate=SAMPLE_RATE, subtype=EXPORT_SAMPLE_FORMAT)

    # 正常終了
    exit(0)
