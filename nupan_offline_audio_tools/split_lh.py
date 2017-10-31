import os
import sys

import numpy

from details import *

# ------------------------------------------------------------------------------
# constants
# ------------------------------------------------------------------------------

# 出力ファイルにつけるプリフィックス
OUTPUT_FILE_PREFIX = 'output\\'

# 出力ファイル名に付属する番号の桁数
FILESTEM_NUMBER_OF_DIGIT = 3

# ------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------

def print_usage():
    'このプログラムの使い方を表示'
    print('Usage : python split_lh.py <sample>')

if __name__ == '__main__':
    # 引数チェック
    if len(sys.argv) != 2:
        print_usage()
        exit(1)

    # 引数のエイリアスを作る
    INPUT_PATH = sys.argv[1]

    # サンプル列をファイルからロード
    try:
        TEMP_INPUT, TEMP_SAMPLE_RATE = load_samples(INPUT_PATH, INTERNAL_SAMPLE_FORMAT)
    except Exception as err:
        print(err)
        print_usage()
        raise

    # low, high に分離
    SAMPLES_LOW = extract_low_band(TEMP_INPUT, TEMP_SAMPLE_RATE)
    SAMPLES_HIGH = extract_low_band(TEMP_INPUT, TEMP_SAMPLE_RATE)

    # サンプル列をファイルにセーブ
    DIRECTORY, STEM, EXTENSION = decompose_path(INPUT_PATH)
    LOW_OUTPUT_PATH = compose_path(DIRECTORY, STEM + '_low', EXTENSION)
    save_samples(LOW_OUTPUT_PATH, SAMPLES_LOW, TEMP_SAMPLE_RATE, EXPORT_SAMPLE_FORMAT)
    HIGH_OUTPUT_PATH = compose_path(DIRECTORY, STEM + '_high', EXTENSION)
    save_samples(HIGH_OUTPUT_PATH, SAMPLES_HIGH, TEMP_SAMPLE_RATE, EXPORT_SAMPLE_FORMAT)

    # 正常終了
    exit(0)
