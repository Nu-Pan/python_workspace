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
    print('Usage1 : python cutoff_extreme_band.py <sample>.wav <sample_1>.wav ... <sample_N>.wav')
    print('Usage2 : python correct_bass.py <direcyory path>')

if __name__ == '__main__':
    # 引数のエイリアスを作る
    INPUT_PATHS = sys.argv[1:]

    # 引数チェック
    if len(INPUT_PATHS) < 1:
        print_usage()
        exit(1)

    # ディレクトリ指定の場合は中身を探索して引数を作る
    if os.path.isdir(INPUT_PATHS[0]):
        INPUT_DIR = INPUT_PATHS[0]
        INPUT_PATHS = glob.glob(os.path.join(INPUT_DIR, '*.wav'))

    # 順番に処理かけて保存する
    for i in INPUT_PATHS:
        # サンプル列をファイルからロード
        try:
            TEMP_INPUT, TEMP_SAMPLE_RATE = load_samples(i, INTERNAL_SAMPLE_FORMAT)
        except Exception as err:
            print(err)
            print_usage()
            raise

        # ultra-low と ultra-high を除去
        TEMP_INPUT = cutoff_extreme_band(TEMP_INPUT, TEMP_SAMPLE_RATE)
        # 200Hz 以上を抽出
        TEMP_INPUT = extract_high_band(TEMP_INPUT, TEMP_SAMPLE_RATE, 200, 0)

        # サンプル列をファイルにセーブ
        directory, stem, extension = decompose_path(i)
        output_path = compose_path(directory, stem + '_bassupper', extension)
        save_samples(output_path, TEMP_INPUT, TEMP_SAMPLE_RATE, EXPORT_SAMPLE_FORMAT)

    # 正常終了
    exit(0)
