import sys
from os import path
import glob

import numpy

import soundfile as sf

# ------------------------------------------------------------------------------
# 定数
# ------------------------------------------------------------------------------

# 出ry校ファイルパス
OUTPUT_FILE_STEM = 'output'
OUTPUT_FILE_EXTENSION = '.wav'

# 計算に使うサンプルのフォーマット
INTERNAL_SAMPLE_FORMAT = 'float64'

# 波形ファイルセーブ時のフォーマット
EXPORT_SAMPLE_FORMAT = 'FLOAT'

# ------------------------------------------------------------------------------
# internal methods
# ------------------------------------------------------------------------------

def decompose_path(path):
    'path を (directory, stem, extension) に分解'
    directory, base_name = path.split(path)
    stem, extension = path.splitext(base_name)
    return (directory, stem, extension)

def compose_path(directory, stem, extension):
    '(directory, stem, extension) からパスを合成'
    return path.join(directory, stem + extension)

# ------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------

if __name__=='__main__':
    # 引数チェック
    if len(sys.argv) != 2:
        print('Invalid number of arguments.')
        print('Specified : %d' % len(sys.argv))
        exit(1)

    # 引数のエイリアス
    INPUT_PATH = sys.argv[1]

    # 引数チェック
    if not path.exists(INPUT_PATH):
        print('"%s" is not existence.' % INPUT_PATH)
        exit(1)
    if not path.isdir(INPUT_PATH):
        print('"%s" is not directory path.' % INPUT_PATH)
        exit(1)

    # 指定ディレクトリ下の wav ファイルを全て列挙
    FILES = glob.glob(path.join(INPUT_PATH, '*.wav'))

    # 全ての wav ファイルを１つに結合
    result = numpy.empty((0, 2), INTERNAL_SAMPLE_FORMAT)
    SAMPLE_RATE = 0
    for p in FILES:
        # ロード & 結合
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
        # 結合
        result = numpy.r_[result, TEMP_INPUT]

    # 結合したファイルを出力
    output_path = compose_path(INPUT_PATH, OUTPUT_FILE_STEM, OUTPUT_FILE_EXTENSION)    
    sf.write(file=output_path, data=result, samplerate=SAMPLE_RATE, subtype=EXPORT_SAMPLE_FORMAT)

    # 正常終了
    exit(0)
