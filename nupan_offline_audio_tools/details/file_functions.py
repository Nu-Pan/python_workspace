import sys
import os
import glob
import re
import soundfile as sf

from .default_constants import *

def decompose_path(path):
    'path を (directory, stem, extension) に分解'
    directory, base_name = os.path.split(path)
    stem, extension = os.path.splitext(base_name)
    return (directory, stem, extension)

def compose_path(directory, stem, extension):
    '(directory, stem, extension) からパスを合成'
    return os.path.join(directory, stem + extension)

def make_directory_exist(path):
    '''
    指定パスのディレクトリが存在しなければ作成する。
    ファイルパスを渡すこともできる
    '''
    if len(path) == 0:
        return
    if not os.path.isdir(path):
        directory, _, _ = decompose_path(path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def detect_stem_tail_number(stem):
    '''
    ファイル名のステム末尾についている番号を検出する。
    見つからなかった場合は 0 を返す。
    '''
    m = re.match('^(.+?)(\\d+)$', stem)
    if m is None:
        return 0
    groups = m.groups()
    if len(groups) == 1:
        return 0
    else:
        return int(groups[1])

def pad_stem_zero(stem, number_of_digit):
    '''
    ファイル名のステム stem の末尾に付いている番号を digit 桁にゼロパディングする。
    末尾に番号の付いていないステムが与えられた場合は入力をそのまま返す。
    '''
    m = re.match('^(.+?)(\\d+)$', stem)
    if m is None:
        return stem
    groups = m.groups()
    if len(groups) == 1:
        return stem
    else:
        return groups[0] + groups[1].zfill(number_of_digit)

def load_samples(samples_path, internal_sample_format=INTERNAL_SAMPLE_FORMAT):
    '''
    wav ファイルからサンプル列をロードする
    '''
    # 存在チェック
    if not os.path.exists(samples_path):
        raise Exception('Specified file "%s" has not existed.' % samples_path)
    # ロード
    input_samples, sample_rate = sf.read(file=samples_path, dtype=internal_sample_format)
    return input_samples, sample_rate

def save_samples(samples_path, samples, samplerate, export_sample_format=EXPORT_SAMPLE_FORMAT):
    '''
    wav ファイルにサンプル列をセーブする
    '''
    make_directory_exist(samples_path)
    sf.write(samples_path, samples, samplerate, export_sample_format)
