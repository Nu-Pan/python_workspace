import sys
import os
import glob
import re
import scipy.io.wavfile as wf
import numpy as np

from .default_constants import *
from .helper_functions import *

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
    if directory == '':
        return
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

def load_samples(samples_path, internal_sample_format):
    '''
    wav ファイルからサンプル列をロードする
    '''
    # 存在チェック
    if not os.path.exists(samples_path):
        raise Exception('Specified file "%s" has not existed.' % samples_path)
    # ロード
    sample_rate, input_samples = wf.read(samples_path)
    # フォーマットを指定のものに変換する
    if internal_sample_format == 'float64':
        converted_samples = input_samples.astype(np.float64)
    elif internal_sample_format == 'float32':
        converted_samples = input_samples.astype(np.float32)
    elif internal_sample_format == 'int32':
        converted_samples = input_samples.astype(np.int32)
    elif internal_sample_format == 'int16':
        converted_samples = input_samples.astype(np.int16)
    else:
        raise RuntimeError('Invalid data type string : ' + internal_sample_format)
    return converted_samples, sample_rate

def save_samples(samples_path, samples, samplerate, export_sample_format):
    '''
    wav ファイルにサンプル列をセーブする
    '''
    make_directory_exist(samples_path)
    # フォーマットを指定のものに変換する
    if export_sample_format == 'float32':
        converted_samples = samples.astype(np.float32)
    elif export_sample_format == 'int32':
        converted_samples = samples.astype(np.int32)
    elif export_sample_format == 'int16':
        converted_samples = samples.astype(np.int16)
    else:
        raise RuntimeError('Invalid data type string : ' + export_sample_format)
    wf.write(samples_path, samplerate, converted_samples)

def find_wav_files(dir_path):
    '''
    指定ディレクトリ内の wav ファイルを検索する。
    '''
    # 指定ディレクトリ中の wav ファイルを列挙
    wav_files = glob.glob(os.path.join(dir_path, '*.wav'))
    if len(wav_files) == 0:
        print('No wav file in directory "%s".' % dir_path)
        return None
    # ファイルステム末尾の番号でソート
    sorted_wav_files = sorted(wav_files, key=lambda path: detect_stem_tail_number(decompose_path(path)[1]))
    # 正常終了
    return sorted_wav_files

def find_ini_file(dir_path):
    '''
    指定ディレクトリ内の ini ファイルを検索する。
    '''
    # 指定ディレクトリ中の ini ファイルを列挙
    ini_files = glob.glob(os.path.join(dir_path, '*.ini'))
    if len(ini_files) == 0:
        print('No ini files in directory "%s".' % dir_path)
        return None
    elif 1 < len(ini_files):
        print('Too many ini files in directory "%s".' % dir_path)
        return None
    # 正常終了
    return ini_files[0]

def load_wav_files(wav_files_path, internal_sample_format):
    '''
    指定ファイル全てをメモリ上にロード
    '''
    samplerate = 0
    samples_list = []
    for p in wav_files_path:
        # ロード
        try:
            temp_input, temp_sampletate = load_samples(p, INTERNAL_SAMPLE_FORMAT)
        except Exception as err:
            print(err)
            raise
        # サンプルレートをチェック
        if samplerate == 0:
            samplerate = temp_sampletate
        elif samplerate != temp_sampletate:
            print('Wrong sample rate is detected in input files.')
            print('File = ' + p)
            print('Expected sample rate = ' + samplerate)
            print('Actual sample rate = ' + temp_sampletate)
            exit(1)
        # 無音サンプルはスキップ
        if is_slient_samples(temp_input):
            continue
        # ロードした波形をリストに追加
        samples_list.append({'stereo': temp_input, 'path': p})
    # 正常終了
    return samples_list, samplerate
