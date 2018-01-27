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

# ------------------------------------------------------------------------------
# structure definitions
# ------------------------------------------------------------------------------

class band_param:
    def __init__(self):
        self.lower_type = 'butter'
        self.lower_mode = 'high'
        self.lower_order = 2
        self.lower_freq = 20
        self.lower_is_zero_phase = True
        self.upper_type = 'butter'
        self.upper_mode = 'low'
        self.upper_order = 2
        self.upper_freq = 20000
        self.upper_is_zero_phase = True
        self.normalization_mode = 'none'
        self.normalization_target_override = False
        self.normalization_target = 0
        self.gain = 0
        self.is_file_out = False
        self.postfix = ''

# ------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    # 引数チェック
    if len(sys.argv)!=2:
        print('Invalid number of arguments.')
        exit(1)

    # エイリアス
    INPUT_DIR = sys.argv[1]

    # ディレクトリ以外の指定は NG
    if not os.path.isdir(INPUT_DIR):
        print('Specified path is not directory. "%s".' % INPUT_DIR)
        exit(1)

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
    output_file_prefix = config['global']['output_file_prefix']
    output_file_sufix = config['global']['output_file_sufix']
    is_serial_connection = config['global']['connection_mode'] == 'serial'
    band_params = []
    for section in config.sections():
        if section in ['global', 'DEFAULT'] :
            continue
        temp = band_param()
        temp.lower_type = config[section]['lower_type']
        temp.lower_mode = config[section]['lower_mode']
        temp.lower_order = int(config[section]['lower_order'])
        temp.lower_freq = float(config[section]['lower_freq'])
        temp.lower_is_zero_phase = string2bool(config[section]['lower_is_zero_phase'])
        temp.upper_type = config[section]['upper_type']
        temp.upper_mode = config[section]['upper_mode']
        temp.upper_order = int(config[section]['upper_order'])
        temp.upper_freq = float(config[section]['upper_freq'])
        temp.upper_is_zero_phase = string2bool(config[section]['upper_is_zero_phase'])
        temp.normalization_mode = config[section]['normalization_mode']
        temp.normalization_target_override = string2bool(config[section]['normalization_target_override'])
        temp.normalization_target = float(config[section]['normalization_target'])
        temp.gain = float(config[section]['gain'])
        temp.is_file_out = string2bool(config[section]['is_file_out'])
        temp.sufix = config[section]['sufix']
        band_params.append(temp)

    # 全ての wav ファイルに対してマルチバンド分離
    for i in INPUTS:
        temp_samples = i['stereo']
        for param in band_params:
            # バンド抽出
            band = create_same_empty(temp_samples)
            band[:] = temp_samples
            if not param.lower_type == 'bypass':
                band = apply_filter(band, param.lower_type, param.lower_mode, param.lower_order, param.lower_freq, SAMPLERATE, param.lower_is_zero_phase)
            if not param.upper_type == 'bypass':
                band = apply_filter(band, param.upper_type, param.upper_mode, param.upper_order, param.upper_freq, SAMPLERATE, param.upper_is_zero_phase)
            # 抽出した分をオリジナルから減算
            if is_serial_connection:
                temp_samples = temp_samples - band
            # バンド波形の基準量（ピークとかRMSとか）を計算
            if param.normalization_mode=='none':
                i['band_criteria_' + param.sufix] = 1.0
            elif param.normalization_mode=='peak':
                i['band_criteria_' + param.sufix] = convert_to_median_prak(band, time2sample(0.3, SAMPLERATE))
            elif param.normalization_mode=='rms':
                i['band_criteria_' + param.sufix] = convert_to_median_rms(band, time2sample(0.3, SAMPLERATE))
            else:
                print('Invalid normalization_mode in loaded .ini file. Pass through normalization and continue.')            
            # バンド波形を保存
            i['band_sample_' + param.sufix] = band

    # バンドごとにノーマライズを実行
    for param in band_params:
        # ノーマライズの基準量を決定
        if param.normalization_target_override:
            # オーバーライドの指定がある場合はその値を目標基準量にする
            target_criteria = to_ratio(param.normalization_target)
        else:
            # オーバーライドが指定されていなければ基準量の中央値を目標基準量とする
            criteria_array = []
            for i in INPUTS:
                criteria_array.append(i['band_criteria_' + param.sufix])
            criteria_array.sort()
            target_criteria = criteria_array[int(len(criteria_array)/2)]
        # 基準量が揃うように振幅を調整＋ゲインを適用
        for i in INPUTS:
            i['band_sample_' + param.sufix] = i['band_sample_' + param.sufix] * (target_criteria / i['band_criteria_' + param.sufix]) * to_ratio(param.gain)

    # ファイル出力
    for i in INPUTS:
        result_samples = create_same_zeros(i['stereo'])
        for param in band_params:
            band = i['band_sample_' + param.sufix]
            # 必要なバンド単位の結果をファイルアウト
            if param.is_file_out:
                dir_path, stem, ext = decompose_path(i['path'])
                outpath = dir_path + '\\' + output_file_prefix + stem + param.sufix + ext
                save_samples(outpath, band, SAMPLERATE, 'float')
            # 結果用変数に加算
            result_samples = result_samples + band
        # 全バンドの加算結果をファイル出力
        dir_path, stem, ext = decompose_path(i['path'])
        outpath = dir_path + '\\' + output_file_prefix + stem + output_file_sufix + ext
        save_samples(outpath, result_samples, SAMPLERATE, 'float')

    # 正常終了
    exit(0)
