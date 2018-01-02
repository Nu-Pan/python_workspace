import numpy
from scipy import signal
import soundfile as sf

from details import *

def normalize_frequency(frequency, sample_rate):
    '''
    frequency[Hz] を [0, sample_rate/2] -> [0.0, 1.0] の値域にマップする。
    '''
    return frequency / (sample_rate / 2.0)

def load_wav_and_ini(dir_path):
    # 指定ファイル全てメモリ上にロード
    SAMPLERATE = 0
    INPUTS = []
    for p in WAV_FILES:
        # ロード
        try:
            TEMP_INPUT, TEMP_SAMPLE_RATE = load_samples(p, INTERNAL_SAMPLE_FORMAT)
        except Exception as err:
            print(err)
            print_usage()
            raise
        # サンプルレートをチェック
        if SAMPLERATE == 0:
            SAMPLERATE = TEMP_SAMPLE_RATE
        elif SAMPLERATE != TEMP_SAMPLE_RATE:
            print('Wrong sample rate is detected in input files.')
            print('File = ' + p)
            print('Expected sample rate = ' + SAMPLERATE)
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
    # 補正処理の挙動を設定ファイルから読み込み
    config = configparser.ConfigParser()
    config.read(INI_FILES[0])
    # TODO 結果戻す


if __name__ == '__main__':
    # TODO ロード系処理が correct_bass と被ってるので共通化

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

    # 指定ファイル全てメモリ上にロード
    SAMPLERATE = 0
    INPUTS = []
    for p in WAV_FILES:
        # ロード
        try:
            TEMP_INPUT, TEMP_SAMPLE_RATE = load_samples(p, INTERNAL_SAMPLE_FORMAT)
        except Exception as err:
            print(err)
            print_usage()
            raise
        # サンプルレートをチェック
        if SAMPLERATE == 0:
            SAMPLERATE = TEMP_SAMPLE_RATE
        elif SAMPLERATE != TEMP_SAMPLE_RATE:
            print('Wrong sample rate is detected in input files.')
            print('File = ' + p)
            print('Expected sample rate = ' + SAMPLERATE)
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

    # LR アンパック（２つのモノラルサンプル列に分離）
    INPUT_SAMPLES_L = INPUT_SAMPLES[:, 0]
    INPUT_SAMPLES_R = INPUT_SAMPLES[:, 1]

    # 定数
    NORMALIZED_CUTOFF_FREQUENCY = normalize_frequency(60, SAMPLERATE)
    FILTER_ORDER = 8

    # 200Hz, -12dB/Oct のロー/ハイパスフィルタを設計
    LPF_B, LPF_A = signal.butter(FILTER_ORDER, NORMALIZED_CUTOFF_FREQUENCY, 'low')
    HPF_B, HPF_A = signal.butter(FILTER_ORDER, NORMALIZED_CUTOFF_FREQUENCY, 'high')

    # ゼロ位相フィルタリングを適用
    OUTPUT_LOW_L = signal.filtfilt(LPF_B, LPF_A, INPUT_SAMPLES_L)
    OUTPUT_LOW_R = signal.filtfilt(LPF_B, LPF_A, INPUT_SAMPLES_R)
    OUTPUT_HIGH_L = signal.filtfilt(HPF_B, HPF_A, INPUT_SAMPLES_L)
    OUTPUT_HIGH_R = signal.filtfilt(HPF_B, HPF_A, INPUT_SAMPLES_R)

    # フィルタリング結果を LR パック（ステレオサンプル列に合体）
    OUTPUT_LOW = numpy.c_[OUTPUT_LOW_L, OUTPUT_LOW_R]
    OUTPUT_HIGH = numpy.c_[OUTPUT_HIGH_L, OUTPUT_HIGH_R]

    # オリジナルとの差分を生成
    OUTPUT_DIFF = INPUT_SAMPLES - (OUTPUT_LOW + OUTPUT_HIGH)

    # 処理結果を書き出し
    sf.write(file='test_out_32bit_float_low.wav', data=OUTPUT_LOW, samplerate=SAMPLERATE, subtype='FLOAT')
    sf.write(file='test_out_32bit_float_high.wav', data=OUTPUT_HIGH, samplerate=SAMPLERATE, subtype='FLOAT')
