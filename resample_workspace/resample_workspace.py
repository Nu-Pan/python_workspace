import numpy
import sys
from scipy import signal
import soundfile as sf

def normalize_frequency(frequency, sample_rate):
    '''
    frequency[Hz] を [0, sample_rate/2] -> [0.0, 1.0] の値域にマップする。
    '''
    return frequency / (sample_rate / 2.0)

if __name__ == '__main__':
    # 引数のエイリアスを作る
    INPUT_PATHS = sys.argv[1:]

    # 引数チェック
    if len(INPUT_PATHS) < 1:
        print_usage()
        exit(1)

    # 元音源をロード
    INPUT_SAMPLES, SAMPLERATE = sf.read(file=INPUT_PATHS[0], dtype='float64')

    # LR アンパック（２つのモノラルサンプル列に分離）
    INPUT_SAMPLES_L = INPUT_SAMPLES[:, 0]
    INPUT_SAMPLES_R = INPUT_SAMPLES[:, 1]

    # リサンプル
    SCALED_L = signal.resample(INPUT_SAMPLES_L, int(INPUT_SAMPLES_L.size * 0.8))
    SCALED_R = signal.resample(INPUT_SAMPLES_R, int(INPUT_SAMPLES_R.size * 0.8))

    # フィルタリング結果を LR パック（ステレオサンプル列に合体）
    OUTPUT = numpy.c_[SCALED_L, SCALED_R]

    # 処理結果を書き出し
    sf.write(file='test_out_32bit_float.wav', data=OUTPUT, samplerate=SAMPLERATE, subtype='FLOAT')
