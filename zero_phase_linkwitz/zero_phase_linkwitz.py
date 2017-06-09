import numpy
from scipy import signal
import soundfile as sf

def normalize_frequency(frequency, sample_rate):
    '''
    frequency[Hz] を [0, sample_rate/2] -> [0.0, 1.0] の値域にマップする。
    '''
    return frequency / (sample_rate / 2.0)

if __name__ == '__main__':
    # 元音源をロード
    INPUT_SAMPLES, SAMPLERATE = sf.read(file='SAW_BASS_D_A.wav', dtype='float64')

    # LR アンパック（２つのモノラルサンプル列に分離）
    INPUT_SAMPLES_L = INPUT_SAMPLES[:, 0]
    INPUT_SAMPLES_R = INPUT_SAMPLES[:, 1]

    # 定数
    NORMALIZED_CUTOFF_FREQUENCY = normalize_frequency(200, SAMPLERATE)
    FILTER_ORDER = 2

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
    sf.write(file='test_out_32bit_float_diff.wav', data=OUTPUT_DIFF, samplerate=SAMPLERATE, subtype='FLOAT')
