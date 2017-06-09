import soundfile as sf

if __name__ == '__main__':
    print(sf.default_subtype('WAV'))
    print(sf.available_subtypes('WAV'))
#    DATA, SAMPLERATE = sf.read(file='test_in_32bit_float.wav', dtype='float32')
#    sf.write(file='test_out_32bit_float.wav', data=DATA, samplerate=SAMPLERATE, subtype='PCM_16')
# read int -> float
#    DATA, SAMPLERATE = sf.read(file='test_in_16bit_int.wav', dtype='float32')
#    sf.write(file='test_out_32bit_float.wav', data=DATA, samplerate=SAMPLERATE, subtype='FLOAT')
# read float -> int
#    DATA, SAMPLERATE = sf.read(file='test_in_32bit_float.wav', dtype='int16')
#    sf.write(file='test_out_16bit_int.wav', data=DATA, samplerate=SAMPLERATE, subtype='PCM_16')
# write int -> float
#    DATA, SAMPLERATE = sf.read(file='test_in_16bit_int.wav', dtype='int16')
#    sf.write(file='test_out_32bit_float.wav', data=DATA, samplerate=SAMPLERATE, subtype='FLOAT')
# write float -> int
    print(sf.info('test_in_32bit_float.wav', False))
    DATA, SAMPLERATE = sf.read(file='test_in_32bit_float.wav', dtype='float32')
    sf.write(file='test_out_16bit_int.wav', data=DATA, samplerate=SAMPLERATE, subtype='PCM_16')
