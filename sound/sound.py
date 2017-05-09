import numpy as np
from scipy.io import wavfile


class Sound:
    def __init__(self, filename):
        self.rate, self.data = wavfile.read(filename)
        
        # self.normalized = [(x / 2**8.)*2 - 1 for x in self.data]
        self.normalized = [float(x + 32768) / 32768.0/2.0 for x in self.data]

        # self.fft = np.fft.fft(self.normalized)
        # length = int(len(self.fft)/2)
        # self.fft = self.fft[:(length - 1)]

    @staticmethod
    def write_to_file(filename, data, rate):
        data = np.array(data).flatten()
        data = np.array(
            [x*32768*2 - 32768 for x in data]
        )
        data = data.astype('int16')

        # print(np.min(data))
        # print(np.max(data))

        wavfile.write(filename, rate, data)


# EOF
