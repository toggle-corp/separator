import numpy as np
from scipy.io import wavfile


class Sound:
    def __init__(self, filename):
        self.rate, self.data = wavfile.read(filename)
        self.normalized = [(x / 2**8.)*2 - 1 for x in self.data]

        # self.fft = np.fft.fft(self.normalized)
        # length = int(len(self.fft)/2)
        # self.fft = self.fft[:(length - 1)]
