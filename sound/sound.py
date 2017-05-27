import numpy as np
from scipy.io import wavfile
from scipy import signal


class Sound:
    def load_from_file(self, filename):
        self.rate, self.data = wavfile.read(filename)
        _, _, self.stft = signal.stft(self.data, self.rate)

    def load_stft(self, stft, rate):
        self.rate, self.stft = rate, stft
        _, self.data = signal.istft(self.stft, self.rate)

    def save(self, filename):
        wavfile.write(filename, self.rate, self.data)
