import numpy as np

from .sound import Sound


class RunData:
    def __init__(self, num_samples, in_file):
        self.input = Sound(in_file)
        
        data = self.input.normalized
        length = int(np.ceil(len(data) / num_samples)
                     * num_samples)
        data = np.lib.pad(
            data, (0, length - len(data)), 'constant')

        self.data = data.reshape(-1, num_samples)
        self.index = 0

        self.output = []

    def get_next(self, num_batches):
        xs = self.data[self.index: self.index + num_batches]
        self.index += num_batches
        return xs

    def get_num(self, num_batches):
        return int(len(self.data) / num_batches)

    def collect_output(self, y):
        self.output.extend(list(y))

    def save_sound(self, filename):
        Sound.write_to_file(filename, self.output, self.input.rate)

# EOF
