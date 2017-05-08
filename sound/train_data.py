import random

import numpy as np

from .sound import Sound


class TrainData:
    def __init__(self, num_samples):
        self.in_data = []
        self.out_data = []
        self.num_samples = num_samples

    def add_test_files(self, in_file, out_file):
        input_data = Sound(in_file).normalized
        output_data = Sound(out_file).normalized

        length = max(len(input_data), len(output_data))
        length = int(np.ceil(length / self.num_samples)
                     * self.num_samples)

        input_data = np.lib.pad(
            input_data, (0, length - len(input_data)), 'constant')
        output_data = np.lib.pad(
            output_data, (0, length - len(output_data)), 'constant')

        in_data = input_data.reshape(-1, self.num_samples)
        out_data = output_data.reshape(-1, self.num_samples)

        self.in_data.extend(in_data)
        self.out_data.extend(out_data)

    def get_random(self, num_batches):
        index = random.randint(0, len(self.in_data) - num_batches)
        xs = self.in_data[index: index + num_batches]
        ys = self.out_data[index: index + num_batches]
        return xs, ys

    # def fill_zeros(self):
    #     """Make each row of data equal length by filling zeros"""
    #     lengths = np.array([len(x) for x in self.data])
    #     mask = np.arange(lengths.max()) < lens[:, None]
    #     result = np.zeros(mask.shape)
    #     result[mask] = np.concatenate(self.data)
    #     self.data = result


# EOF
