import random
import numpy as np

from nn.test_network import TestNetwork
from sound.sound import Sound


sound_in = Sound()
sound_in.load_from_file('train_files/roam/in.wav')
stft_in = np.abs(sound_in.stft)

sound_out = Sound()
sound_out.load_from_file('train_files/roam/out.wav')
stft_out = sound_out.stft

length = min(stft_in.shape[1], stft_out.shape[1])

stft_in = stft_in[:, :length]
stft_out = stft_out[:, :length]

batches = 10
height = stft_in.shape[0]
width = stft_in.shape[1]

nn = TestNetwork(batches, 100, 100)

xs = stft_in
ys = stft_out

# xs = xs.reshape([1, height, width, 1])
# ys = ys.reshape([1, height, width, 1])

for i in range(10000):
    xs = []
    ys = []
    for j in range(batches):
        k = random.randint(0, len(stft_in)-100)
        xs.append(stft_in[k:k+100, :100].tolist())
        ys.append(stft_out[k:k+100, :100].tolist())

    xs = np.array(xs)
    ys = np.array(ys)

    xs = xs.reshape([batches, 100, 100, 1])
    ys = ys.reshape([batches, 100, 100, 1])

    nn.train(xs, ys)
    if i % 100 == 0 or i == 1000 - 1:
        print('Step {:04d} : error: {}'.format(
            i+1, np.sqrt(nn.evaluate(xs, ys))))


# EOF
