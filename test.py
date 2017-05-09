import os

from separator import Separator


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

Separator().train(
    [
        'train_files/happy/in.wav',
        'train_files/happy/out.wav'
    ],
    batches=100,
    samples=100,
    iterations=500,
    name='test'
)
