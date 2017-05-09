# Sound separator using neural network

The test example tries a convolutional neural network to train a
convolutional neural network to separate recorded sound mixed
with music and evaluates it with test data itself.

The example uses tensorflow and scipy modules.

The example code is contained in `test.py`.

```bash
$ python test.py
```

We can also train and run the neural network from the
`separator.py` module.

Following command will save use *input.wav* and *output.wav* files
for training and save the trained network in the name *test*.

Using same name in future *train* commands will continue the training
in the same network.

```bash
$ python separator.py train input.wav output.wav --name test
```

You can also configure the network parameters.

```bash
$ python separator.py train input.wav output.wav --name test --batches 100 --samples 100 --iterations 500
```

To run the neural network to get output, use the following command.

```bash
$ python separator.py run input.wav --name test
$ python separator.py run input.wav --name test --batches 100 --samples 100
```

The output is saved in *out.wav* file.
