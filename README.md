# Sound separator using neural network

The test example tries a convolutional neural network to train a
convolutional neural network to separate recorded sound mixed
with music and evaluates it with test data itself.

The example uses tensorflow and scipy modules.

The example code is contained in `test.py`.

```bash
$ python test.py
```

A sample output is as follows:
```
Succesfully created a neural network
Input file loaded
Output file loaded
Training
Step 0 : accuracy: 0.0
Step 100 : accuracy: 0.019999999552965164
Step 200 : accuracy: 0.949999988079071
Step 300 : accuracy: 0.9799999594688416
Step 400 : accuracy: 0.019999999552965164
Step 500 : accuracy: 0.029999999329447746
Step 600 : accuracy: 0.9999999403953552
Step 700 : accuracy: 0.9999999403953552
Step 800 : accuracy: 0.9999999403953552
Step 900 : accuracy: 0.9999999403953552
Step 1000 : accuracy: 0.9999999403953552
...
```
