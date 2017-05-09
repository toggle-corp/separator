import os
import argparse

import numpy as np

from nn.separator_network import SeparatorNetwork
from sound.train_data import TrainData
from sound.run_data import RunData


class Separator:

    def train(self, files, batches, samples, iterations, name):

        nn = SeparatorNetwork(batches, samples)
        print("Completed creating a neural network")

        if name:
            nn.load(name)

        files = [(files[i], files[i+1])
                 for i in range(0, len(files), 2)]

        train_data = TrainData(samples)
        for f in files:
            print("Training files loading from {} and {}"
                  .format(f[0], f[1]))
            train_data.add_test_files(*f)

        print("Completed loading training data")
        print("Started training")
        print("Number of samples per batch: {}".format(samples))
        print("Number of batches per iteration: {}".format(batches))
        print("Number of iterations: {}".format(iterations))

        for i in range(iterations):
            xs, ys = train_data.get_random(batches)
            nn.train(xs, ys)

            # Every 100th step, print out error of neural network
            # based on the current batch of test data
            if i % 100 == 0 or i == iterations - 1:
                print("Step {} : error: {}".format(i+1, nn.evaluate(xs, ys)))

        if name:
            nn.save(name)

    def run(self, input_file, batches, samples, name):
        nn = SeparatorNetwork(batches, samples)
        print("Completed creating a neural network")

        if name:
            nn.load(name)

        run_data = RunData(samples, input_file)
        print("Completed loading sound data")

        print("Started neural network feeding")
        print("Number of samples per batch: {}".format(samples))
        print("Number of batches per iteration: {}".format(batches))

        passes = run_data.get_num(batches)
        print("Total number of passes required: {}".format(passes))

        for i in range(passes):
            run_data.collect_output(nn.run(run_data.get_next(batches)))

            if i % 10 == 0 or i == passes-1:
                print("Step {} completed".format(i+1))

        print("Done")

        print("Saving")
        run_data.save_sound('out.wav')
        print("Done")


def main():
    # Tensorflow log levels
    # 0 : show all
    # 1 : hide info messages
    # 2 : hide warning and info messages
    # 4 : hide errors, warning and info messages
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="commands", dest='command')

    # train command
    train_parser = subparsers.add_parser(
        'train', help="Train neural network")

    train_parser.add_argument(
        'train_files', nargs='+')

    train_parser.add_argument(
        '-n', '--name')

    train_parser.add_argument(
        '-s', '--samples', type=int, default=100)

    train_parser.add_argument(
        '-b', '--batches', type=int, default=100)

    train_parser.add_argument(
        '-i', '--iterations', type=int, default=500)

    # run command
    run_parser = subparsers.add_parser(
        'run', help="Run neural network")
    run_parser.add_argument('input_file')

    run_parser.add_argument(
        '-n', '--name')

    run_parser.add_argument(
        '-s', '--samples', type=int, default=100)

    run_parser.add_argument(
        '-b', '--batches', type=int, default=100)

    args = parser.parse_args()
    if args.command == 'train':
        if len(args.train_files) % 2 != 0:
            print("Train files must be pair of input output files")
            print("Please provide an even number files")
            return

        Separator().train(args.train_files, args.batches,
                          args.samples, args.iterations, args.name)

    elif args.command == 'run':
        Separator().run(args.input_file, args.batches, args.samples,
                        args.name)


if __name__ == '__main__':
    main()
