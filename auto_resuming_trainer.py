import errno
import os
import subprocess
import sys

from callbacks import TrainingConfigWriter


def main():
    for _ in range(100):
        subprocess.run(['python', 'train.py',
                        '-ta', '/home/tom/datasets/rnn-dummy/train.txt',
                        '-va', '/home/tom/datasets/rnn-dummy/val.txt',
                        '-t', '/home/tom/datasets/rnn-dummy/table.txt',
                        '-ar',
                        '-w', '750',
                        '-b', '128',
                        '-e', '100',
                        '-lr', '0.0002'
                        ], stdout=sys.stdout, bufsize=1)

        # If it finished training successfully then the config file will have been removed
        if not os.path.exists(TrainingConfigWriter.get_config_path()):
            break


if __name__ == '__main__':
    main()
