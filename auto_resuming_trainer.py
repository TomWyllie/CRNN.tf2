import os
import subprocess
import sys

from callbacks import TrainingConfigWriter


def main():
    # If it fails 100 times something is really badly wrong.
    for _ in range(100):
        subprocess.run(['python', 'train.py',
                        # '--dir', '/home/tom/datasets/folkfriend/',
                        '--dir', '/home/tom/datasets/rnn-dummy/',
                        '-ar',
                        '-w', '749',
                        '-b', '128',
                        '-e', '500',
                        # Recommend starting with 0.002 and progressing down to
                        #   0.0002 and then 0.00002, adjusting by hand once loss
                        #   has flattened out (and restarting training).
                        '-lr', '0.002'
                        # '-lr', '0.0001'
                        # '-lr', '0.00002'
                        ], stdout=sys.stdout, bufsize=1)

        # If it finished training successfully then the config file will have been removed
        if not os.path.exists(TrainingConfigWriter.get_config_path()):
            break


if __name__ == '__main__':
    main()
