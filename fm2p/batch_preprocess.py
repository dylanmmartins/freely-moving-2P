import argparse
from tqdm import tqdm
import fm2p


def read_file_to_list(filepath):
    lines = []
    try:
        with open(filepath, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
    return lines

def batch_preprocess():

    parser = argparse.ArgumentParser()
    parser.add_argument('-txt', '--txt', type=str, default=None)
    args = parser.parse_args()

    if args.txt is not None:
        txt_path = args.txt

    else:
        txt_path = fm2p.select_file(
            title='Choose .txt file that lists all config file paths.',
            filetypes=[('txt', '*.txt'),]
        )

    cfg_file_list = read_file_to_list(txt_path)
    # to remove newline characters
    cfg_file_list = [line.strip() for line in cfg_file_list]

    print('  -> Found {} config files.'.format(len(cfg_file_list)))

    for i, cfg_path in tqdm(enumerate(cfg_file_list)):
        print('  -> Analyzing {}'.format(cfg_path))
        # fm2p.preprocess(cfg_path=cfg_path)


if __name__ == '__main__':

    batch_preprocess()

