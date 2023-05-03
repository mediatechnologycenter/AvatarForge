import os
from argparse import ArgumentParser
from tqdm import tqdm

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--folder", type=str, help='Path to folder to rename.')

    args = parser.parse_args()

    path = args.folder
    file_list = os.listdir(path)
    num_files = len(file_list)
    print(f'Total number of files: {num_files}')

    for i, file_name in enumerate(tqdm(file_list)):
        name, ext = os.path.splitext(file_name)
        new_name = os.path.join(path, name, file_name)
        os.makedirs(os.path.join(path, name), exist_ok=True)
        os.rename(os.path.join(path, file_name), new_name)
