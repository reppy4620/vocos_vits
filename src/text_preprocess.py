from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser

from text import pp_symbols


def main(args):
    lab_dir = Path(args.lab_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lab_files = list(sorted(lab_dir.glob('*.lab')))
    text_list = list()
    for lab_file in tqdm(lab_files, total=len(lab_files)):
        bname = lab_file.stem
        with open(lab_file, 'r') as f:
            fullcontext = f.readlines()        
        label = pp_symbols(fullcontext)
        s = f'{bname}|{" ".join(label)}'
        text_list.append(s)
    with open(out_dir / 'all.txt', 'w') as f:
        for s in text_list:
            f.write(f'{s}\n')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--lab_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    args = parser.parse_args()
    main(args)
