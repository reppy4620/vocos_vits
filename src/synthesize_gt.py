import torch
import torchaudio

from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
from lightning import seed_everything

import config as cfg
from model import VITSModule


@torch.no_grad()
def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wav_dir = Path(args.wav_dir)

    seed_everything(cfg.train.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = VITSModule.load_from_checkpoint(args.ckpt_path, params=cfg).to(device)

    with open(args.label_file, mode="r") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]

    for line in tqdm(lines, total=len(lines)):
        bname, _ = line.split("|")
        wav, _ = torchaudio.load(wav_dir / f"{bname}.wav")
        spec = module.spec_tfm.to_spec(wav.to(device))
        wav = module.infer_gt(spec)
        torchaudio.save(
            filepath=out_dir / f"{bname}.wav", src=wav, sample_rate=cfg.sample_rate
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--label_file", type=str, required=True)
    parser.add_argument("--wav_dir", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()
    main(args)
