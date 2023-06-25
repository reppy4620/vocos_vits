import numpy as np
import torch
import torchaudio

from pathlib import Path

from audio import SpectrogramTransform
from text import text_to_sequence, phonemes


class TextAudioDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, wav_dir, lab_dir, audio_config, split='|'):
        with open(file_path) as f:
            lines = f.readlines()
            data = list()
            for line in lines:
                bname, label = line.strip().split(split)
                data.append((bname, label))
        self.data = data
        self.wav_dir = Path(wav_dir)
        self.lab_dir = Path(lab_dir)
        
        self.audio_config = audio_config
        self.hop_length = audio_config.hop_length
        self.sample_rate = audio_config.sample_rate

        self.spec_tfm = SpectrogramTransform(**audio_config)

    def get_duration(self, filepath, label, mel_length):
        with open(filepath, 'r') as f:
            labels = f.readlines()
        durations = []
        cnt = 0
        for s in label.split():
            if s in phonemes or s in ['^', '$', '?', '_']:
                s, e, _ = labels[cnt].split()
                s, e = int(s), int(e)
                dur = (e - s) * 1e-7 / (self.hop_length / self.sample_rate)
                durations.append(dur)
                cnt += 1
            else:
                durations.append(1)
        # adjust length, differences are caused by round op.
        round_durations = np.round(durations)
        diff_length = np.sum(round_durations) - mel_length
        if diff_length == 0:
            return torch.FloatTensor(round_durations)
        elif diff_length > 0:
            durations_diff = round_durations - durations
            d = -1
        else: # diff_length < 0
            durations_diff = durations - round_durations
            d = 1
        sort_dur_idx = np.argsort(durations_diff)[::-1]
        for i, idx in enumerate(sort_dur_idx, start=1):
            round_durations[idx] += d
            if i == abs(diff_length):
                break
        assert np.sum(round_durations) == mel_length
        return torch.FloatTensor(round_durations)


    def __getitem__(self, idx):
        bname, label = self.data[idx]

        phonemes = torch.LongTensor(text_to_sequence(label.split()))

        wav, _ = torchaudio.load(self.wav_dir / f'{bname}.wav')
        spec = self.spec_tfm.to_spec(wav).squeeze()
        spec_length = spec.size(-1)

        duration = self.get_duration(self.lab_dir / f'{bname}.lab', label, spec_length)

        return (
            bname,
            phonemes,
            duration,
            wav,
            spec
        )

    def __len__(self):
        return len(self.data)


def collate_fn(batch):
    (
        bnames,
        phonemes,
        durations,
        wavs,
        specs
    ) = tuple(zip(*batch))

    B = len(bnames)
    x_lengths = [len(x) for x in phonemes]
    frame_lengths = [x.size(-1) for x in specs]
    sample_lengths = [x.size(-1) for x in wavs]

    x_max_length = max(x_lengths)
    frame_max_length = max(frame_lengths)
    sample_max_length = max(sample_lengths)
    spec_dim = specs[0].size(0)

    x_pad = torch.zeros(size=(B, x_max_length), dtype=torch.long)
    dur_pad = torch.zeros(size=(B, 1, x_max_length), dtype=torch.float)
    spec_pad = torch.zeros(size=(B, spec_dim, frame_max_length), dtype=torch.float)
    wav_pad = torch.zeros(size=(B, 1, sample_max_length), dtype=torch.float)
    for i in range(B):
        x_l, f_l, s_l = x_lengths[i], frame_lengths[i], sample_lengths[i]
        x_pad[i, :x_l] = phonemes[i]
        dur_pad[i, :, :x_l] = durations[i]
        spec_pad[i, :, :f_l] = specs[i]
        wav_pad[i, :, :s_l] = wavs[i]

    x_lengths = torch.LongTensor(x_lengths)
    frame_lengths = torch.LongTensor(frame_lengths)
    sample_lengths = torch.LongTensor(sample_lengths)

    return (
        bnames,
        x_pad,
        dur_pad,
        x_lengths,
        wav_pad,
        sample_lengths,
        spec_pad,
        frame_lengths,
    )
