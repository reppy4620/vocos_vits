# Vocos-VITS

This is almost VITS but decoder is replaced with Vocos for performance.  

# Usage
Running run.sh will automatically download the data and begin training.

```sh
cd scripts
./run.sh
```

synthesize.sh uses last.ckpt by default, so if you want to use a specific weight, change it.

```sh
cd scripts
./synthesis.sh
```

# Requirements
```sh
pip install torch torchaudio lightning tqdm pandas matplotlib
```

# Result
WIP
