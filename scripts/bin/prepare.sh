#!/bin/bash -eu

download_dir=../downloads
data_dir=../data
mkdir -p $download_dir $data_dir

if [ ! -d $download_dir/jsut_ver1.1 ]; then
    echo "Download JSUT corpus"
    cd $download_dir
    curl -LO http://ss-takashi.sakura.ne.jp/corpus/jsut_ver1.1.zip
    unzip -o jsut_ver1.1.zip
    cd -
fi
if [ ! -d $download_dir/jsut-label ]; then
    echo "Download JSUT label - fullcontext"
    cd $download_dir
    curl -LO https://github.com/sarulab-speech/jsut-label/archive/v0.0.2.zip
    unzip -o v0.0.2.zip
    ln -s jsut-label-0.0.2 jsut-label
    cd -
fi

if [ ! -d $data_dir/wav24k ]; then
    echo "Resample BASIC5000"
    mkdir $data_dir/wav24k
    for wav_path in $download_dir/jsut_ver1.1/basic5000/wav/*.wav;
    do
        fname=$(basename $wav_path)
        echo $fname
        sox $wav_path -r 24000 $data_dir/wav24k/$fname
    done
fi

if [ ! -d $data_dir/label ]; then
    echo "Preprocess label"
    label_dir=$data_dir/label
    python ../src/text_preprocess.py \
        --lab_dir $download_dir/jsut-label/labels/basic5000 \
        --out_dir $label_dir
    
    data_path=$label_dir/all.txt
    all_length=$(cat $data_path | wc -l)
    valid_length=$(echo $all_length | awk '{print int($1*0.02)}')
    train_length=$(echo $all_length $valid_length | awk '{print $1-$2}')

    echo "train valid all" > $label_dir/num.txt
    echo $train_length $valid_length $all_length >> $label_dir/num.txt
    cat $label_dir/num.txt

    head -n $valid_length $data_path > $label_dir/valid.txt
    tail -n $train_length $data_path > $label_dir/train.txt
fi
