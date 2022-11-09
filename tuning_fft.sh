#!/bin/sh
source ./lock_utils.sh

CHECKPOINT_BASE="/ComplexHyperbolicKGE/checkpoints"
EPOCHS=500

Embed_Args=(--multi_c \
    --max_epochs $EPOCHS \
    --patience 30 \
    --valid 5 \
    --init_size 0.001 \
    --gamma 0.0 \
    --bias learn \
    )

echo $Embed_Args

set -e -hxiaoaf

excute_list_args(){
    gpu=$1; shift;
    dataset=$1; shift;
    model=$1; shift;
    regularizer=$1; shift;
    reg=$1; shift;
    optimizer=$1; shift;
    rank=$1; shift;
    batch_size=$1; shift;
    neg_sample_size=$1; shift;
    lr=$1; shift;
    double_neg=$1; shift;


    dtype=double

    DATE_TIME=$(date +%Y%m%d_%H%M%S)
    checkpoint_dir=$CHECKPOINT_BASE/"$dataset"_"$model"_"$rank"_"$lr"_"$DATE_TIME"
    checkpoint_dir=$(make_sure_dir $checkpoint_dir)
    checkpoint_dir=$(realpath $checkpoint_dir)
    log_file="$checkpoint_dir/embed.log"
    touch "$log_file"
    echo "\$Parameters: $model $regularizer $reg $optimizer $rank $batch_size $neg_sample_size $lr $double_neg" >> "$log_file"
    local_embed_args=(--gpu $gpu --dataset $dataset --model $model --regularizer $regularizer \
    --reg $reg --optimizer $optimizer --rank $rank --batch_size $batch_size --neg_sample_size $neg_sample_size \
    --learning_rate $lr --save_dir $checkpoint_dir --double_neg $double_neg --dtype $dtype \
    ${Embed_Args[@]})

    python run.py ${local_embed_args[@]} | tee -a "$log_file"
}

excute_list_args $@