#!/bin/bash
####
# Run linear evaluation
####

dataset="imagenet100"
cuda_device=$1
first_char=${cuda_device:0:1}
port=$((10101 + first_char))

# use sqrt scaling for epochs
percentages=(1 0.1 0.01 0.001)
batchSizes=(1024 1024 1024 1024)
n_epochs=(90 285 900 900)

length=${#percentages[@]}
for (( i=0; i<$length; i++ )); do
    sample_images=${percentages[$i]}
    epochs=${n_epochs[$i]}
    batch=${batchSizes[$i]}

    echo "Running on GPU $cuda_device with sample_images $sample_images"

    # python -m debugpy --listen 5678 --wait-for-client lincls.py \
    CUDA_VISIBLE_DEVICES=$cuda_device python lincls.py \
        --dist-url "tcp://localhost:${port}" \
        -b $batch --lr 0.1 --epochs $epochs \
        --multiprocessing-distributed --world-size 1 --rank 0 --workers 6 \
        --data_name imagenet100 \
        --data $dataset \
        --save_dir lincls_saved \
        --log_dir $2 \
        --pretrained $4/$2/checkpoint_0199.pth.tar \
        --seed $3 \
        --sample_images $sample_images \
        || { echo "Failed training"; exit 1; }
done
