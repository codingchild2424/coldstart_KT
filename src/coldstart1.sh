#!/bin/bash

list="100 200 300 400 500 600 700 800 900 1000 \
  1100 1200 1300 1400 1500 1600 1700 1800 1900 2000 \
  2500 3000 3500 4000 4500 5000 5500 6000 6500 7000 \
  7500 8000 8500 9000 9500 10000 10500 11000 11500 12000 \
  12500 13000 13500 14000 14500 15000 15500 16000 16500 17000 \
  17500 18000 18500 19000 19500 \
  19917"
model_name="dkt dkvmn sakt"

for model in ${model_name}
do
    for stu_num in ${list}
    do
        echo -n "model_name ${model}; stu_num ${stu_num}" >> ../records/coldstart1_record.tsv

        python \
        train.py \
        --model_fn model.pth \
        --dataset_name coldstart1 \
        --model_name ${model} \
        --five_fold True \
        --stu_num ${stu_num} \
        --record_path ../records/coldstart1_record.tsv \
        --n_epochs 100
    done
done