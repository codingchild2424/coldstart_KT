#!/bin/bash

opportunities="3 4 5 6 7 8"
model_name="dkt dkvmn sakt"
skill_nums="0 1 2 3"

for model in ${model_name}
do
    for opportunity in ${opportunities}
    do
        for skill_num in ${skill_nums}
        do
            echo -n "model_name ${model}; skill_num ${skill_num}; opportunity ${opportunity}" \
             >> ../records/coldstart2_record.tsv

            python \
            train.py \
            --model_fn model.pth \
            --dataset_name coldstart2 \
            --model_name ${model} \
            --five_fold True \
            --skill_num ${skill_num} \
            --opportunity ${opportunity} \
            --record_path ../records/coldstart2_record.tsv \
            --n_epochs 200
        done
    done
done