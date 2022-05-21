#!/bin/bash

list="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20"
data_list="1 2 3 4 5 6 7 8 9 10"
model_list="dkt dkvmn sakt"

for j in ${data_list}
do
    echo "---new data ${j}---" >> ../records/coldstart1_record.txt

    for model in ${model_list}
    do

        for i in ${list}
        do
            echo "dataset_name coldstart1_assist2015; model_name ${model}; cold1_stu_num ${i} " >> ../records/coldstart1_record.txt
            
            python \
            train.py \
            --model_fn coldstart1.pth \
            --dataset_name coldstart1_assist2015 \
            --n_epochs 100 \
            --model_name ${model} \
            --record_path ../records/coldstart1_record.txt \
            --cold1_stu_num ${i}

        done

    done

done
