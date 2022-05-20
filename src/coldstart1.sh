#!/bin/bash

list="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20"

for i in ${list}
do
    python \
    train.py \
    --model_fn coldstart1.pth \
    --dataset_name coldstart1_assist2015 \
    --n_epochs 100 \
    --model_name dkt \
    --cold1_stu_num ${i}
done

