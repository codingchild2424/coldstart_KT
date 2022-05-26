#!/bin/bash

#python train.py --model_fn model.pth --dataset_name coldstart1  --five_fold True --stu_num 100


# #처음 2000까지는 100단위
# #2000이후는 500단위
# list="100 200 300 400 500 600 700 800 900 1000 \
#  1100 1200 1300 1400 1500 1600 1700 1800 1900 2000 \
#  2500 3000 3500 4000 4500 5000 5500 6000 6500 7000 \
#  7500 8000 8500 9000 9500 10000 10500 11000 11500 12000 \
#  12500 13000 13500 14000 14500 15000 15500 16000 16500 17000 \
#  17500 18000 18500 19000 19500 \
#  19917"
# total_list="1 2 3 4 5"
# model_list="dkt dkvmn sakt"

# #처음에 데이터셋 생성
# for stu in ${list}
# do
#     python ../preprocessors/preprocessor_2015.py --stu_num ${stu}
# done

# for j in ${total_list}
# do
#     for model in ${model_list}
#     do

#         for stu_num in ${list}
#         do
#             echo -n "data reset${j}: dataset_name coldstart1_assist2015; model_name ${model}; cold1_stu_num ${stu_num}" >> ../records/coldstart1_record.tsv
            
#             python \
#             train.py \
#             --model_fn coldstart1.pth \
#             --dataset_name coldstart1_assist2015 \
#             --n_epochs 100 \
#             --model_name ${model} \
#             --record_path ../records/coldstart1_record.tsv \
#             --cold1_stu_num ${stu_num} \
#             --five_fold ${j}

#         done

#     done

#     #데이터 갱신
#     for stu in ${list}
#         do
#             python ../preprocessors/preprocessor_2015.py --stu_num ${stu}
#         done

# done
