import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import random

#2015 skill data 불러오기
data_path = "../datasets/2015_100_skill_builders_main_problems.csv"

pd_data = pd.read_csv(data_path, encoding="ISO-8859-1")

#랜덤으로 섞기
pd_data = shuffle(pd_data)

#일단 전체 데이터를 랜덤으로 8:2로 분리하기
train_dataset, test_dataset = train_test_split(pd_data, test_size = 0.2)

#test_set부터 먼저 설정하기
test_path = "../datasets/coldstart_datasets/coldstart1_2015_testdatasets.csv"
test_dataset.to_csv(test_path)

train_dataset['user_id'].unique()
train_dataset['user_id'].value_counts().values

#20명 이상 30명 아하의 학습자들의 데이터를 train data로 만들기

train_user = train_dataset['user_id'].value_counts().index.to_list()
train_values = train_dataset['user_id'].value_counts().values.tolist()

index_up20_un30 = []

for idx, user in enumerate(train_user):
    if train_values[idx] >= 20 | train_values[idx] <=30:
        index_up20_un30.append(user)

train_dataset = train_dataset[
    train_dataset['user_id'].isin(index_up20_un30)
    ]

#여기서 random으로 인덱스를 랜덤으로 20개를 정하기
random_index = random.sample(index_up20_un30, 20)

coldstart1_train_folder_path = "../datasets/coldstart_datasets/coldstart1_2015_traindatasets"

#한명의 데이터만 가져와서 csv로 저장
num1_path = coldstart1_train_folder_path + "/coldstart1_num1.csv"

num1_list = [random_index[0]]
coldstart1_train_num1 = train_dataset[train_dataset['user_id'].isin(num1_list)]

coldstart1_train_num1.to_csv(num1_path)

for i in range(len(random_index)):
    num_path = coldstart1_train_folder_path + "/coldstart1_num" + str(i) + ".csv"
    num_list = []

    for j in range(i+1):
        num_list.append(random_index[j])

    coldstart_train_num = train_dataset[train_dataset['user_id'].isin(num_list)]

    coldstart_train_num.to_csv(num_path)