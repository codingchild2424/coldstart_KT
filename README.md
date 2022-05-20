# Introduce
This repository was made for estimating cold start problems in KT models.
DKT, DKVMN, SAKT, SAINT, GKT are targets for research.

# todo_list
1. add ASSISTment 2009-2010 datasets + preprocessing
2. Visualizer have to be changed
3. made SAKT, SAINT, GKT

# Coldstart Prob 1 - 학생 수가 적을 때(실험 중)
0. 2009, 2015 둘 다 사용
1. 먼저, preprocessor_2015를 통해, 1명부터 20명까지의 csv를 만듦
    - 단, 모든 모델에 대한 1번의 실험이 끝나기 전까지는 데이터를 새로 만들지 않음
    - 모든 모델에 대한 실험이 끝나면 다시 생성
2. argument를 통해 cold1_stu_num 값을 전달함
    - 해당 값은 1부터 20까지 넣을 수 있음
    - 각 값을 넣어가면서 실험 결과를 확인
3. epochs은 100까지 실험
4. 실험방법
    - 먼저 주피터 노트북으로 데이터셋 생성
    - 각 모델에 num1부터 num20까지 데이터셋을 넣으며 성능 측정
    - 모든 모델의 성능 측정이 끝났다면, 다시 주피터 노트북으로 데이터셋 갱신
    - 이후 반복
4. 성능 기록
    - https://docs.google.com/spreadsheets/d/1XMMQEjAPiotXWdfOVNNF5DLAfqi1DCz-5DEcLvK61HY/edit?usp=sharing

### 실행 명령어 예시
python train.py --model_fn coldstart1.pth --dataset_name coldstart1_assist2015 --n_epochs 100 --model_name dkt --cold1_stu_num 1

### 실행 명령을 위한 shell(작성 중)
./coldstart1.sh


# Coldstart Prob 2 - 문제 수가 적을 때(수정 중)
0. 2009, 2015 둘 다 사용
1. 5 fold cross validation 사용해서 8:2(train:test)로 데이터셋 분류

# Coldstart Prob 3 - 정선된 데이터를 활용하여, 각 개념에 대한 문항에 대해 학습
0. 전처리 된 2009만 사용
    - 4개의 유형에 대한 문항 샘플링
    - 각 유형별 1~8회를 푼 문항만 추출해서 정리
1. 
2. 4개의 유형의 문제 추출
3. 각 유형별 1~8회를 푼 문항만 추출해서 정리
4. 각 유형별 1~8에 대한 성능 추출

### Check the GPU status
watch -n .5 nvidia-smi

# Reference
hcnoh's github
...