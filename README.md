# Introduce
This repository was made for estimating cold start problems in KT models.
DKT, DKVMN, SAKT, SAINT, GKT are targets for research.

# todo_list
1. add ASSISTment 2009-2010 datasets + preprocessing
2. Visualizer have to be changed
3. made SAKT, SAINT, GKT

# Coldstart Prob 1 - 학생 수가 적을 때(실험 중)
0. 2015만 사용함, 2009는 전처리 후 사용
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
    - 이후 반복(총 10회)
4. 최종 성능 기록
    - records에 txt로 1차 기록
    - 구글 시트로 옮기기
    - https://docs.google.com/spreadsheets/d/1XMMQEjAPiotXWdfOVNNF5DLAfqi1DCz-5DEcLvK61HY/edit?usp=sharing
    - 시각화 프로그램으로 그래프 깔끔하게 시각화하기

### 실행 명령(shell file 활용)
./coldstart1.sh

# Coldstart Prob 2 - 문제 수가 적을 때(코드 작업 중)
0. 2015만 사용함, 2009는 전처리 후 사용
1. 주피터 파일로 데이터에서 100문항 이상 해결한 학생의 결과 확인해보기
    - 100문항은 없고, 최대 45문항을 해결한 학생이 있음
2. 40개 이상의 문항을 해결한 학생의 데이터 활용(24명)
    - 그래야 실제 정답값과 예측값을 비교하여 성능을 측정할 수 있기 때문
3. 모델에 학생 1명의 시퀀스 데이터를 넣고, (RNN이라면) 각 문항에서 나온 전체 예측값을 가지고, 실제 결과와 비교해보기
    - 각 문항에서 최종적으로 나온 결과 값은 모든 문항에 대한 확률값을 가지고 있음
    - 따라서 해당 학생의 실제 결과와 비교하였을 때, 성능을 측정할 수 있음
    - 이때 확률값이 0.5를 넘으면 정답으로 예측한 것으로 보고, 아니라면 오답으로 예측한 것으로 추정하기

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