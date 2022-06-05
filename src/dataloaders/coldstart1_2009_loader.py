import numpy as np
import pandas as pd

from torch.utils.data import Dataset


DATASET_DIR = "../datasets/assistment2009_2010/skill_builder_data.csv"

#여기서 해야하는 것은 몇 명인지만을 추리면 됨
class COLDSTART1(Dataset):
    def __init__(self, stu_num=None, random_idx=None, dataset_dir=DATASET_DIR) -> None:
        super().__init__()

        #이것으로 학생의 수를 조절하기
        self.stu_num = stu_num
        self.random_idx = random_idx

        self.dataset_dir = dataset_dir
        
        self.q_seqs, self.r_seqs, self.q_list, self.u_list, self.q2idx, \
            self.u2idx = self.preprocess() #가장 아래에서 각각의 요소를 가져옴

        self.num_u = self.u_list.shape[0]
        self.num_q = self.q_list.shape[0]

        self.len = len(self.q_seqs)

    def __getitem__(self, index):
        #출력되는 벡터는 모두 101개로 전처리되어있고, 만약 빈칸이 있는 데이터의 경우에는 -1로 채워져있음
        return self.q_seqs[index], self.r_seqs[index]

    def __len__(self):
        return self.len

    def preprocess(self):
        df = pd.read_csv(self.dataset_dir, encoding="ISO-8859-1")
        df = df[(df["correct"] == 0).values + (df["correct"] == 1).values]

        #여기에서 user들 중에서 self.random_idx(인덱스)에 해당하는 user 추출
        u_list = np.unique(df["user_id"].values) #중복되지 않은 user의 목록
        q_list = np.unique(df["problem_id"].values) #중복되지 않은 question의 목록

        #stu_num이 있다면, 추출
        if self.stu_num != None:
            u_list = u_list[self.random_idx]

        u2idx = {u: idx for idx, u in enumerate(u_list)} #중복되지 않은 user에게 idx를 붙여준 딕셔너리
        q2idx = {q: idx for idx, q in enumerate(q_list)} #중복되지 않은 question에 idx를 붙여준 딕셔너리

        q_seqs = [] #로그 기준으로 각 user별 질문 목록을 담은 리스트
        r_seqs = [] #로그 기준으로 각 user별 정답 목록을 담은 리스트

        for u in u_list:
            df_u = df[df["user_id"] == u]

            q_seq = np.array([q2idx[q] for q in df_u["problem_id"].values])
            r_seq = df_u["correct"].values

            q_seqs.append(q_seq)
            r_seqs.append(r_seq)

        return q_seqs, r_seqs, q_list, u_list, q2idx, u2idx