import numpy as np
import pandas as pd

from torch.utils.data import Dataset


DATASET_DIR = "../datasets/assistment2009_2010/skill_builder_data.csv"

#여기서 해야하는 것은 몇 명인지만을 추리면 됨
class COLDSTART2(Dataset):
    def __init__(self, skill_num=None, opportunity=None, dataset_dir=DATASET_DIR) -> None:
        super().__init__()

        #이것으로 학생의 수를 조절하기
        self.skill_num = skill_num
        self.opportunity = opportunity

        self.dataset_dir = dataset_dir
        
        self.q_seqs, self.r_seqs, self.q_list, self.u_list, self.q2idx, \
            self.u2idx = self.preprocess(self.skill_num, self.opportunity) #가장 아래에서 각각의 요소를 가져옴

        self.num_u = self.u_list.shape[0]
        self.num_q = self.q_list.shape[0]

        self.len = len(self.q_seqs)

    def __getitem__(self, index):
        #출력되는 벡터는 모두 101개로 전처리되어있고, 만약 빈칸이 있는 데이터의 경우에는 -1로 채워져있음
        return self.q_seqs[index], self.r_seqs[index]

    def __len__(self):
        return self.len

    def preprocess(self, skill_num, opportunity):

        df = pd.read_csv(self.dataset_dir, encoding="ISO-8859-1")
        df = df[(df["correct"] == 0).values + (df["correct"] == 1).values]
        df = df[['user_id', 'problem_id', 'correct', 'skill_name', 'opportunity']]

        skill_names = [
            "Addition and Subtraction Fractions",
            "Addition and Subtraction Integers",
            "Conversion of Fraction Decimals Percents",
            "Equation Solving Two or Fewer Steps"
            ]

        skill_name = skill_names[skill_num]

        #skill name과 opportunity에 따라서 데이터 추출하기
        df = df[
            (df['skill_name'] == skill_name)
            & (df['opportunity'] <= opportunity)
            ]

        u_list = np.unique(df["user_id"].values) #중복되지 않은 user의 목록
        q_list = np.unique(df["problem_id"].values) #중복되지 않은 question의 목록

        u2idx = {u: idx for idx, u in enumerate(u_list)} #중복되지 않은 user에게 idx를 붙여준 딕셔너리
        q2idx = {q: idx for idx, q in enumerate(q_list)} #중복되지 않은 question에 idx를 붙여준 딕셔너리

        q_seqs = [] #로그 기준으로 각 user별 질문 목록을 담은 리스트
        r_seqs = [] #로그 기준으로 각 user별 정답 목록을 담은 리스트

        for u in u_list:
            df_u = df[df["user_id"] == u] #log_id 정보가 없어서 sort하지는 못함

            q_seq = np.array([q2idx[q] for q in df_u["problem_id"].values])
            r_seq = df_u["correct"].values

            q_seqs.append(q_seq)
            r_seqs.append(r_seq)

        return q_seqs, r_seqs, q_list, u_list, q2idx, u2idx