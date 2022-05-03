import os

import numpy as np
import torch

from torch.nn import Module, Parameter, Embedding, Linear
from torch.nn.init import kaimiing_normal_
from torch.nn.functional import binary_cross_entropy
from sklearn import metrics

class DKVMN(Module):
    '''
        Args:
            num_q: the total number of the questions(KCs) in the given dataset
            dim_s: the dimension of the state vectors in this model
            size_m: the memory size of this model
    '''
    def __init__(self, num_q, dim_s, size_m):
        super().__init__()
        self.num_q = num_q
        self.dim_s = dim_s
        self.size_m = size_m

        self.k_emb_layer = Embedding(self.num_q, self.dim_s) #여기는 q값만 들어오므로, embedding vector의 갯수가 문항수와 동일함
        
        #Mk: Key Matrix로 N개의 Latent Concept(C1, ..., CN)을 인코딩하는 행렬
        #모델의 추론 과정에서 시간과 관계없이 변하지 않는 Static한 메모리
        self.Mk = Parameter(
            torch.Tensor(self.size_m, self.dim_s)
        )

        #Mvt의 초깃값으로 추측됨
        #해당값은 Value Matrix로 (dv, N)의 크기를 가지고 있고
        #N개의 Latent Concept(C1, ..., CN) 각각에 대한 사용자의 Concept State(s1, ..., sN)을 인코딩하고 있음
        #dynamic하게 변화하는 값
        self.Mv0 = Parameter(
            torch.Tensor(self.size_m, self.dim_s)
        )

        kaimiing_normal_(self.Mk)
        kaimiing_normal_(self.Mv0)

        self.v_emb_layer = Embedding(self.num_q * 2, self.dim_s) #여기는 (q, r)이 동시에 들어오므로, embedding vector가 두배

        #read 과정에서 사용
        self.f_layer = Linear(self.dim_s * 2, self.dim_s)
        self.p_layer = Linear(self.dim_s, 1)

        #write 과정에서 사용
        self.e_layer = Linear(self.dim_s, self.dim_s)
        self.a_layer = Linear(self.dim_s, self.dim_s)

    def forward(self, q, r):
        
        # 문항과 문제 번호에 따라서 값을 줌
        x = q + self.num_q * r

        batch_size = x.shape[0]
        Mvt = self.Mv0.unsqueeze(0).repeat(batch_size, 1, 1)

        Mv = [Mvt]

        k = self.k_emb_layer(q)
        v = self.v_emb_layer(x)

        w = torch.softmax(
            torch.matmul(k, self.Mk.T),
            dim = -1
        )

        #Write Process
        e = torch.sigmoid( self.e_layer(v) )
        a = torch.tanh( self.a_layer(v) )

        for et, at, wt in zip(
            e.permute(1, 0, 2), a.permute(1, 0, 2), w.permute(1, 0, 2)
        ):
            Mvt = Mvt * (1 - (wt.unsqueeze(-1) * et.unsqueeze(1)) ) + ( wt.unsqueeze(-1) * at.unsqueeze(1) )
            Mv.append(Mvt)

        Mv = torch.stack(Mv, dim = 1)

        #Read Process
        f = torch.tanh(
            self.f_layer(
                torch.cat(
                    [
                        (w.unsqueeze(-1) * Mv[:, :-1]).sum(-2),
                        k
                    ],
                    dim = -1
                )
            )
        )

        p = torch.sigmoid( self.p_layer(f) ).squeeze()

        return p, Mv