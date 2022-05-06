import torch
from torch.nn import Module, Parameter, Embedding, \
    Sequential, Linear, ReLU, MultiheadAttention, LayerNorm, Dropout
from torch.nn.init import kaiming_normal_

class SAKT(Module):
    def __init__(self, num_q, n, d, num_attn_heads, dropout=.2):
        super().__init__()
        self.num_q = num_q #문항의 갯수
        self.n = n #length of the sequence of questions and responses
        self.d = d #dimension of the hidden vectors in this model
        self.num_attn_heads = num_attn_heads #the number of the attention heads in the multi-head attention
        self.dropout = dropout

        self.M = Embedding(self.num_q * 2, self.d)
        self.E = Embedding(self.num_q, d)
        self.P = Parameter(torch.Tensor(self.n, self.d))

        kaiming_normal_(self.P)

        self.attn = MultiheadAttention(
            self.d, self.num_attn_heads, dropout = self.dropout
        )
        self.attn_dropout = Dropout(self.dropout)
        self.attn_layer_norm = LayerNorm(self.d)

        self.FFN = Sequential(
            Linear(self.d, self.d),
            ReLU(),
            Dropout(self.dropout),
            Linear(self.d, self.d),
            Dropout(self.dropout),
        )

        self.FFN_layer_norm = LayerNorm(self.d)

        self.pred = Linear(self.d, 1)

    def forward(self, q, r, qry):

        #|q| = (bs, sq)
        #|r| = (bs, sq)
        #|qry| = (bs, sq)

        x = q + self.num_q * r

        M = self.M(x).permute(1, 0, 2) #|M| = (sq, bs, d)
        E = self.E(qry).permute(1, 0, 2) #|E| = (sq, bs, d)
        P = self.P.unsqueeze(1) #|P| = (n, 1, d)

        causal_mask = torch.triu(
            torch.ones([ E.shape[0], M.shape[0] ]), diagonal = 1
        ).bool()

        #여기서 형상이 안맞는 오류 발생
        M = M + P

        S, attn_weights = self.attn(E, M, M, attn_mask = causal_mask)
        S = self.attn_dropout(S)
        S = S.permute(1, 0, 2)
        M = M.permute(1, 0, 2)
        E = E.permute(1, 0, 2)

        S = self.atten_layer_norm(S + M + E)

        F = self.FFN(S)
        F = self.FFN_layer_norm(F + S)

        p = torch.sigmoid(self.pred(F)).squeeze()

        return p, attn_weights