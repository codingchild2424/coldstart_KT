from models.dkt import DKT
from models.dkvmn import DKVMN

def get_models(num_q, device, config):

    if config.model_name == "dkt":
        model = DKT(
            num_q = num_q,
            emb_size = config.dkt_emb_size, #default = 100
            hidden_size = config.dkt_hidden_size #default = 100
        ).to(device)
    #-> 추가적인 모델 정의
    elif config.model_name == "dkvmn":
        model = DKVMN(
            num_q = num_q,
            dim_s = config.dkvmn_dim_s, #default = 50
            size_m = config.dkvmn_size_m #default = 20
        ).to(device)
    else:
        print("Wrong model_name was used...")

    return model