from trainers.dkt_trainer import DKT_trainer
from trainers.dkvmn_trainer import DKVMN_trainer

def get_trainers(model, optimizer, device, num_q, crit, config):

    #trainer 실행
    if config.model_name == "dkt":
        trainer = DKT_trainer(
            model = model,
            optimizer = optimizer,
            n_epochs = config.n_epochs,
            device = device,
            num_q = num_q,
            crit = crit
        )
    elif config.model_name == "dkvmn":
        trainer = DKVMN_trainer(
            model = model,
            optimizer = optimizer,
            n_epochs = config.n_epochs,
            device = device,
            num_q = num_q,
            crit = crit
        )
    else:
        print("Wrong trainer_name was used...")

    return trainer
