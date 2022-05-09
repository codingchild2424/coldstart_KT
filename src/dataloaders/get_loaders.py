from torch.utils.data import DataLoader, random_split
from utils import collate_fn
from dataloaders.assist2015_loader import ASSIST2015
from dataloaders.assist2009_loader import ASSIST2009

#get_loaders를 따로 만들고, 이 함수를 train에서 불러내기
def get_loaders(config):

    #1. dataset 선택
    if config.dataset_name == "assist2015":
        dataset = ASSIST2015()
    #-> 추가적인 데이터셋
    elif config.dataset_name == "assist2009":
        dataset = ASSIST2009
        
    #일단 작동시키고 나중에 모듈화 할 것

    #coldstart 실험에 대한 2009 dataloader 세팅
    elif config.dataset_name == "coldstart1_assist2009":
        pass
    elif config.dataset_name == "coldstart2_assist2009":
        pass
    elif config.dataset_name == "coldstart3_assist2009":
        pass

    #coldstart 실험에 대한 2015 dataloader 세팅
    elif config.dataset_name == "coldstart1_assist2015":
        pass
    elif config.dataset_name == "coldstart2_assist2015":
        pass
    elif config.dataset_name == "coldstart3_assist2015":
        pass

    else:
        print("Wrong dataset_name was used...")

    #num_q 받아오기
    num_q = dataset.num_q

    #train, test 사이즈 나누기
    train_size = int( len(dataset) * config.train_ratio)
    test_size = len(dataset) - train_size

    #train, test 각각 랜덤하게 섞어서 나누기
    train_dataset, test_dataset = random_split(
        dataset, [ train_size, test_size ]
    )

    #train, test 데이터 섞기
    train_loader = DataLoader(
        train_dataset,
        batch_size = config.batch_size,
        shuffle = True,
        collate_fn = collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size = config.batch_size,
        shuffle = True,
        collate_fn = collate_fn
    )

    return train_loader, test_loader, num_q