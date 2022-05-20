from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
from utils import collate_fn
from dataloaders.assist2015_loader import ASSIST2015
from dataloaders.assist2009_loader import ASSIST2009

from dataloaders.coldstart_dataloaders.cold2015_loader import CODL2015_TRAIN, CODL2015_TEST


#get_loaders를 따로 만들고, 이 함수를 train에서 불러내기
def get_loaders(config):

    #1. dataset 선택
    if config.dataset_name == "assist2015":
        dataset = ASSIST2015()
    #-> 추가적인 데이터셋
    elif config.dataset_name == "assist2009":
        dataset = ASSIST2009()
    #coldstart 실험에 대한 2015 dataloader 세팅
    elif config.dataset_name == "coldstart1_assist2015":
        #num_q를 받아오기 위해 설정
        dataset = ASSIST2015()
        
        #학생별 인원수를 다르게 하며 실험하기 위해 따로 설정함
        train_path = "../datasets/coldstart_datasets/coldstart1_2015_traindatasets/coldstart1_num" \
            + config.cold1_stu_num + ".csv"

        #train과 test를 받아오기 위해 설정
        #고정된 cold_train과 cold_test를 가져옴
        cold_train = CODL2015_TRAIN(
            dataset_dir =  train_path
        )
        cold_test = CODL2015_TEST()
    else:
        print("Wrong dataset_name was used...")

    #총 퀴즈의 숫자
    num_q = dataset.num_q

    #train, test 사이즈 나누기
    train_size = int( len(dataset) * config.train_ratio)
    test_size = len(dataset) - train_size

    #2. dataset을 trainset과 testset으로 분류, num_q 추출
    if config.dataset_name == "coldstart1_assist2015":
        #여기서 고정된 dataset 가져오기
        train_dataset = cold_train
        test_dataset = cold_test

        #shuffle control, coldstart1문제일 경우,
        #다른 모델과의 통일된 실험을 위해서는 shuffle을 하지 않은 상태로 진행해야 함
        #trainset shuffle은 preprocessor를 통해 csv파일 자체를 고치는 방향으로 실험
        train_shuffle = False
    else:
        #train, test 각각 랜덤하게 섞어서 나누기
        train_dataset, test_dataset = random_split(
            dataset, [ train_size, test_size ]
        )
        #train_shuffle, 기존 모델에서는 섞어줘야 함
        train_shuffle = True

    #train, test 데이터 섞기
    train_loader = DataLoader(
        train_dataset,
        batch_size = config.batch_size,
        #config를 통해 coldstart일때는 섞지 않도록 설정함
        shuffle = train_shuffle,
        collate_fn = collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size = config.batch_size,
        shuffle = True,
        collate_fn = collate_fn
    )

    return train_loader, test_loader, num_q