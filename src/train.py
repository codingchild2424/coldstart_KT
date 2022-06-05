import numpy as np

import torch
from dataloaders.get_loaders import get_loaders, COLDSTART1
from models.get_models import get_models
from trainers.get_trainers import get_trainers
from visualizers.get_visualizers import get_visualizers
from utils import get_optimizers, get_crits, recoder

from define_argparser import define_argparser

def main(config, train_loader=None, test_loader=None, num_q=None):
    #0. device 선언
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device('cuda:%d' % config.gpu_id)
    
    #1. 데이터 받아오기
    if config.five_fold == True:
        train_loader = train_loader
        test_loader = test_loader
        num_q = num_q
    else:
        train_loader, test_loader, num_q = get_loaders(config)

    #2. model 선택
    model = get_models(num_q, device, config)
    
    #3. optimizer 선택
    optimizer = get_optimizers(model, config)
    
    #4. criterion 선택
    crit = get_crits(config)
    
    #5. trainer 선택
    trainer = get_trainers(model, optimizer, device, num_q, crit, config)

    #6. 훈련 및 score 계산
    y_true_record, y_score_record, highest_auc_score = trainer.train(train_loader, test_loader)

    #7. model 기록 저장 위치
    #각 모델별로 따로 기록 저장하도록 폴더 만들어서 관리하기
    #파일 이름에 auc기록과 시간이 자동으로 기록되도록 넣기
    model_path = '../model_records/' + config.model_fn

    #8. model 기록
    torch.save({
        'model': trainer.model.state_dict(),
        'config': config
    }, model_path)

    #9. 시각화 결과물 만들기, + 시각화 만들기 전에 csv 형태로 기록하기
    # get_visualizers(
    #     y_true_record, y_score_record,model, model_path, test_loader, device, config
    # )

    #10. highest_auc_score 기록하기
    #recoder(highest_auc_score, config)

    return highest_auc_score

#main
if __name__ == "__main__":
    config = define_argparser() #define_argparser를 불러옴

    random_idx = None

    #랜덤 인덱스 생성
    if config.dataset_name == 'coldstart1' or config.dataset_name == 'coldstart1_2009':
        dataset = COLDSTART1()
        u_list = dataset.u_list
        random_idx = np.random.choice(len(u_list), config.stu_num, replace=False)

        if config.five_fold == True:

            highest_auc_scores = []

            for idx in range(5):
                train_loader, test_loader, num_q = get_loaders(config, idx, random_idx)
                highest_auc_score = main(config, train_loader, test_loader, num_q)
                highest_auc_scores.append(highest_auc_score)

            #highest_auc_scores 이걸 평균낸 값

            highest_auc_scores_average = sum(highest_auc_scores)/5 #five fold이므로

            recoder(highest_auc_scores_average, config)

    elif config.dataset_name == 'coldstart2':

        if config.five_fold == True:

            highest_auc_scores = []

            for idx in range(5):
                train_loader, test_loader, num_q = get_loaders(config, idx)
                highest_auc_score = main(config, train_loader, test_loader, num_q)
                highest_auc_scores.append(highest_auc_score)

            #highest_auc_scores 이걸 평균낸 값

            highest_auc_scores_average = sum(highest_auc_scores)/5 #five fold이므로

            recoder(highest_auc_scores_average, config)
        
    else:
        highest_auc_score = main(config)
        recoder(highest_auc_score, config)