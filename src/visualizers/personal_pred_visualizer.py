from paddle import unsqueeze
import torch
import random
import matplotlib.pyplot as plt
import numpy as np

def dkt_personal_pred_visualizer(model, model_path, test_loader, device, img_name):

    img_path = '../imgs/personal_pred/' + img_name + '_persoanl_pred'

    model = model
    checkpoint = torch.load(model_path)
    model.load_state_dict( checkpoint['model'] )
    model.to(device)

    #test
    with torch.no_grad():
        for data in test_loader:
            
            q_seqs, r_seqs, _, _, mask_seqs = data

            random_idx = random.randint(0, len(q_seqs) - 1) #|len(q_seqs)| = bs

            q_seqs = q_seqs[random_idx].to(device)
            r_seqs = r_seqs[random_idx].to(device)
            mask_seqs = mask_seqs[random_idx].to(device)

            masked_q_seqs = torch.masked_select( q_seqs, mask_seqs )
            masked_r_seqs = torch.masked_select( r_seqs, mask_seqs )

            masked_unsqueeze_q_seqs = torch.unsqueeze(masked_q_seqs, 0)
            masked_unsqueeze_r_seqs = torch.unsqueeze(masked_r_seqs, 0)

            y_hat = model( masked_unsqueeze_q_seqs.long(), masked_unsqueeze_r_seqs.long() )

            pred = y_hat[0, :, :].detach().cpu().numpy()

            plt.subplot(121)
            plt.figure(figsize=(12,5))
            plt.imshow(pred)
            plt.xlabel('Index of item')
            plt.ylabel('Number of responses')
            plt.colorbar()

            plt.savefig(img_path + '_imshow' + '.jpg')

            plt.subplot(122)
            plt.figure(figsize=(12,5))
            plt.plot(np.mean(pred ,axis=1), c='red')
            plt.plot(pred, c='black', alpha=0.15)
            plt.legend(['Mean', 'Each item'])
            plt.xlabel('Number of responses')
            
            plt.savefig(img_path + '_plot' + '.jpg')

            #반복못하도록 한번만 돌리기
            break

def dkvmn_personal_pred_visualizer(model, model_path, test_loader, device, img_name):

    img_path = '../imgs/personal_pred/' + img_name + '_persoanl_pred'

    model = model
    checkpoint = torch.load(model_path)
    model.load_state_dict( checkpoint['model'] )
    model.to(device)

    #test
    with torch.no_grad():
        for data in test_loader:
            
            q_seqs, r_seqs, _, _, mask_seqs = data

            random_idx = random.randint(0, len(q_seqs) - 1) #|len(q_seqs)| = bs

            q_seqs = q_seqs[random_idx].to(device)
            r_seqs = r_seqs[random_idx].to(device)
            mask_seqs = mask_seqs[random_idx].to(device)

            masked_q_seqs = torch.masked_select( q_seqs, mask_seqs )
            masked_r_seqs = torch.masked_select( r_seqs, mask_seqs )

            masked_unsqueeze_q_seqs = torch.unsqueeze(masked_q_seqs, 0)
            masked_unsqueeze_r_seqs = torch.unsqueeze(masked_r_seqs, 0)

            y_hat, _ = model( masked_unsqueeze_q_seqs.long(), masked_unsqueeze_r_seqs.long() )

            #y_hat에서 예측한 값이 각 문항에 대한 확률값만 반환됨. 각 문항을 통해 전체 문항에 대한 예측값이 나와야 그래프를 그릴 수 있음

            pred = y_hat.unsqueeze(0).detach().cpu().numpy()
        
            plt.subplot(121)
            plt.figure(figsize=(12,5))
            plt.imshow(pred)
            plt.xlabel('Index of item')
            plt.ylabel('Number of responses')
            plt.colorbar()

            plt.savefig(img_path + '_imshow' + '.jpg')

            plt.subplot(122)
            plt.figure(figsize=(12,5))
            plt.plot(np.mean(pred ,axis=1), c='red')
            plt.plot(pred, c='black', alpha=0.15)
            plt.legend(['Mean', 'Each item'])
            plt.xlabel('Number of responses')
            
            plt.savefig(img_path + '_plot' + '.jpg')

            #반복못하도록 한번만 돌리기
            break