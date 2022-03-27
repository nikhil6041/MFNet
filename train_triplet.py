import argparse
import datetime
import time
import pandas as pd
import numpy as np
import os 
import yaml
import torch
import torch.optim as optim
from torch.nn.modules.distance import PairwiseDistance
from torch.optim import lr_scheduler

from dataset import get_dataloader_triplets
from eval_metrics import evaluate, plot_roc
from loss import TripletLoss
from model import FaceNetModel
from utils import ModelSaver, init_log_just_created,write_csv,eval_facenet_model
from pprint import pprint as ppt

l2_dist = PairwiseDistance(2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

modelsaver = ModelSaver()

def save_if_best(state, acc,dir,ckptname):
    modelsaver.save_if_best(acc, state,dir,ckptname)

def save_last_checkpoint(state,dir,ckptname):
    torch.save(state, os.path.join(dir,ckptname))


def train_valid_triplet( model, optimizer, trip_loss, margin, scheduler, epoch, dataloaders , batch_size , data_size , save_dir , logs_dir, last_ckpt_name , best_ckpt_name ):
    
    for phase in ['train', 'val']:

        labels, distances = [], []
        triplet_loss_sum = 0.0

        if phase == 'train':
            model.train()
        else:
            model.eval()

        for _, batch_sample in enumerate(dataloaders[phase]):

            anc_img = batch_sample['anc_img'].to(device)
            pos_img = batch_sample['pos_img'].to(device)
            neg_img = batch_sample['neg_img'].to(device)

       
            with torch.set_grad_enabled(phase == 'train'):

                # anc_embed, pos_embed and neg_embed are encoding(embedding) of image
                anc_embed, pos_embed, neg_embed = model(anc_img), model(pos_img), model(neg_img)

                # choose the semi hard negatives only for "training"
                pos_dist = l2_dist.forward(anc_embed, pos_embed)
                neg_dist = l2_dist.forward(anc_embed, neg_embed)

                all = (neg_dist - pos_dist < margin).cpu().numpy().flatten()
                if phase == 'train':
                    hard_triplets = np.where(all == 1)
                    if len(hard_triplets[0]) == 0:
                        continue
                else:
                    hard_triplets = np.where(all >= 0)

                anc_hard_embed = anc_embed[hard_triplets]
                pos_hard_embed = pos_embed[hard_triplets]
                neg_hard_embed = neg_embed[hard_triplets]

                # anc_hard_img = anc_img[hard_triplets]
                # pos_hard_img = pos_img[hard_triplets]
                # neg_hard_img = neg_img[hard_triplets]

            
                # model.module.forward_classifier(anc_hard_img)
                # model.module.forward_classifier(pos_hard_img)
                # model.module.forward_classifier(neg_hard_img)

                triplet_loss = trip_loss.forward(anc_hard_embed, pos_hard_embed, neg_hard_embed)

                if phase == 'train':
                    optimizer.zero_grad()
                    triplet_loss.backward()
                    optimizer.step()

                distances.append(pos_dist.data.cpu().numpy())
                labels.append(np.ones(pos_dist.size(0)))

                distances.append(neg_dist.data.cpu().numpy())
                labels.append(np.zeros(neg_dist.size(0)))

                triplet_loss_sum += triplet_loss.item()

        if phase == 'train':
            print("Stepping LR")
            scheduler.step()
            if scheduler.last_epoch % scheduler.step_size == 0:
                print("LR decayed to:", ', '.join(map(str, scheduler.get_last_lr())))
    
        avg_triplet_loss = triplet_loss_sum / data_size[phase]

        labels = np.array([sublabel for label in labels for sublabel in label])
        
        distances = np.array([subdist for dist in distances for subdist in dist])

        tpr, fpr, accuracy, val, val_std, far = evaluate(distances, labels)
        
        print('  {} set - Triplet Loss       = {:.8f}'.format(phase, avg_triplet_loss))
        
        print('  {} set - Accuracy           = {:.8f}'.format(phase, np.mean(accuracy)))

     
        time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lr = '_'.join(map(str, scheduler.get_last_lr()))

        write_csv(f'{logs_dir}/{phase}.csv', [time, epoch, np.mean(accuracy), avg_triplet_loss, batch_size, lr])

        if phase == 'val':
            save_last_checkpoint({'epoch': epoch,
                                  'state_dict': model.state_dict(),
                                  'optimizer_state': optimizer.state_dict(),
                                  'accuracy': np.mean(accuracy),
                                  'loss': avg_triplet_loss
                                  },
                                  save_dir,
                                  last_ckpt_name)
            save_if_best({'epoch': epoch,
                          'state_dict': model.state_dict(),
                          'optimizer_state': optimizer.state_dict(),
                          'accuracy': np.mean(accuracy),
                          'loss': avg_triplet_loss
                          }, 
                          np.mean(accuracy),
                          save_dir,
                          best_ckpt_name)
        else:
            plot_roc(fpr, tpr, figure_name='{}/roc_valid_epoch_{}.png'.format(logs_dir,epoch))    

if __name__ == '__main__':


    config_file_path = "config.yaml"
    try:
        ## loading the configuration
        with open(config_file_path, "r") as stream:
            config = yaml.safe_load(stream)
            
    except:
        print("Could not find the config file,please pass the path correctly")
        exit()
    parser = argparse.ArgumentParser(description='Masked Face Recognition')
    parser.add_argument('--variant',type=str,help="Name of variant a) triplet b) quad triplet")
    args = vars(parser.parse_args())

    variant = args['variant']
    config = config[variant]
    
    tr_params = config['training_params']
    ds_params = config['dataset']
    model_params = config['model']

    num_epochs = tr_params['num_epochs']
    learning_rate = tr_params['learning_rate']
    margin = tr_params['margin']
    step_size = tr_params['step_size']
    logs_dir = tr_params['logs_dir']

    pretrain_checkpoint = model_params['pretrain_checkpoint']
    last_ckpt_name = model_params['last_ckpt_name']
    best_ckpt_name = model_params['best_ckpt_name']
    fc_only = model_params['fc_only']
    except_fc = model_params['except_fc']
    train_all = model_params['train_all']

    num_triplets = ds_params['num_triplets']
    batch_size = ds_params['batch_size']
    num_workers = ds_params['num_workers']
    root_dir = ds_params['root_dir']
    save_dir = ds_params['save_dir']
    val_size = ds_params['val_size']
    test_size = ds_params['test_size']
    

    init_log_just_created(f"{logs_dir}/val.csv")
    init_log_just_created(f"{logs_dir}/train.csv")
    
    valid = pd.read_csv(f"{logs_dir}/val.csv")
    max_acc = valid['acc'].max()


    start_epoch = 0
    print(f"Transfer learning: {pretrain_checkpoint}")
    print("Train fc only:", fc_only)
    print("Train except fc:", except_fc)
    print("Train all layers:", train_all)
    print(f"Max acc: {max_acc:.4f}")
    print(f"Learning rate will decayed every {step_size}th epoch")
    print("Save dir",save_dir)
    print("Last checkpoint name",last_ckpt_name)
    print("Best checkpoint name",best_ckpt_name)
    model = FaceNetModel(pretrained=pretrain_checkpoint)
    model.to(device)


    triplet_loss = TripletLoss(margin).to(device)

    if fc_only:
        model.unfreeze_only(['fc', 'classifier'])
    if except_fc:
        model.freeze_only(['fc', 'classifier'])
    if train_all:
        model.unfreeze_all()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    
    for epoch in range(start_epoch, num_epochs + start_epoch):
        print(120 * '=')
        print('Epoch [{}/{}]'.format(epoch, num_epochs + start_epoch - 1))

        time0 = time.time()
        data_loaders, data_size = get_dataloader_triplets(root_dir, val_size,test_size,
                                                 num_triplets,
                                                 batch_size, num_workers)

        train_valid_triplet( model, optimizer, triplet_loss, margin, scheduler, epoch, data_loaders , batch_size , data_size , save_dir ,logs_dir, last_ckpt_name , best_ckpt_name )
        print(f'  Execution time                 = {time.time() - time0}')
    print(120 * '=')
    eval_facenet_model(model,data_loaders,phase='test',margin=margin,data_size=data_size)

