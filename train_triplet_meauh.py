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

from dataset import get_dataloader_quadtriplets
from eval_metrics import evaluate, plot_roc
from loss import TripletLoss_meauh
from model import FaceNetModel
from utils import ModelSaver, eval_quad_facenet_model, init_log_just_created,write_csv
from pprint import pprint as ppt

l2_dist = PairwiseDistance(2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

modelsaver = ModelSaver()

def save_if_best(state, acc,dir,ckptname):
    modelsaver.save_if_best(acc, state,dir,ckptname)

def save_last_checkpoint(state,dir,ckptname):
    torch.save(state, os.path.join(dir,ckptname))


def get_triplets(anc_embed,pos_embed,neg_embed,margin,phase):
    # choose the semi hard negatives only for "training"
    pos_dist = l2_dist.forward(anc_embed, pos_embed)
    neg_dist = l2_dist.forward(anc_embed, neg_embed)

    all = (neg_dist - pos_dist < margin).cpu().numpy().flatten()
    if phase == 'train':
        hard_triplets = np.where(all == 1)
    else:
        hard_triplets = np.where(all >= 0)

    anc_hard_embed = anc_embed[hard_triplets]
    pos_hard_embed = pos_embed[hard_triplets]
    neg_hard_embed = neg_embed[hard_triplets]

    if anc_hard_embed.dim() == 2 and pos_hard_embed.dim() == 2 and neg_hard_embed.dim() == 2:
        return anc_hard_embed,pos_hard_embed,neg_hard_embed,pos_dist,neg_dist
    else:
        return None

def train_valid_triplet_meauh( model, optimizer, qtrip_loss, alpha1 , scheduler, epoch, dataloaders , batch_size , data_size , save_dir , logs_dir, last_ckpt_name , best_ckpt_name ):
    
    for phase in ['train', 'val']:

        labels_u, distances_u = [], []
        labels_m, distances_m = [], []
        triplet_meahu_sum = 0.0

        if phase == 'train':
            model.train()
        else:
            model.eval()

        for _, batch_sample in enumerate(dataloaders[phase]):

            anc_img_orig = batch_sample['anc_img_orig'].to(device)
            pos_img_orig = batch_sample['pos_img_orig'].to(device)
            neg_img_orig = batch_sample['neg_img_orig'].to(device)
            anc_img_mask = batch_sample['anc_img_mask'].to(device)
            pos_img_mask = batch_sample['pos_img_mask'].to(device)
            neg_img_mask = batch_sample['neg_img_mask'].to(device)
       
            with torch.set_grad_enabled(phase == 'train'):

                # anc_embed, pos_embed and neg_embed are encoding(embedding) of image
                anc_embed_u, pos_embed_u, neg_embed_u  = model(anc_img_orig), model(pos_img_orig), model(neg_img_orig)
                anc_embed_m, pos_embed_m, neg_embed_m  = model(anc_img_mask), model(pos_img_mask), model(neg_img_mask)

                if phase == "train":
                    embs_u = get_triplets(anc_embed_u , pos_embed_u , neg_embed_u , alpha1 ,phase )
                    if embs_u is not None:
                        anc_hard_embed_u , pos_hard_embed_u , neg_hard_embed_u  , pos_dist_u , neg_dist_u = embs_u
                        print(f"Unmasked anc {anc_hard_embed_u.size()} , pos {pos_hard_embed_u.size()} , neg {neg_hard_embed_u.size()}")
                    else:
                        continue
                    embs_m = get_triplets(anc_embed_m , pos_embed_m , neg_embed_m , alpha1 , phase)
                    if embs_m is not None:
                        anc_hard_embed_m , pos_hard_embed_m , neg_hard_embed_m, pos_dist_m , neg_dist_m = embs_m
                        print(f"Masked anc {anc_hard_embed_m.size()} , pos {pos_hard_embed_m.size()} , neg {neg_hard_embed_m.size()}")

                    else:
                        continue
                else:
                    
                    embs_u = get_triplets(anc_embed_u , pos_embed_u , neg_embed_u , alpha1 ,phase )
                    anc_hard_embed_u , pos_hard_embed_u , neg_hard_embed_u  , pos_dist_u , neg_dist_u = embs_u
                    print(f"Unmasked anc {anc_hard_embed_u.size()} , pos {pos_hard_embed_u.size()} , neg {neg_hard_embed_u.size()}")
         
                    embs_m = get_triplets(anc_embed_m , pos_embed_m , neg_embed_m , alpha1 , phase )
                    anc_hard_embed_m , pos_hard_embed_m , neg_hard_embed_m, pos_dist_m , neg_dist_m = embs_m
                    print(f"Masked anc {anc_hard_embed_m.size()} , pos {pos_hard_embed_m.size()} , neg {neg_hard_embed_m.size()}")

                if anc_hard_embed_u.size(dim = 0) == anc_hard_embed_m.size(dim = 0) == pos_hard_embed_u.size(dim=0) == pos_hard_embed_m.size(dim=0) == neg_hard_embed_u.size(dim=0) == neg_hard_embed_m.size(dim=0):
                    triplet_meahu = qtrip_loss.forward( anc_hard_embed_u , pos_hard_embed_u , neg_hard_embed_u , anc_hard_embed_m )

                else:
                    l = [anc_hard_embed_u.size(dim = 0), anc_hard_embed_m.size(dim = 0) , pos_hard_embed_u.size(dim=0) , pos_hard_embed_m.size(dim=0) , neg_hard_embed_u.size(dim=0) , neg_hard_embed_m.size(dim=0)]
                    ln = min(l)
                    if ln > 0:
                        anc_hard_embed_u = anc_hard_embed_u[:ln]
                        anc_hard_embed_m = anc_hard_embed_m[:ln]
                        pos_hard_embed_u = pos_hard_embed_u[:ln]
                        pos_hard_embed_m = pos_hard_embed_m[:ln]
                        neg_hard_embed_u = neg_hard_embed_u[:ln]
                        neg_hard_embed_m = neg_hard_embed_m[:ln]

                        print(f"Unmasked anc {anc_hard_embed_u.size()} , pos {pos_hard_embed_u.size()} , neg {neg_hard_embed_u.size()}")

                        print(f"Masked anc {anc_hard_embed_m.size()} , pos {pos_hard_embed_m.size()} , neg {neg_hard_embed_m.size()}")
                        triplet_meahu = qtrip_loss.forward( anc_hard_embed_u , pos_hard_embed_u , neg_hard_embed_u , anc_hard_embed_m )

                    else:
                        continue

                if phase == 'train':
                    optimizer.zero_grad()
                    triplet_meahu.backward()
                    optimizer.step()

                distances_u.append(pos_dist_u.data.cpu().numpy())
                labels_u.append(np.ones(pos_dist_u.size(0)))

                distances_u.append(neg_dist_u.data.cpu().numpy())
                labels_u.append(np.zeros(neg_dist_u.size(0)))

                distances_m.append(pos_dist_m.data.cpu().numpy())
                labels_m.append(np.ones(pos_dist_m.size(0)))

                distances_m.append(neg_dist_m.data.cpu().numpy())
                labels_m.append(np.zeros(neg_dist_m.size(0)))

                triplet_meahu_sum += triplet_meahu.item()

        if phase == 'train':
            print("Stepping LR")
            scheduler.step()
            if scheduler.last_epoch % scheduler.step_size == 0:
                print("LR decayed to:", ', '.join(map(str, scheduler.get_last_lr())))
    
        avg_triplet_meahu = triplet_meahu_sum / data_size[phase]

        labels_u = np.array([sublabel for label in labels_u for sublabel in label])
        
        distances_u = np.array([subdist for dist in distances_u for subdist in dist])

        labels_m = np.array([sublabel for label in labels_m for sublabel in label])
        
        distances_m = np.array([subdist for dist in distances_m for subdist in dist])

        tpr_u, fpr_u, accuracy_u, val_u, val_std_u, far_u = evaluate(distances_u, labels_u)
        tpr_m, fpr_m, accuracy_m, val_m, val_std_m, far_m = evaluate(distances_m, labels_m)
        
        print('  {} set - QuadTriplet Loss       = {:.8f}'.format(phase, avg_triplet_meahu))
        
        print('  {} set - Accuracy (unmasked)          = {:.8f}'.format(phase, np.mean(accuracy_u)))

        print('  {} set - Accuracy (masked)          = {:.8f}'.format(phase, np.mean(accuracy_m)))
     
        time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lr = '_'.join(map(str, scheduler.get_last_lr()))

        write_csv(f'{logs_dir}/{phase}.csv', [time, epoch, np.mean(accuracy_u), avg_triplet_meahu, batch_size, lr, "unmasked"])
        write_csv(f'{logs_dir}/{phase}.csv', [time, epoch, np.mean(accuracy_m), avg_triplet_meahu, batch_size, lr , "masked"])

        if phase == 'val':
            save_last_checkpoint({'epoch': epoch,
                                  'state_dict': model.state_dict(),
                                  'optimizer_state': optimizer.state_dict(),
                                  'accuracy_m': np.mean(accuracy_m),
                                  'accuracy_u': np.mean(accuracy_u),
                                  'loss': avg_triplet_meahu
                                  },
                                  save_dir,
                                  last_ckpt_name)
            save_if_best({'epoch': epoch,
                          'state_dict': model.state_dict(),
                          'optimizer_state': optimizer.state_dict(),
                          'accuracy_m': np.mean(accuracy_m),
                          'accuracy_u': np.mean(accuracy_u),
                          'loss': avg_triplet_meahu
                          }, 
                          np.mean(accuracy_m),
                          save_dir,
                          best_ckpt_name)
        else:
            plot_roc(fpr_u, tpr_u, figure_name='{}/roc_unmasked_valid_epoch_{}.png'.format(logs_dir,epoch))    
            plot_roc(fpr_m, tpr_m, figure_name='{}/roc_masked_valid_epoch_{}.png'.format(logs_dir,epoch))    

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
    alpha1 = tr_params['alpha1']
    alpha2 = tr_params['alpha2']
    alpha3 = tr_params['alpha3']
    alpha4 = tr_params['alpha4']
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
    root_dir_original = ds_params['root_dir_original']
    root_dir_masked = ds_params['root_dir_masked']
    save_dir = ds_params['save_dir']
    val_size = ds_params['val_size']
    test_size = ds_params['test_size']
    
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
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

    triplet_meahu = TripletLoss_meauh(alpha1).to(device)

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
        data_loaders, data_size = get_dataloader_quadtriplets(root_dir_original, root_dir_masked,val_size,test_size,
                                                 num_triplets,
                                                 batch_size, num_workers)

        train_valid_triplet_meauh( model, optimizer, triplet_meahu, alpha1 , scheduler, epoch, data_loaders , batch_size , data_size , save_dir ,logs_dir, last_ckpt_name , best_ckpt_name )
        print(f'  Execution time                 = {time.time() - time0}')
    print(120 * '=')
    eval_quad_facenet_model(model,data_loaders,'test',alpha1 , alpha2 , alpha3 , alpha4  ,data_size=data_size)

