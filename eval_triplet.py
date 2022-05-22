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

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    
    data_loaders, data_size = get_dataloader_triplets(root_dir, val_size,test_size,
                                                 num_triplets,
                                                 batch_size, num_workers)
    print(save_dir + best_ckpt_name)
    ckpt_path = torch.load(save_dir + best_ckpt_name)

    model.load_state_dict(ckpt_path['state_dict'])
    
    eval_facenet_model(model,data_loaders,phase='train',margin=margin,data_size=data_size)
    eval_facenet_model(model,data_loaders,phase='val',margin=margin,data_size=data_size)
    eval_facenet_model(model,data_loaders,phase='test',margin=margin,data_size=data_size)

