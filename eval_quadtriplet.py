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
from loss import QuadTripletLoss
from model import FaceNetModel
from utils import ModelSaver, eval_quad_facenet_model, init_log_just_created,write_csv,eval_facenet_model
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


    qtriplet_loss = QuadTripletLoss(alpha1,alpha2,alpha3,alpha4).to(device)

    if fc_only:
        model.unfreeze_only(['fc', 'classifier'])
    if except_fc:
        model.freeze_only(['fc', 'classifier'])
    if train_all:
        model.unfreeze_all()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    data_loaders, data_size = get_dataloader_quadtriplets(root_dir_original, root_dir_masked,val_size,test_size,
                                                 num_triplets,
                                                 batch_size, num_workers)

    ckpt_path = save_dir + best_ckpt_name

    model.load_state_dict(ckpt_path)
    
    eval_quad_facenet_model(model,data_loaders,'train',alpha1 , alpha2 , alpha3 , alpha4  ,data_size = data_size)
    eval_quad_facenet_model(model,data_loaders,'val',alpha1 , alpha2 , alpha3 , alpha4  ,data_size = data_size)
    eval_quad_facenet_model(model,data_loaders,'test',alpha1 , alpha2 , alpha3 , alpha4  ,data_size = data_size)

