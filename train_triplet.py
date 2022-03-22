import argparse
import datetime
import time
import pandas as pd
import numpy as np
import os 
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

parser = argparse.ArgumentParser(description='Face Recognition using Triplet Loss')

parser.add_argument('--num-epochs', default=200,required=True, type=int,
                    help='number of epochs to train (default: 200)')
parser.add_argument('--num-triplets', default=10000,required=True, type=int,
                    help='number of triplets for training (default: 10000)')
parser.add_argument('--batch-size', default=32,required=True, type=int, 
                    help='batch size (default: 32)')
parser.add_argument('--num-workers', default=0, type=int,
                    help='number of workers (default: 0)')
parser.add_argument('--learning-rate', default=0.001,required=True, type=float,
                    help='learning rate (default: 0.001)')
parser.add_argument('--margin', default=0.5, required=True,type=float, 
                    help='margin (default: 0.5)')
parser.add_argument('--root-dir',type = str, default='D:\Academics\HonsProject1\Labelled Faces In The Wild Dataset\lfw-deepfunneled\lfw-deepfunneled',
                    help='path to train root dir')
parser.add_argument('--save-dir',type = str, default='D:\Academics\HonsProject1',help='path to save dir')
parser.add_argument('--val-size' , type=float , default=0.2,help='validation split ratio(default:0.2)')
parser.add_argument('--test-size' , type=float, default=0.2,help='test split ratio(default:0.2)')
parser.add_argument('--step-size', default=10, type=int,
                    help='Decay learning rate schedules every --step-size (default: 50)')
parser.add_argument('--unfreeze', type=str,  default='',
                    help='Provide an option for unfreezeing given layers')
parser.add_argument('--freeze', type=str, default='',
                    help='Provide an option for freezeing given layers')
parser.add_argument('--pretrain_checkpoint',default='casia-webface' ,type=str,help='Pretrained checkpoints casia-webface or vggface2')
parser.add_argument('--fc-only',default= False, action='store_true',help='Train fc only')
parser.add_argument('--except-fc',default=False, action='store_true',help='Train the base except fc layer')
parser.add_argument('--load-best',default=False, action='store_true',help='Load the best checkpoint')
parser.add_argument('--load-last',default=False, action='store_true',help='Load the last checkpoint')
parser.add_argument('--continue-step',default=False, action='store_true',help='resume training from last checkpoint')
parser.add_argument('--train-all', action='store_true',default=True, help='Train all layers')
parser.add_argument('--last_ckpt_name',required=True,help='Last checkpoint name')
parser.add_argument('--best_ckpt_name',required=True,help='Best checkpoint name')
args = parser.parse_args()

print(args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

l2_dist = PairwiseDistance(2)

modelsaver = ModelSaver()

def save_if_best(state, acc,dir,ckptname):
    modelsaver.save_if_best(acc, state,dir,ckptname)

def main():
    init_log_just_created("log/val.csv")
    init_log_just_created("log/train.csv")
    
    valid = pd.read_csv('log/val.csv')
    max_acc = valid['acc'].max()

    pretrain = args.pretrain_checkpoint
    fc_only = args.fc_only
    except_fc = args.except_fc
    train_all = args.train_all
    unfreeze = args.unfreeze.split(',')
    freeze = args.freeze.split(',')
    save_dir = args.save_dir
    start_epoch = 0
    print(f"Transfer learning: {pretrain}")
    print("Train fc only:", fc_only)
    print("Train except fc:", except_fc)
    print("Train all layers:", train_all)
    print("Unfreeze only:", ', '.join(unfreeze))
    print("Freeze only:", ', '.join(freeze))
    print(f"Max acc: {max_acc:.4f}")
    print(f"Learning rate will decayed every {args.step_size}th epoch")
    print("Save dir",save_dir)
    print("Last checkpoint name",args.last_ckpt_name)
    print("Best checkpoint name",args.best_ckpt_name)
    model = FaceNetModel(pretrained=pretrain)
    model.to(device)


    triplet_loss = TripletLoss(args.margin).to(device)

    if fc_only:
        model.unfreeze_only(['fc', 'classifier'])
    if except_fc:
        model.freeze_only(['fc', 'classifier'])
    if train_all:
        model.unfreeze_all()
    if len(unfreeze) > 0:
        model.unfreeze_only(unfreeze)
    if len(freeze) > 0:
        model.freeze_only(freeze)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    if args.load_best or args.load_last:
        try:
            checkpoint = './log/best_state.pth' if args.load_best else './log/last_checkpoint.pth'
            print('loading', checkpoint)
            checkpoint = torch.load(checkpoint)
            modelsaver.current_acc = max_acc
            start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['state_dict'])
        except ValueError as e:
            print("Can't load last checkpoint")
            print(e)
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
        except ValueError as e:
            print("Can't load last optimizer")
            print(e)
        if args.continue_step:
            scheduler.step(checkpoint['epoch'])
        print(f"Loaded checkpoint epoch: {checkpoint['epoch']}\n"
              f"Loaded checkpoint accuracy: {checkpoint['accuracy']}\n"
              f"Loaded checkpoint loss: {checkpoint['loss']}")


    model = torch.nn.DataParallel(model)
    
    for epoch in range(start_epoch, args.num_epochs + start_epoch):
        print(120 * '=')
        print('Epoch [{}/{}]'.format(epoch, args.num_epochs + start_epoch - 1))

        time0 = time.time()
        data_loaders, data_size = get_dataloader(args.root_dir, args.val_size,args.test_size,
                                                 args.num_triplets,
                                                 args.batch_size, args.num_workers)

        train_valid(model, optimizer, triplet_loss, scheduler, epoch, data_loaders, data_size,save_dir)
        print(f'  Execution time                 = {time.time() - time0}')
    print(120 * '=')
    eval_facenet_model(model,data_loaders,phase='test',margin=args.margin,data_size=data_size)
    


def save_last_checkpoint(state,dir,ckptname):
    torch.save(state, os.path.join(dir,ckptname))


def train_valid(model, optimizer, trip_loss, scheduler, epoch, dataloaders, data_size,dir):
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

                all = (neg_dist - pos_dist < args.margin).cpu().numpy().flatten()
                if phase == 'train':
                    hard_triplets = np.where(all == 1)
                    if len(hard_triplets[0]) == 0:
                        continue
                else:
                    hard_triplets = np.where(all >= 0)

                anc_hard_embed = anc_embed[hard_triplets]
                pos_hard_embed = pos_embed[hard_triplets]
                neg_hard_embed = neg_embed[hard_triplets]

                anc_hard_img = anc_img[hard_triplets]
                pos_hard_img = pos_img[hard_triplets]
                neg_hard_img = neg_img[hard_triplets]

            
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
        layers = '+'.join(args.unfreeze.split(','))
        write_csv(f'log/{phase}.csv', [time, epoch, np.mean(accuracy), avg_triplet_loss, layers, args.batch_size, lr])

        if phase == 'val':
            save_last_checkpoint({'epoch': epoch,
                                  'state_dict': model.module.state_dict(),
                                  'optimizer_state': optimizer.state_dict(),
                                  'accuracy': np.mean(accuracy),
                                  'loss': avg_triplet_loss
                                  },
                                  dir,
                                  args.last_ckpt_name)
            save_if_best({'epoch': epoch,
                          'state_dict': model.module.state_dict(),
                          'optimizer_state': optimizer.state_dict(),
                          'accuracy': np.mean(accuracy),
                          'loss': avg_triplet_loss
                          }, 
                          np.mean(accuracy),
                          dir,
                          args.best_ckpt_name)
        else:
            plot_roc(fpr, tpr, figure_name='./log/roc_valid_epoch_{}.png'.format(epoch))


if __name__ == '__main__':
    main()
