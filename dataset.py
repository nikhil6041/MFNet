import os
import numpy as np
import pandas as pd
import torch
from skimage import io
from torch.utils.data import Dataset,SubsetRandomSampler
from torchvision import transforms
from PIL import Image

class TripletFaceDataset(Dataset):

    def __init__(self, root_dir, num_triplets, transform=None):

        self.root_dir = root_dir
        self.num_triplets = num_triplets
        self.transform = transform
        self.training_triplets = self.generate_triplets(self.root_dir, self.num_triplets)

    @staticmethod
    def generate_triplets(root_dir, num_triplets):

        def make_dictionary_for_face_class(root_dir):

            '''
              - face_classes = {'class0': [class0_id0, ...], 'class1': [class1_id0, ...], ...}
            '''
            # d = dict()
            face_classes = dict()

            for label in os.listdir(root_dir):
                if label not in face_classes.keys():
                    vals = list(os.listdir(os.path.join(root_dir,label)))
                    face_classes[label] = vals
                    # for val in vals:
                    #     if val not in d.keys():
                    #         d[val] = label
            # df = pd.DataFrame({
            #     'name':list(d.keys()),
            #     'class':list(d.values())
            # })

            return face_classes

        triplets = []
        face_classes = make_dictionary_for_face_class(root_dir)
        classes = list(face_classes.keys())

        for _ in range(num_triplets):

            '''
              - randomly choose anchor, positive and negative images for triplet loss
              - anchor and positive images in pos_class
              - negative image in neg_class
              - at least, two images needed for anchor and positive images in pos_class
              - negative image should have different class as anchor and positive images by definition
            '''

            pos_class = np.random.choice(classes)
            neg_class = np.random.choice(classes)
            while len(face_classes[pos_class]) < 2:
                pos_class = np.random.choice(classes)
            while pos_class == neg_class:
                neg_class = np.random.choice(classes)


            if len(face_classes[pos_class]) == 2:
                ianc, ipos = np.random.choice(2, size=2, replace=False)
            else:
                ianc = np.random.randint(0, len(face_classes[pos_class]))
                ipos = np.random.randint(0, len(face_classes[pos_class]))
                while ianc == ipos:
                    ipos = np.random.randint(0, len(face_classes[pos_class]))
            ineg = np.random.randint(0, len(face_classes[neg_class]))

            anc_id = face_classes[pos_class][ianc]         
            pos_id = face_classes[pos_class][ipos]
            neg_id = face_classes[neg_class][ineg]

            triplets.append(
                [anc_id, pos_id, neg_id, pos_class, neg_class])

        return triplets

    def __getitem__(self, idx):

        anc_id, pos_id, neg_id, pos_class, neg_class = self.training_triplets[idx]

        anc_img = os.path.join(self.root_dir, str(pos_class) ,str(anc_id))
        pos_img = os.path.join(self.root_dir, str(pos_class), str(pos_id))
        neg_img = os.path.join(self.root_dir, str(neg_class), str(neg_id))

        anc_img = Image.open(anc_img).convert('RGB')
        pos_img = Image.open(pos_img).convert('RGB')
        neg_img = Image.open(neg_img).convert('RGB')

        # pos_class = torch.from_numpy(np.array([pos_class]).astype('long'))
        # neg_class = torch.from_numpy(np.array([neg_class]).astype('long'))

        sample = {'anc_img': anc_img, 'pos_img': pos_img, 'neg_img': neg_img, 'pos_class': pos_class,
                  'neg_class': neg_class}

        if self.transform:
            sample['anc_img'] = self.transform(sample['anc_img'])
            sample['pos_img'] = self.transform(sample['pos_img'])
            sample['neg_img'] = self.transform(sample['neg_img'])

        return sample

    def __len__(self):
        return len(self.training_triplets)


def get_dataloader(root_dir,val_size,
                   test_size,num_triplets,
                   batch_size, num_workers):
    # data_transforms = {
    #     'train': transforms.Compose([
    #         transforms.ToPILImage(),
    #         transforms.RandomRotation(15),
    #         transforms.RandomResizedCrop(224),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                              std=[0.229, 0.224, 0.225])]),
    #     'valid': transforms.Compose([
    #         transforms.ToPILImage(),
    #         transforms.Resize(224),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                              std=[0.229, 0.224, 0.225])])
    # }
    data_transforms = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

    face_dataset = TripletFaceDataset(
        root_dir=root_dir,
        num_triplets=num_triplets,
        transform=data_transforms
    )

    dataset_size = len(face_dataset)
    indices = list(range(0,dataset_size))

    split = int(np.floor(test_size*dataset_size))

    np.random.shuffle(indices)

    tv_indices , test_indices = indices[split:] , indices[:split]

    v_split = int(np.floor(len(tv_indices)*val_size))

    train_indices , val_indices = tv_indices[v_split:] , tv_indices[:v_split]

    face_dataset_indices = {
        'train':train_indices,
        'val':val_indices,
        'test':test_indices
    }

    face_dataset_sampler = {
        'train': SubsetRandomSampler(train_indices), 
        'val': SubsetRandomSampler(val_indices),
        'test': SubsetRandomSampler(test_indices) 
    }

    dataloaders = {
        x: torch.utils.data.DataLoader(face_dataset, batch_size=batch_size, sampler = face_dataset_sampler[x] , num_workers=num_workers,drop_last = True)
        for x in ['train', 'val','test']}

    data_size = {x: len(face_dataset_indices[x]) for x in ['train', 'val','test']}

    return dataloaders, data_size
