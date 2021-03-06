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

            face_classes = dict()

            for label in os.listdir(root_dir):
               
                if label not in face_classes.keys():
                  
                    vals = list(os.listdir(os.path.join(root_dir,label)))
                  
                    if len(vals) >= 10:
                        face_classes[label] = vals

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

class FaceDataset(Dataset):

    def __init__(self, root_dir, transform=None):

        self.root_dir = root_dir
        self.transform = transform
        images = []
        labels = []
        for idx,subdir in enumerate(os.listdir(self.root_dir)):
            subdir_path = os.path.join(self.root_dir,subdir)
            for img in os.listdir(subdir_path):
                if len(os.listdir(subdir_path)) >= 10:
                    img_path = os.path.join(subdir_path,img)
                    images.append(img_path)
                    labels.append(idx+1)

        self.images = images
        self.labels = labels

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img,label

    def __len__(self):
        return len(self.images)

def get_face_dataloader(root_dir,batch_size, num_workers):

    data_transforms = transforms.Compose([
            transforms.Resize(128),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

    face_dataset = FaceDataset(
        root_dir=root_dir,
        transform=data_transforms
    )

    dataset_size = len(face_dataset)
    indices = list(range(0,dataset_size))
    dataset_sampler =  SubsetRandomSampler(indices)
    dataloader = torch.utils.data.DataLoader(face_dataset, batch_size=batch_size, sampler = dataset_sampler , num_workers=num_workers,drop_last = True)
    return dataloader

def get_dataloader_triplets(root_dir,val_size,
                   test_size,num_triplets,
                   batch_size, num_workers):

    data_transforms = transforms.Compose([
            # transforms.ToPILImage(),
            # transforms.RandomRotation(15),
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.Resize(128),
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

class QuadTripletFaceDataset(Dataset):

    def __init__(self, root_dir_original , root_dir_masked , num_triplets, transform=None):

        self.root_dir_original = root_dir_original
        self.root_dir_masked = root_dir_masked
        self.num_triplets = num_triplets
        self.transform = transform
        self.face_classes_masked = self.make_dictionary_for_face_class(self.root_dir_masked,"masked")
        self.face_classes_original = self.make_dictionary_for_face_class(self.root_dir_original,"original")
        self.training_triplets_masked = self.generate_triplets( self.num_triplets,"masked")
        self.training_triplets_original = self.generate_triplets( self.num_triplets,"original")

    def make_dictionary_for_face_class(self,root_dir,type_of_image):

        '''
            - face_classes = {'class0': [class0_id0, ...], 'class1': [class1_id0, ...], ...}
        '''

        face_classes = dict()

        for label in os.listdir(root_dir):
           
            if label not in face_classes.keys() and  type_of_image == "masked":
           
                vals = list(os.listdir(os.path.join(root_dir,label)))
           
                if len(vals) >= 10:
                    face_classes[label] = vals
           
            elif type_of_image == "original" and label in self.face_classes_masked:
           
                vals = list(os.listdir(os.path.join(root_dir,label)))
           
                if len(vals) >= 10:
                    face_classes[label] = vals
        
        return face_classes


    def generate_triplets(self, num_triplets,type_of_image):


        triplets = []
        if type_of_image == "original":
            face_classes = self.face_classes_original
        elif type_of_image == "masked":
            face_classes = self.face_classes_masked
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

        anc_id_orig, pos_id_orig, neg_id_orig, pos_class_orig, neg_class_orig = self.training_triplets_original[idx]

        anc_img_orig = os.path.join(self.root_dir_original, str(pos_class_orig) ,str(anc_id_orig))
        pos_img_orig = os.path.join(self.root_dir_original, str(pos_class_orig), str(pos_id_orig))
        neg_img_orig = os.path.join(self.root_dir_original, str(neg_class_orig), str(neg_id_orig))

        anc_img_orig = Image.open(anc_img_orig).convert('RGB')
        pos_img_orig = Image.open(pos_img_orig).convert('RGB')
        neg_img_orig = Image.open(neg_img_orig).convert('RGB')

        anc_id_mask, pos_id_mask, neg_id_mask, pos_class_mask, neg_class_mask = self.training_triplets_masked[idx]

        anc_img_mask = os.path.join(self.root_dir_masked, str(pos_class_mask) ,str(anc_id_mask))
        pos_img_mask = os.path.join(self.root_dir_masked, str(pos_class_mask), str(pos_id_mask))
        neg_img_mask = os.path.join(self.root_dir_masked, str(neg_class_mask), str(neg_id_mask))

        anc_img_mask = Image.open(anc_img_mask).convert('RGB')
        pos_img_mask= Image.open(pos_img_mask).convert('RGB')
        neg_img_mask = Image.open(neg_img_mask).convert('RGB')


        sample_orig = {'anc_img_orig': anc_img_orig, 'pos_img_orig': pos_img_orig, 'neg_img_orig': neg_img_orig, 'pos_class_orig': pos_class_orig,
                  'neg_class_orig': neg_class_orig}

        sample_mask = {'anc_img_mask': anc_img_orig, 'pos_img_mask': pos_img_orig, 'neg_img_mask': neg_img_orig, 'pos_class_mask': pos_class_orig,
                  'neg_class_mask': neg_class_orig}

        sample = {**sample_orig,**sample_mask}
        
        if self.transform:
            sample['anc_img_orig'] = self.transform(sample['anc_img_orig'])
            sample['pos_img_orig'] = self.transform(sample['pos_img_orig'])
            sample['neg_img_orig'] = self.transform(sample['neg_img_orig'])
            sample['anc_img_mask'] = self.transform(sample['anc_img_mask'])
            sample['pos_img_mask'] = self.transform(sample['pos_img_mask'])
            sample['neg_img_mask'] = self.transform(sample['neg_img_mask'])

        return sample

    def __len__(self):
        return min( len(self.training_triplets_original) , len(self.training_triplets_masked))

def get_dataloader_quadtriplets(root_dir_original,root_dir_masked,
                    val_size,test_size,num_triplets,
                    batch_size, num_workers):

    data_transforms = transforms.Compose([
            # transforms.ToPILImage(),
            # transforms.RandomRotation(15),
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.Resize(128),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

    face_dataset = QuadTripletFaceDataset(
        root_dir_original=root_dir_original,
        root_dir_masked = root_dir_masked,
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
