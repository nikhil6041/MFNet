from torchvision import datasets,transforms 
from facenet_pytorch import MTCNN, training
import torch

def create_facial_cropped_dataset(data_dir,mtcnn,workers,batch_size):

    dataset = datasets.ImageFolder(data_dir, transform=transforms.Resize((256, 256)))
    dataset.samples = [
        (p, p.replace(data_dir, data_dir + '_cropped'))
            for p, _ in dataset.samples
    ]
            
    loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=workers,
        batch_size=batch_size,
        collate_fn=training.collate_pil
    )

    for i, (x, y) in enumerate(loader):
        mtcnn(x, save_path=y)
        print('\rBatch {} of {}'.format(i + 1, len(loader)), end='')
        
    # Remove mtcnn to reduce GPU memory usage
    del mtcnn

# create_facial_cropped_dataset(data_dir,mtcnn)