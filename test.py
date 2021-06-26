from dataset import get_dataloader
import time
from pprint import pprint as ppt
start_time = time.time()
dataloaders , data_size = get_dataloader(
    root_dir="D:\Academics\HonsProject1\Labelled Faces In The Wild Dataset\lfw-deepfunneled\lfw-deepfunneled",
    val_size=0.2,
    test_size=0.2,
    num_triplets=10000,
    batch_size=32,
    num_workers=0
)

ppt(time.time() - start_time)
ppt(dataloaders)

for batch in dataloaders['train']:

    for k  in batch.keys():
        ppt(k)
        if not isinstance(batch[k],list):
            ppt(batch[k].size())
        else:
            # ppt(batch[k])

ppt(data_size)