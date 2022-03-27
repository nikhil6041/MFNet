from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import argparse
import torch 
import yaml
import pandas as pd
from model import FaceNetModel
from dataset import get_face_dataloader



if __name__ == "__main__":
    config_file_path = "config.yaml"

    try:
        ## loading the configuration
        with open(config_file_path, "r") as stream:
            config = yaml.safe_load(stream)
            
    except:
        print("Could not find the config file,please pass the path correctly")
        exit()
    # parser = argparse.ArgumentParser(description='Masked Face Recognition')
    # parser.add_argument('--variant',type=str,help="Name of variant a) triplet b) quad triplet")
    # args = vars(parser.parse_args())

    variant = 'triplet_model'
    config = config[variant]

    ckpt_path = config['dataset']['save_dir'] + config['model']['best_ckpt_name']
    dataset_dir = config['dataset']['root_dir']
    batch_size = config['dataset']['batch_size']
    num_workers = config['dataset']['num_workers']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = FaceNetModel('casia-webface').to(device)
    state_dict = torch.load(ckpt_path)['state_dict']
    model.load_state_dict(state_dict)

    embeddings = []
    labels = []
    dataset = get_face_dataloader(dataset_dir,batch_size,num_workers)

    for idx,(batch_img,batch_label) in enumerate(dataset):

        embs = model(batch_img.to(device))

        embeddings.extend(embs.detach().cpu().numpy())

        labels.extend(batch_label)

    df = pd.DataFrame({
        'embeddings':embeddings,
        'labels':labels
    })

    pca = PCA(n_components=3)
    pca_res = pca.fit_transform(embeddings)
    df['pca-one'] = pca_res[:,0]
    df['pca-two'] = pca_res[:,1] 
    df['pca-three'] = pca_res[:,2]
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

    plt.figure(figsize=(16,10))

    sns.scatterplot(
        x="pca-one", y="pca-two",
        hue = "labels",
        palette=sns.color_palette("hls", 10),
        data=df,
        legend="full",
        alpha=0.3   
    )

    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(embeddings)

    df['tsne-2d-one'] = tsne_results[:,0]
    df['tsne-2d-two'] = tsne_results[:,1]
    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue = "labels",
        palette=sns.color_palette("hls", 10),
        data=df,
        legend="full",
        alpha=0.3
    )

    plt.figure(figsize=(16,7))
    ax1 = plt.subplot(1, 2, 1)
    sns.scatterplot(
        x="pca-one", y="pca-two",
        hue = "labels",
        palette=sns.color_palette("hls", 10),
        data=df,
        legend="full",
        alpha=0.3,
        ax=ax1
    )
    ax2 = plt.subplot(1, 2, 2)
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue = "labels",
        palette=sns.color_palette("hls", 10),
        data=df,
        legend="full",
        alpha=0.3,
        ax=ax2
    )
