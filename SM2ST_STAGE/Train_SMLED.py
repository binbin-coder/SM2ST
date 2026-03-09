import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.sparse as sp
import os
from .SMLED import Encoder, Decoder
from .utils import Transfer_pytorch_Data, positional_pixel_step, recovery_coord, generation_coord, Cal_Spatial_Net
from .dataset import *
import random
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import os
import torch.nn.functional as F
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix
from torch_sparse import SparseTensor

import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn


loss1 = nn.L1Loss()
loss2 = nn.MSELoss()


def rand_projections(
        embedding_dim,
        num_samples=50,
        device='cpu'
):
    """This function generates `num_samples` random samples from the latent space's unit sphere.

        Args:
            embedding_dim (int): embedding dimensionality
            num_samples (int): number of random projection samples

        Return:
            torch.Tensor: tensor of size (num_samples, embedding_dim)
    """
    projections = [w / np.sqrt((w**2).sum())  # L2 normalization
                   for w in np.random.normal(size=(num_samples, embedding_dim))]
    projections = np.asarray(projections)
    return torch.from_numpy(projections).type(torch.FloatTensor).to(device)


def _sliced_wasserstein_distance(
        encoded_samples,
        distribution_samples,
        num_projections=50,
        p=2,
        device='cpu'
):
    """ Sliced Wasserstein Distance between encoded samples and drawn distribution samples.

        Args:
            encoded_samples (toch.Tensor): tensor of encoded training samples
            distribution_samples (torch.Tensor): tensor of drawn distribution training samples
            num_projections (int): number of projections to approximate sliced wasserstein distance
            p (int): power of distance metric
            device (torch.device): torch device (default 'cpu')

        Return:
            torch.Tensor: tensor of wasserstrain distances of size (num_projections, 1)
    """
    # derive latent space dimension size from random samples drawn from latent prior distribution
    embedding_dim = distribution_samples.size(1)
    # generate random projections in latent space
    projections = rand_projections(embedding_dim, num_projections).to(device)
    # calculate projections through the encoded samples
    encoded_projections = encoded_samples.matmul(projections.transpose(0, 1).to(device))
    # calculate projections through the prior distribution random samples
    distribution_projections = (distribution_samples.matmul(projections.transpose(0, 1)))
    # calculate the sliced wasserstein distance by
    # sorting the samples per random projection and
    # calculating the difference between the
    # encoded samples and drawn random samples
    # per random projection
    wasserstein_distance = (torch.sort(encoded_projections.transpose(0, 1), dim=1)[0] -
                            torch.sort(distribution_projections.transpose(0, 1), dim=1)[0])
    # distance between latent space prior and encoded distributions
    # power of 2 by default for Wasserstein-2
    wasserstein_distance = torch.pow(wasserstein_distance, p)
    # approximate mean wasserstein_distance for each projection
    return wasserstein_distance.mean()


def sliced_wasserstein_distance(
        encoded_samples,
        transformed_samples,
        num_projections=50,
        p=2,
        device='cpu'
):
    """ Sliced Wasserstein Distance between encoded samples and drawn distribution samples.

        Args:
            encoded_samples (toch.Tensor): tensor of encoded training samples
            distribution_samples (torch.Tensor): tensor of drawn distribution training samples
            num_projections (int): number of projections to approximate sliced wasserstein distance
            p (int): power of distance metric
            device (torch.device): torch device (default 'cpu')

        Return:
            torch.Tensor: tensor of wasserstrain distances of size (num_projections, 1)
    """
    # derive batch size from encoded samples
    # draw random samples from latent space prior distribution

    # approximate mean wasserstein_distance between encoded and prior distributions
    # for each random projection
    swd = _sliced_wasserstein_distance(encoded_samples, transformed_samples, num_projections, p, device)
    return swd



class WeightedMSELoss(torch.nn.Module):
    def __init__(self, weights):
        super(WeightedMSELoss, self).__init__()
        self.weights = weights

    def forward(self, y_pred, y_true):
        # Ensure that the shape of the weights is consistent with that of the input tensor
        return torch.mean(self.weights * (y_pred -y_true) ** 2)

class WeightedMAELoss(torch.nn.Module):
    def __init__(self, weights):
        super(WeightedMAELoss, self).__init__()
        self.weights = weights

    def forward(self, y_pred, y_true):
        # Ensure that the shape of the weights is consistent with that of the input tensor
        return torch.mean(self.weights * torch.abs(y_pred - y_true))


def rand_projections(
        embedding_dim,
        num_samples=50,
        device='cpu'
):
    """This function generates `num_samples` random samples from the latent space's unit sphere.

        Args:
            embedding_dim (int): embedding dimensionality
            num_samples (int): number of random projection samples

        Return:
            torch.Tensor: tensor of size (num_samples, embedding_dim)
    """
    projections = [w / np.sqrt((w**2).sum())  # L2 normalization
                   for w in np.random.normal(size=(num_samples, embedding_dim))]
    projections = np.asarray(projections)
    return torch.from_numpy(projections).type(torch.FloatTensor).to(device)


def wasserstein_loss(disc_real, disc_fake):
    return -torch.mean(disc_real) + torch.mean(disc_fake)

def gradient_penalty(discriminator, real_data, fake_data, device, lambda_gp=10):
    alpha = torch.rand(real_data.size(0), 1).to(device)
    interpolated = alpha * real_data + ((1 - alpha) * fake_data)
    interpolated = interpolated.requires_grad_(True)
    mixed_scores = discriminator(interpolated)
    gradients = torch.autograd.grad(
        inputs=interpolated,
        outputs=mixed_scores,
        grad_outputs=torch.ones(mixed_scores.size()).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients_norm = torch.norm(gradients.view(gradients.size(0), -1), dim=1)
    gradient_penalty = lambda_gp * ((gradients_norm - 1) ** 2).mean()
    return gradient_penalty


def train_SMLED(adata=None, X_dim = 2, delta = 1.0, train_epoch=1000,lr=0.001,mask_ratio=0.5,alpha=1.0,key_added='SMLED',step_size=10000,gamma=1.0,
                relu=True, gradient_clipping=5., experiment='generation', weight_decay=0.0001, verbose=True, batch_size = 1000,lambda_gp = 1.0,
                random_seed=2025, save_path = './SMLED_pyG_result',down_ratio = 0., coord_sf=1.0, 
                WMMSE=0.0, res = 2.0, device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    """\
    Training GAN auto-encoder.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    delta
        Coordinate scaling.
    train_epoch
        Number of total epochs in training.
    lr
        Learning rate for AdamOptimizer.
    key_added
        The latent embeddings are saved in adata.obsm[key_added].
    gradient_clipping
        Gradient Clipping.
    weight_decay
        Weight decay for AdamOptimizer.
    mask_ratio
        Random masking ratio.
    WMMSE
        The weight distribution of wmse.
    device
        See torch.device.

    Returns
    -------
    AnnData
    """

    # seed_everything()
    seed=random_seed
    fix_seed(seed)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    if verbose:
        print('Size of Input: ', adata.X.shape)

    if experiment=='recovery':
        # adata, masked_adata, adata_filtered, picked_index, remaining_index = masked_anndata(adata = adata, mask_ratio=0.5)
        coor, full_coor, sample_index, sample_barcode = recovery_coord(adata,name='spatial',mask_ratio = mask_ratio)
        used_gene, normed_data, adata_sample = get_data(adata, experiment=experiment, sample_index=sample_index, sample_barcode=sample_barcode)
        xlabel_df,full_xlabel_df = positional_pixel_step(coor, full_coor, delta, coord_sf)
        print(xlabel_df,full_xlabel_df)
        transformed_dataset = MyDataset(normed_data=normed_data, coor_df=xlabel_df, transform=transforms.Compose([ToTensor()]))
        train_loader = DataLoader(transformed_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False)
        
    elif experiment == 'higher_res':
        coor, full_coor = generation_coord(adata,name='spatial',res=res)
        used_gene, normed_data = get_data(adata, experiment=experiment)
        xlabel_df,full_xlabel_df = positional_pixel_step(coor, full_coor, delta, coord_sf)
        print(xlabel_df,full_xlabel_df)
        transformed_dataset = MyDataset(normed_data=normed_data, coor_df = xlabel_df, transform=transforms.Compose([ToTensor()]))
        train_loader = DataLoader(transformed_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False)

    elif experiment == 'generation':
        coor = adata.obsm['spatial']
        full_coor = adata.uns['coord']
        used_gene, normed_data = get_data(adata, experiment=experiment)
        xlabel_df,full_xlabel_df = positional_pixel_step(coor, full_coor, delta, coord_sf)
        print(xlabel_df,full_xlabel_df)
        transformed_dataset = MyDataset(normed_data=normed_data, coor_df=xlabel_df, transform=transforms.Compose([ToTensor()]))
        train_loader = DataLoader(transformed_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False)
    
    gene_number = len(used_gene)
    encoder, decoder = Encoder(gene_number, X_dim), Decoder(gene_number, X_dim)
    
    if WMMSE>0:
        if sp.issparse(adata.X):
            matrix = adata.X.A
        else:
            matrix = adata.X
        column_sums = matrix.sum(axis=0)
        normalized = column_sums * (WMMSE / column_sums.sum())
        weights = WMMSE - normalized
        
        weights = torch.tensor(weights, dtype=torch.float32,device = device)
        loss2 = WeightedMSELoss(weights)
        loss1 = WeightedMAELoss(weights)
    else:
        loss2 = torch.nn.MSELoss()
        loss1 = torch.nn.L1Loss()
    MAE = torch.nn.L1Loss()
    w_recon=0.1
    w_w=0.1
    w_l1=0.1
    encoder.train()
    decoder.train()

    encoder, decoder = encoder.to(device), decoder.to(device)

    enc_optim = optim.Adam(encoder.parameters(), lr=lr)
    dec_optim = optim.Adam(decoder.parameters(), lr=lr)

    enc_sche = optim.lr_scheduler.StepLR(enc_optim, step_size=step_size, gamma=gamma)
    dec_sche = optim.lr_scheduler.StepLR(dec_optim, step_size=step_size, gamma=gamma)

    with tqdm(range(train_epoch), total=train_epoch, desc='Epochs', ncols=0) as epoch:
        for j in epoch:

            train_loss = []
            train_lc_loss = []
            train_re_loss = []

            for xdata, xlabel in train_loader:
                xdata = xdata.to(torch.float32)
                xlabel = xlabel.to(torch.float32)

                enc_optim.zero_grad()
                dec_optim.zero_grad()

                xdata, xlabel, = Variable(xdata.to(device)), Variable(xlabel.to(device))

                latent = encoder(xdata, relu)
                latent = latent.view(-1, X_dim)
                xlabel = xlabel.float().to(device)
                latent_loss = MAE(latent, xlabel) + w_w * sliced_wasserstein_distance(latent, xlabel, 1000, device=device)
                xrecon = decoder(latent, relu)
                recon_loss = loss2(xrecon, xdata) + w_l1 * loss1(xrecon, xdata)

                total_loss = 0.1 * latent_loss + 0.1 * w_recon * recon_loss

                total_loss.backward()

                enc_optim.step()
                dec_optim.step()

                enc_sche.step()
                dec_sche.step()

                train_lc_loss.append(latent_loss.item())
                train_re_loss.append(recon_loss.item())
                train_loss.append(total_loss.item())

            epoch_info = 'latent_loss: %.5f, recon_loss: %.5f, total_loss: %.5f' % \
                         (torch.mean(torch.FloatTensor(train_lc_loss)),
                          torch.mean(torch.FloatTensor(train_re_loss)),
                          torch.mean(torch.FloatTensor(train_loss)))
            epoch.set_postfix_str(epoch_info)
                

    torch.save(encoder, save_path+'/encoder.pth')
    torch.save(decoder, save_path+'/decoder.pth')

    encoder.eval()
    decoder.eval()
    # Get generated or recovered data
    if experiment=='generation' or experiment=='recovery' or experiment=='higher_res':
        full_coor_df = full_xlabel_df.copy()
        full_coor_t = torch.from_numpy(np.array(full_coor_df))
        full_coor_t = full_coor_t.to(torch.float32)
        full_coor_t = Variable(full_coor_t.to(device))
        # if experiment=='higher_res':
        dataloader_t = DataLoader(full_coor_t, batch_size=1000, shuffle=False)
        generate_profile_list = []
        for batch_coor_t in dataloader_t:
            batch_coor_t = batch_coor_t.to(torch.float32)
            batch_coor_t = Variable(batch_coor_t.to(device))
            batch_generate_profile = decoder(batch_coor_t, relu)
            batch_generate_profile = batch_generate_profile.cpu().detach().numpy()
            generate_profile_list.append(batch_generate_profile)
            generate_profile = np.concatenate(generate_profile_list, axis=0)
        # else:
        #     generate_profile = decoder(full_coor_t, relu)
        #     generate_profile = generate_profile.cpu().detach().numpy()
        if not relu:
            generate_profile = np.clip(generate_profile, a_min=0, a_max=None)

        if experiment=='recovery':
            np.savetxt(save_path+"/fill_data.txt", generate_profile)
            
        st_intensity = csr_matrix(generate_profile, dtype=np.float32)
        adata_SMLED = sc.AnnData(st_intensity)
        # adata_SMLED = sc.AnnData(generate_profile)
        adata_SMLED.obsm["spatial"] = full_coor
        adata_SMLED.var.index = used_gene

        adata.write(save_path + '/original_data.h5ad')

        if experiment=='generation' or experiment=='higher_res':
            adata_SMLED.write(save_path + '/generated_data.h5ad')
            return adata_SMLED
        elif experiment=='recovery':
            adata_sample.write(save_path + '/sampled_data.h5ad')
            adata_SMLED.obs = adata.obs
            adata_SMLED.write(save_path + '/recovered_data.h5ad')
            return adata_sample, adata_SMLED


def fix_seed(seed):
    #seed = 2025
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  