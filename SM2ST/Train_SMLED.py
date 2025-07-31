import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.sparse as sp
import os
from .SMLED import Encoder, Decoder, Discriminator_A,Discriminator_B
from .utils import Transfer_pytorch_Data, positional_pixel_step, recovery_coord, generation_coord, Cal_Spatial_Net
from .dataset import *
import random
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import os
import torch.nn.functional as F
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix

def sce_loss(x, y, alpha=1.0):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss

class WeightedMSELoss(torch.nn.Module):
    def __init__(self, weights):
        super(WeightedMSELoss, self).__init__()
        self.weights = weights

    def forward(self, y_pred, y_true):
        # 确保权重和输入张量的形状一致
        return torch.mean(self.weights * (y_pred -y_true) ** 2)

class WeightedMAELoss(torch.nn.Module):
    def __init__(self, weights):
        super(WeightedMAELoss, self).__init__()
        self.weights = weights

    def forward(self, y_pred, y_true):
        # 确保权重和输入张量的形状一致
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

from torch_sparse import SparseTensor
import scipy.sparse as sp

def edge_index_to_dense_adj(edge_index, num_nodes, device=None):
    """
    将 edge_index 转换为密集格式的邻接矩阵。
    :param edge_index: 形状为 [2, E] 的张量，表示边的索引。
    :param num_nodes: 节点的总数。
    :param device: 指定设备（CPU 或 GPU）。
    :return: 形状为 [N, N] 的密集邻接矩阵。
    """
    # 确保 edge_index 在指定设备上
    edge_index = edge_index.to(device)

    # 创建边的权重（全为 1 的张量），并确保它在相同的设备上
    edge_weight = torch.ones(edge_index.shape[1], device=device)

    # 创建稀疏张量
    sparse_adj = SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_weight, sparse_sizes=(num_nodes, num_nodes))

    # 转换为密集格式
    dense_adj = sparse_adj.to_dense()
    return dense_adj

def compute_adjacency_matrix(z):
    """
    计算邻接矩阵。
    :param z: 节点的嵌入表示，形状为 [N, D]。
    :return: 邻接矩阵，形状为 [N, N]。
    """
    # 计算内积矩阵
    inner_product_matrix = torch.mm(z, z.t())

    # 应用 Sigmoid 函数
    adjacency_matrix = F.sigmoid(inner_product_matrix)
    return adjacency_matrix

def train_SMLED(adata=None, X_dim = 2, delta = 1.0, train_epoch=15000,lr=0.001,mask_ratio=0.5,alpha=1.0,key_added='SMLED',step_size=10000,gamma=1.0,
                relu=True, gradient_clipping=5., experiment='generation', weight_decay=0.0001, verbose=True, batch_size = 1000,lambda_gp = 1.0,
                random_seed=2025, save_loss=False, save_reconstrction=False, save_path = './SMLED_pyG_result',down_ratio = 0., coord_sf=1.0, 
                WMMSE=0.0, res = 2.0, device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    """\
    Training graph attention auto-encoder.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    hidden_dims
        The dimension of the encoder.
    n_epochs
        Number of total epochs in training.
    lr
        Learning rate for AdamOptimizer.
    key_added
        The latent embeddings are saved in adata.obsm[key_added].
    gradient_clipping
        Gradient Clipping.
    weight_decay
        Weight decay for AdamOptimizer.
    save_loss
        If True, the training loss is saved in adata.uns['STMGraph_loss'].
    save_reconstrction
        If True, the reconstructed expression profiles are saved in adata.layers['STMGraph_ReX'].
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
    encoder, decoder = Encoder(gene_number, X_dim, down_ratio), Decoder(gene_number, X_dim, down_ratio=0.)
    discriminator_AB  = Discriminator_A(X_dim) #, Discriminator_B(gene_number) #, discriminator_BA
    # encoder.train()
    # decoder.train()

    encoder, decoder = encoder.to(device), decoder.to(device)
    discriminator_AB = discriminator_AB.to(device) # ,discriminator_BA.to(device) , discriminator_BA

    enc_optim = torch.optim.Adam(encoder.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-8, betas=(0.5, 0.999))#
    dec_optim = torch.optim.Adam(decoder.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-8, betas=(0.5, 0.999))

    disc_optim_AB = torch.optim.Adam(discriminator_AB.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-8, betas=(0.5, 0.999))
    # enc_optim_gan = torch.optim.Adam(encoder.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-8, betas=(0.5, 0.999)) #
    # dec_optim_gan = torch.optim.Adam(decoder.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-8, betas=(0.5, 0.999))

    n_gen = 1
    n_crit = 2
    # disc_optim_BA = torch.optim.Adam(discriminator_BA.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-8, betas=(0.5, 0.999))
    enc_sche = torch.optim.lr_scheduler.StepLR(enc_optim, step_size=n_gen*step_size, gamma=gamma)
    dec_sche = torch.optim.lr_scheduler.StepLR(dec_optim, step_size=n_gen*step_size, gamma=gamma)
    disc_sche_AB = torch.optim.lr_scheduler.StepLR(disc_optim_AB, step_size=n_crit*step_size, gamma=gamma)
    # enc_sche_gan = torch.optim.lr_scheduler.StepLR(enc_optim_gan, step_size=step_size, gamma=gamma)
    # dec_sche_gan = torch.optim.lr_scheduler.StepLR(dec_optim_gan, step_size=step_size, gamma=gamma)
    # loss function
    criterion = torch.nn.BCELoss()
    
    # loss function
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
    
    # criterion_GAN = torch.nn.MSELoss()
    # criterion_cycle = torch.nn.L1Loss()

    # encoder = torch.load(save_path + '/encoder.pth')
    # decoder = torch.load(save_path + '/decoder.pth')
    # discriminator_AB = torch.load(save_path + '/discriminator_AB.pth')
    # discriminator_BA = torch.load(save_path + '/discriminator_BA.pth')
    
    # with tqdm(range(train_epoch), total=train_epoch, desc='Epochs') as epoch:
    #     for j in epoch:
    #         train_Gloss = []
    #         train_D_Aloss = []
    #         train_D_Bloss = []
    #         for xdata, xlabel in train_loader:
    #             xdata = xdata.to(torch.float32)
    #             xlabel = xlabel.to(torch.float32)

    #             enc_optim.zero_grad()
    #             dec_optim.zero_grad()

    #             xdata, xlabel = Variable(xdata.to(device)), Variable(xlabel.to(device))
    #             fake_xlabel = encoder(xdata, relu)
    #             fake_xdata = decoder(xlabel, relu)152.055573563462
                
    #             # fake_xlabel = latent.view(-1, X_dim)
    #             # xlabel = xlabel.view(-1, X_dim)
    #             # xlabel = xlabel.float().to(device)

    #             loss_GAN_A = criterion_GAN(discriminator_BA(fake_xlabel), torch.ones_like(discriminator_BA(fake_xlabel)))
    #             loss_GAN_B = criterion_GAN(discriminator_AB(fake_xdata), torch.ones_like(discriminator_AB(fake_xdata)))
    #             # loss_idt_A = criterion_cycle(encoder(decoder(fake_xlabel, relu), relu), xlabel)
    #             # loss_idt_B = criterion_cycle(decoder(encoder(fake_xdata, relu), relu), xdata)
    #             loss_cycle_A = criterion_cycle(encoder(fake_xdata, relu), xlabel)
    #             loss_cycle_B = criterion_cycle(decoder(fake_xlabel, relu), xdata)
    #             loss_G = loss_GAN_A + loss_GAN_B + 10*(loss_cycle_A + loss_cycle_B) #+ 5*(loss_idt_A + loss_idt_B)
    #             loss_G.backward()
    #             enc_optim.step()
    #             dec_optim.step()

    #             for _ in range(2):
    #                 # train label discriminator_BA
    #                 print(xlabel)
    #                 print(fake_xlabel.detach())
    #                 disc_optim_BA.zero_grad()
    #                 loss_real_A = criterion_GAN(discriminator_BA(xlabel), torch.ones_like(discriminator_BA(xlabel)))
    #                 loss_fake_A = criterion_GAN(discriminator_BA(fake_xlabel.detach()), torch.zeros_like(discriminator_BA(fake_xlabel)))
    #                 loss_D_A = (loss_real_A + loss_fake_A) / 2
    #                 loss_D_A.backward()
    #                 disc_optim_BA.step()
    
    #                 # train data discriminator_AB
    #                 disc_optim_AB.zero_grad()
    #                 loss_real_B = criterion_GAN(discriminator_AB(xdata), torch.ones_like(discriminator_AB(xdata)))
    #                 loss_fake_B = criterion_GAN(discriminator_AB(fake_xdata.detach()), torch.zeros_like(discriminator_AB(fake_xdata)))
    #                 loss_D_B = (loss_real_B + loss_fake_B) / 2
    #                 loss_D_B.backward()
    #                 disc_optim_AB.step()

    #             train_Gloss.append(loss_G.item())
    #             train_D_Aloss.append(loss_D_A.item())
    #             train_D_Bloss.append(loss_D_B.item())

    #         epoch_info = 'loss_G: %.5f, loss_D_A: %.5f, loss_D_B: %.5f' % \
    #                      (torch.mean(torch.FloatTensor(train_Gloss)),
    #                       torch.mean(torch.FloatTensor(train_D_Aloss)),
    #                       torch.mean(torch.FloatTensor(train_D_Bloss)))152.055573563462
    #         epoch.set_postfix_str(epoch_info)
    with tqdm(range(train_epoch), total=train_epoch, desc='Epochs') as epoch:
        for j in epoch:
            train_reloss = []
            train_GAloss = []
            train_latloss = []
            train_loss = []
            train_DAloss = []
            # train_DBloss = []
    
            for xdata, xlabel in train_loader:
                xdata = xdata.to(torch.float32)
                xlabel = xlabel.to(torch.float32)
                xdata, xlabel = Variable(xdata.to(device)), Variable(xlabel.to(device))
                
                # for _ in range(1):#
                #     encoder.train()
                #     decoder.train()
                #     enc_optim.zero_grad()
                #     dec_optim.zero_grad()
                #     fake_xlabel = encoder(xdata, relu)
                #     fake_xdata = decoder(fake_xlabel, relu)
                #     latent_loss = loss1(fake_xlabel, xlabel)
                #     recon_loss = loss2(fake_xdata, xdata) + 0.1 * loss1(fake_xdata, xdata)
                #     loss = recon_loss # + latent_loss
                #     train_latloss.append(latent_loss.item())
                #     train_reloss.append(recon_loss.item())
                #     train_loss.append(loss.item())
                #     loss.backward()
                #     enc_optim.step()
                #     dec_optim.step()
                #     enc_sche.step()
                #     dec_sche.step()
                # encoder.eval()
                # decoder.eval()
                
                for _ in range(n_crit): #3
                    discriminator_AB.train()
                    disc_optim_AB.zero_grad()
                    fake_xlabel = encoder(xdata, relu)
                    # fake_xdata = decoder(fake_xlabel, relu)
                    # fake_xdata = decoder(fake_xlabel, relu)
                    # combined_xlabel = torch.cat((xdata, xlabel), dim=1)
                    # combined_fake_xlabel = torch.cat((fake_xdata, fake_xlabel), dim=1)
                    # disc_realA = discriminator_AB(combined_xlabel)
                    # disc_fakeA = discriminator_AB(combined_fake_xlabel)
                    disc_realA = discriminator_AB(xlabel)
                    disc_fakeA = discriminator_AB(fake_xlabel)
                    # d_loss = wasserstein_loss(disc_realA, disc_fakeA)
                    # gp = gradient_penalty(discriminator_AB, xlabel, fake_xlabel, device, lambda_gp = lambda_gp)
                    # d_total_loss = d_loss + gp
                    disc_real = disc_realA.view(-1)
                    disc_fake = disc_fakeA.view(-1)
                    loss_dis_real = criterion(disc_real, torch.ones_like(disc_real))
                    loss_dis_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
                    d_total_loss = loss_dis_real + loss_dis_fake
                    train_DAloss.append(d_total_loss.item())
                    d_total_loss.backward()
                    # torch.nn.utils.clip_grad_norm_(discriminator_AB.parameters(), gradient_clipping)
                    disc_optim_AB.step()
                    disc_sche_AB.step()
                    # discriminator_BA.train()
                    # disc_optim_BA.zero_grad()
                    # fake_xlabel = encoder(xdata, relu)
                    # fake_xdata = decoder(fake_xlabel, relu)
                    # disc_real = discriminator_BA(xdata)
                    # disc_fake = discriminator_BA(fake_xdata)
                    # d_loss = wasserstein_loss(disc_real, disc_fake)
                    # gp = gradient_penalty(discriminator_BA, xdata, fake_xdata, device, lambda_gp = lambda_gp)
                    # d_total_loss = d_loss + gp
                    # train_DBloss.append(d_total_loss.item())
                    # d_total_loss.backward()
                    # torch.nn.utils.clip_grad_norm_(discriminator_BA.parameters(), gradient_clipping)
                    # disc_optim_BA.step()
                discriminator_AB.eval()
                # discriminator_BA.eval()
                
                for _ in range(n_gen):#
                    encoder.train()
                    decoder.train()
                    enc_optim.zero_grad()
                    dec_optim.zero_grad()
                    fake_xlabel = encoder(xdata, relu)
                    fake_xdata = decoder(fake_xlabel, relu)
                    # fake_xdata_ = decoder(xlabel, relu)
                    # disc_fakeA = discriminator_AB(fake_xlabel)
                    # disc_fake = disc_fakeA.view(-1)
                    # gA_loss = criterion(disc_fake, torch.ones_like(disc_fake))
                    # combined_xlabel = torch.cat((xdata, xlabel), dim=1)
                    # combined_fake_xlabel = torch.cat((fake_xdata, fake_xlabel), dim=1)
                    # disc_realA = discriminator_AB(combined_xlabel)
                    # disc_fakeA = discriminator_AB(combined_fake_xlabel)
                    disc_realA = discriminator_AB(xlabel)
                    disc_fakeA = discriminator_AB(fake_xlabel)
                    # gA_loss = -wasserstein_loss(disc_realA, disc_fakeA)
                    gA_loss = torch.abs(wasserstein_loss(disc_realA, disc_fakeA))
                    # gA_loss = torch.abs(wasserstein_loss(disc_realA, disc_fakeA))
                    # gp = gradient_penalty(discriminator_AB, xlabel, fake_xlabel, device, lambda_gp = lambda_gp)
                    # d_total_loss = gA_loss + gp
                    # disc_fakeB = discriminator_BA(fake_xdata)
                    # gA_loss = -disc_fakeA.mean()
                    # gB_loss = -disc_fakeB.mean()
                    
                    latent_loss = MAE(fake_xlabel, xlabel)
                    # + 0.1 * sliced_wasserstein_distance(fake_xlabel, xlabel, 1000, device=device)
                    recon_loss = loss2(fake_xdata, xdata) + 0.1*loss1(fake_xdata, xdata)
                    
                    loss = recon_loss + 0.3*latent_loss + gA_loss  #
                    # loss = 0.4*recon_loss + 0.6*latent_loss + gA_loss # last best
                    train_latloss.append(latent_loss.item())
                    train_GAloss.append(gA_loss.item())
                    # train_GBloss.append(gB_loss.item())
                    train_reloss.append(recon_loss.item())
                    # train_Gloss.append(g_loss.item())
                    train_loss.append(loss.item())
                    loss.backward()
                    # gA_loss.backward()
                    # torch.nn.utils.clip_grad_norm_(encoder.parameters(), gradient_clipping)
                    # torch.nn.utils.clip_grad_norm_(decoder.parameters(), gradient_clipping)
                    enc_optim.step()
                    dec_optim.step()
                    enc_sche.step()
                    dec_sche.step()
                encoder.eval()
                decoder.eval()
                
            #, loss_GB: %.5f , loss_DB: %.5f 
            epoch_info = 'loss_re: %.5f, loss_lat: %.5f, loss_GA: %.5f, loss: %.5f, loss_DA: %.5f' % \
                         (torch.mean(torch.FloatTensor(train_reloss)),
                          torch.mean(torch.FloatTensor(train_latloss)),
                          torch.mean(torch.FloatTensor(train_GAloss)),
                          torch.mean(torch.FloatTensor(train_loss)),
                          torch.mean(torch.FloatTensor(train_DAloss))
                          # torch.mean(torch.FloatTensor(train_DBloss))
                         )#
            epoch.set_postfix_str(epoch_info)
                

    torch.save(encoder, save_path+'/encoder.pth')
    torch.save(decoder, save_path+'/decoder.pth')

    # torch.save(discriminator_AB, save_path+'/discriminator_AB.pth')
    # torch.save(discriminator_BA, save_path+'/discriminator_BA.pth')
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