import pandas as pd
import numpy as np
import torch
import scanpy as sc
import scipy.sparse as sp
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import os

def get_data(
        adata,
        experiment='generation',
        sample_index=None,
        sample_barcode=None
):
    """ Get training data used to generation from original AnnData object

        Args:
            adata: AnnData object storing original data. Raw data should to be normalized. Highly variable genes should be identified.
            experiment: Different tasks. Available options are: "generation", "recovery". Default is "generation".
            sample_index: Index of downsampled data. Available when experiment = "recovery".
            sample_barcode: Barcode of downsampled data. Available when experiment = "recovery".

        Return:
            used_ion: Highly variable ions used to generation from original AnnData object
            normed_data: Normalized data extracted from original AnnData object.
            adata_sample: Down-sampled AnnData object. Available when experiment = "recovery".
    """
    if 'highly_variable' in adata.var.columns:
        used_ion = np.array(adata.var.index[adata.var.highly_variable])
    else:
        used_ion = np.array(adata.var.index)

    if experiment=='generation' or experiment=='higher_res':
        if 'highly_variable' in adata.var.columns:
            normed_data = sp.coo_matrix(adata.X[:, adata.var.highly_variable].T).todense()
        else:
            normed_data = sp.coo_matrix(adata.X.T).todense()
        normed_data = pd.DataFrame(normed_data)
        return used_ion, normed_data
    elif experiment=='recovery':
        adata_sample = adata[sample_barcode]
        if 'highly_variable' in adata.var.columns:
            normed_data = sp.coo_matrix(adata.X[sample_index][:, adata.var.highly_variable].T).todense()
        else:
            normed_data = sp.coo_matrix(adata.X[sample_index].T).todense()
        # normed_data_all = sp.coo_matrix(adata.X[:, adata.var.highly_variable].T).todense()
        normed_data = pd.DataFrame(normed_data)
        
        return used_ion, normed_data, adata_sample

class MyDataset(Dataset):
    """Operations with the datasets."""

    def __init__(self, normed_data, coor_df, transform=None):
        """
        Args:
            normed_data: Normalized data extracted from original AnnData object.
            coor_df: Spatial location extracted from original AnnData object.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = normed_data
        self.label = coor_df
        self.transform = transform

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, idx):
        data = np.array(self.data.iloc[:, idx])
        label = np.array(self.label.iloc[idx,])
        sample = (data, label)
        if self.transform:
            sample = self.transform(sample)
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        data, label = sample[0], sample[1]

        return (torch.from_numpy(data), torch.from_numpy(label))

