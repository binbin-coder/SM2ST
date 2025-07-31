import pandas as pd
import numpy as np
import sklearn.neighbors
import scipy.sparse as sp
import seaborn as sns
import matplotlib.pyplot as plt
import random
import torch
from torch_geometric.data import Data

def Transfer_pytorch_Data(adata):
    G_df = adata.uns['Spatial_Net'].copy()
    cells = np.array(adata.obs_names)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)

    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    G = G + sp.eye(G.shape[0])

    edgeList = np.nonzero(G)
    if type(adata.X) == np.ndarray:
        data = Data(edge_index=torch.LongTensor(np.array(
            [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X))  # .todense()
    else:
        data = Data(edge_index=torch.LongTensor(np.array(
            [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X.todense()))  # .todense()
    return data

def Batch_Data(adata, num_batch_x, num_batch_y, spatial_key=['X', 'Y'], plot_Stats=False):
    Sp_df = adata.obs.loc[:, spatial_key].copy()
    Sp_df = np.array(Sp_df)
    batch_x_coor = [np.percentile(Sp_df[:, 0], (1/num_batch_x)*x*100) for x in range(num_batch_x+1)]
    batch_y_coor = [np.percentile(Sp_df[:, 1], (1/num_batch_y)*x*100) for x in range(num_batch_y+1)]

    Batch_list = []
    for it_x in range(num_batch_x):
        for it_y in range(num_batch_y):
            min_x = batch_x_coor[it_x]
            max_x = batch_x_coor[it_x+1]
            min_y = batch_y_coor[it_y]
            max_y = batch_y_coor[it_y+1]
            temp_adata = adata.copy()
            temp_adata = temp_adata[temp_adata.obs[spatial_key[0]].map(lambda x: min_x <= x <= max_x)]
            temp_adata = temp_adata[temp_adata.obs[spatial_key[1]].map(lambda y: min_y <= y <= max_y)]
            Batch_list.append(temp_adata)
    if plot_Stats:
        f, ax = plt.subplots(figsize=(1, 3))
        plot_df = pd.DataFrame([x.shape[0] for x in Batch_list], columns=['#spot/batch'])
        sns.boxplot(y='#spot/batch', data=plot_df, ax=ax)
        sns.stripplot(y='#spot/batch', data=plot_df, ax=ax, color='red', size=5)
    return Batch_list

def Cal_Spatial_Net(adata, rad_cutoff=None, k_cutoff=None, model='Radius', verbose=True,delta_err=1):
    """\
    Construct the spatial neighbor networks.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    rad_cutoff
        radius cutoff when model='Radius'
    k_cutoff
        The number of nearest neighbors when model='KNN'
    model
        The network construction model. When model=='Radius', the spot is connected to spots whose distance is less than rad_cutoff. When model=='KNN', the spot is connected to its first k_cutoff nearest neighbors.
    
    Returns
    -------
    The spatial networks are saved in adata.uns['Spatial_Net']
    """

    assert(model in ['Radius', 'KNN'])
    if verbose:
        print('------Calculating spatial graph...')
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']

    if model == 'Radius':
        nbrs = sklearn.neighbors.NearestNeighbors(radius=rad_cutoff).fit(coor)
        distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it]*indices[it].shape[0], indices[it], distances[it])))
    
    if model == 'KNN':
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k_cutoff+1).fit(coor)
        distances, indices = nbrs.kneighbors(coor)
        KNN_list = []
        distance_threshold=np.sort(distances[:,-1])[0]
        distance_threshold = distance_threshold+delta_err
        for it in range(indices.shape[0]):
            close_indices = indices[it, distances[it, :] <= distance_threshold]
            close_distances = distances[it, distances[it, :] <= distance_threshold]
            KNN_list.append(pd.DataFrame(zip([it]*len(close_indices), close_indices, close_distances)))
            # KNN_list.append(pd.DataFrame(zip([it]*indices.shape[1],indices[it,:], distances[it,:])))

    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

    Spatial_Net = KNN_df.copy()
    Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance']>0,]
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
    if verbose:
        print('The graph contains %d edges, %d cells.' %(Spatial_Net.shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' %(Spatial_Net.shape[0]/adata.n_obs))

    adata.uns['Spatial_Net'] = Spatial_Net


def Stats_Spatial_Net(adata):
    import matplotlib.pyplot as plt
    Num_edge = adata.uns['Spatial_Net']['Cell1'].shape[0]
    Mean_edge = Num_edge/adata.shape[0]
    plot_df = pd.value_counts(pd.value_counts(adata.uns['Spatial_Net']['Cell1']))
    plot_df = plot_df/adata.shape[0]
    fig, ax = plt.subplots(figsize=[3,2])
    plt.ylabel('Percentage')
    plt.xlabel('')
    plt.title('Number of Neighbors (Mean=%.2f)'%Mean_edge)
    ax.bar(plot_df.index, plot_df)

def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='STMGraph', random_seed=52, dist=None):
    """
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    refined_pred=refine(adata)
    adata.obs["refine_mclust"]=refined_pred
    adata.obs["refine_mclust"]=adata.obs["refine_mclust"].astype('category')
    return adata

def refine(adata=None):
    refined_pred=[]
    dis_df=adata.uns['Spatial_Net'].reset_index(drop=True)
    sample_id=adata.obs.index.tolist()
    pred=adata.obs['mclust'].tolist()
    pred=pd.DataFrame({"pred": pred}, index=sample_id)
    for index in sample_id:
        num_index=dis_df[dis_df.loc[:,'Cell1']==index].index
        num_nbs=len(num_index)
        self_pred=pred.loc[index, "pred"]
        if num_nbs>0:
            dis_tmp=dis_df.loc[num_index,:]
            nbs_pred=pred.loc[dis_tmp.loc[:,'Cell2'].to_list(), "pred"]
           
            v_c=nbs_pred.value_counts()
            if self_pred in v_c.index:
                if (v_c.loc[self_pred]<num_nbs/2) and (np.max(v_c)>num_nbs/2):
                    refined_pred.append(v_c.idxmax())
                else:           
                    refined_pred.append(self_pred)
            else:
                refined_pred.append(v_c.idxmax())
        else:
            refined_pred.append(self_pred)
    return refined_pred

import math
def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def closest_pair(px, py):
    n = len(px)
    if n <= 3:
        min_dist = float('inf')
        for i in range(n):
            for j in range(i + 1, n):
                min_dist = min(min_dist, distance(px[i], px[j]))
        return min_dist
    
    # Divide
    mid = n // 2
    midpoint = px[mid]
    pl = px[:mid]
    pr = px[mid:]
    
    # Create pyl and pyr
    pyl = []
    pyr = []
    for point in py:
        if point[0] <= midpoint[0]:
            pyl.append(point)
        else:
            pyr.append(point)
    
    # Conquer
    delta_left = closest_pair(pl, pyl)
    delta_right = closest_pair(pr, pyr)
    delta = min(delta_left, delta_right)
    
    # Merge
    # Filter points within delta distance from the midpoint
    strip = []
    for point in py:
        if abs(point[0] - midpoint[0]) < delta:
            strip.append(point)
    
    # Compare points in the strip
    min_strip = delta
    for i in range(len(strip)):
        for j in range(i + 1, len(strip)):
            if strip[j][1] - strip[i][1] >= min_strip:
                break  # Early exit as points are sorted by y
            d = distance(strip[i], strip[j])
            if d < min_strip:
                min_strip = d
    return min(min_strip, delta)

def minimum_distance(points):
    px = sorted(points, key=lambda x: x[0])
    py = sorted(points, key=lambda x: x[1])
    return closest_pair(px, py)


def positional_pixel_step(coords, full_coor, delta, coord_sf):
    """
    Batch generation of 2D position codes.
    
    Parameters:
    coords (numpy.ndarray): a two-dimensional array of coordinates of shape (n, 2), where n is the number of coordinates.
    delta (int): bias.
    
    Back:
    encoding (numpy.ndarray): Position encoding with shape (n, 2).
    """
    if delta <= 0:
        raise ValueError("delta must be greater than 0.")
    
    num1 = coords.shape[0]  # num
    num2 = full_coor.shape[0]  # num2
    coords_c = coords.copy()
    coords_c = coords_c.astype(np.float64)
    full_coor_c = full_coor.copy()
    full_coor_c = full_coor_c.astype(np.float64)
    pixel_step = minimum_distance(full_coor_c)*coord_sf # max(minimum_distance(coords_c), minimum_distance(full_coor_c))*coord_sf
    skew_step_x = min(np.min(coords_c[:,0]), np.min(full_coor_c[:,0]))
    skew_step_y = min(np.min(coords_c[:,1]), np.min(full_coor_c[:,1]))
    print(skew_step_x,skew_step_y)
    print("pixel_step:", pixel_step) #, "skew_step:", skew_step
    # int_result = int(pixel_step)  # 截断为整数
    # float_result = float(int_result)  # 转换为浮点数
    coords_c[:,0] = (coords_c[:,0]-skew_step_x+pixel_step)/pixel_step
    coords_c[:,1] = (coords_c[:,1]-skew_step_y+pixel_step)/pixel_step
    full_coor_c[:,0] = (full_coor_c[:,0]-skew_step_x+pixel_step)/pixel_step
    full_coor_c[:,1] = (full_coor_c[:,1]-skew_step_y+pixel_step)/pixel_step
    
    # print(type(coords_c[0,0]))
    # print(type(full_coor_c[0,0]))
    # if np.max(full_coor_c[:,0]) >= np.max(full_coor_c[:,1]):
    #     xy = float(np.max(full_coor_c[:,0])//np.max(full_coor_c[:,1]))
    #     coords_c[:,0] = coords_c[:,0]/xy
    #     full_coor_c[:,0] = full_coor_c[:,0]/xy
    #     print(max(coords_c[:,0]),max(full_coor_c[:,0]))
    # else:
    #     xy = float(np.max(full_coor_c[:,1])//np.max(full_coor_c[:,0]))
    #     coords_c[:,1] = coords_c[:,1]/xy
    #     full_coor_c[:,1] = full_coor_c[:,1]/xy
    #     print(max(coords_c[:,1]),max(full_coor_c[:,1]))
    # print(xy)
    
    # x = coords_c[:, 0]
    # y = coords_c[:, 1]
    # # creat position_indices, len = d_model // 2
    # position_indices = np.arange(0, d_model // 2, 2)
    # div_term = 10000 ** (position_indices / d_model)
    
    # # initialize encoding
    # encoding = np.zeros((n, d_model))
    
    # # encoding x 
    # encoding[:, 0:d_model // 2:2] = np.sin(x[:, None] / div_term)  # x 的正弦部分
    # encoding[:, 1:d_model // 2:2] = np.cos(x[:, None] / div_term)  # x 的余弦部分
    
    # # encoding y 
    # encoding[:, d_model // 2::2] = np.sin(y[:, None] / div_term)  # y 的正弦部分
    # encoding[:, d_model // 2 + 1::2] = np.cos(y[:, None] / div_term)  # y 的余弦部分
    # encoding_df = pd.DataFrame(encoding)
    encoding_df = pd.DataFrame(coords_c)
    full_encoding_df = pd.DataFrame(full_coor_c)
    return encoding_df, full_encoding_df

def masked_anndata(adata = None, mask_ratio=0.5):
    total_numbers = adata.X.shape[0]
    numbers_to_pick = int(total_numbers * 0.5)
    # Generates a list from 0 to total_numbers
    number_list = list(range(total_numbers))
    # Random selection using the sample function
    masked_index = random.sample(number_list, numbers_to_pick)

    masked_adata = adata.copy()
    masked_adata.X[masked_index] = 0
    set_number_list = set(number_list)
    set_masked_index = set(masked_index)

    # 使用差集操作获取结果
    remaining_index = list(set_number_list.difference(set_masked_index))
    # 创建一个布尔数组，用于过滤spots
    # 初始化为False，表示默认不选择
    filter_mask = np.zeros(total_numbers, dtype=bool)

    # 将masked_index对应的位置设置为True，表示选择这些spots
    filter_mask[masked_index] = True

    # 使用布尔索引过滤adata中的spots
    # 这里我们选择那些没有被随机挑选中的spots
    filtered_adata = masked_adata.copy()[~filter_mask, :]
    return adata, masked_adata, filtered_adata, masked_index, remaining_index

def generation_coord(
        adata,
        name='spatial',
        res = 2.0
        
):
    """ This function generates spatial location for Spatial Transcriptomics data.
        Args:
            adata: AnnData object storing original data.
            name: Item in adata.obsm used for get spatial location. Default is "coord".
        Return:
            coor_df: Spatial location of original data.
            fill_coor_df: Spatial location of generated data.
    """
    if res ==2:
        coor_df = pd.DataFrame(adata.obsm[name], dtype=np.float64)
        pixel_step = minimum_distance(adata.obsm[name])
        coor_df_1 = coor_df.copy()
        coor_df_1.iloc[:, 1] = coor_df_1.iloc[:, 1] + pixel_step/2.0
    
        coor_df_2 = coor_df.copy()
        coor_df_2.iloc[:, 0] = coor_df_2.iloc[:, 0] + pixel_step/2.0
    
        coor_df_3 = coor_df.copy()
        coor_df_3.iloc[:, 0] = coor_df_3.iloc[:, 0] + pixel_step/2.0
        coor_df_3.iloc[:, 1] = coor_df_3.iloc[:, 1] + pixel_step/2.0
    
        fill_coor_df = pd.concat([coor_df, coor_df_1, coor_df_2, coor_df_3])
    elif res ==3:
        coor_df = pd.DataFrame(adata.obsm[name], dtype=np.float64)
        pixel_step = minimum_distance(adata.obsm[name])
        coor_df_1 = coor_df.copy()
        coor_df_1.iloc[:, 1] = coor_df_1.iloc[:, 1] + pixel_step/3.0
    
        coor_df_2 = coor_df.copy()
        coor_df_2.iloc[:, 0] = coor_df_2.iloc[:, 0] + pixel_step/3.0
    
        coor_df_3 = coor_df.copy()
        coor_df_3.iloc[:, 0] = coor_df_3.iloc[:, 0] + pixel_step/3.0
        coor_df_3.iloc[:, 1] = coor_df_3.iloc[:, 1] + pixel_step/3.0

        coor_df_4 = coor_df.copy()
        coor_df_4.iloc[:, 0] = coor_df_4.iloc[:, 0] + pixel_step/3.0*2
        coor_df_4.iloc[:, 1] = coor_df_4.iloc[:, 1] + pixel_step/3.0

        coor_df_5 = coor_df.copy()
        coor_df_5.iloc[:, 0] = coor_df_5.iloc[:, 0] + pixel_step/3.0
        coor_df_5.iloc[:, 1] = coor_df_5.iloc[:, 1] + pixel_step/3.0*2

        coor_df_6 = coor_df.copy()
        coor_df_6.iloc[:, 1] = coor_df_6.iloc[:, 1] + pixel_step/3.0*2
    
        coor_df_7 = coor_df.copy()
        coor_df_7.iloc[:, 0] = coor_df_7.iloc[:, 0] + pixel_step/3.0*2
    
        coor_df_8 = coor_df.copy()
        coor_df_8.iloc[:, 0] = coor_df_8.iloc[:, 0] + pixel_step/3.0*2
        coor_df_8.iloc[:, 1] = coor_df_8.iloc[:, 1] + pixel_step/3.0*2
        fill_coor_df = pd.concat([coor_df, coor_df_1, coor_df_2, coor_df_3, coor_df_4, coor_df_5, coor_df_6,coor_df_7, coor_df_8])
    fill_coor_df = fill_coor_df.drop_duplicates(subset=fill_coor_df.columns)

    coor_df.index=adata.obs.index
    coor_df.columns=["x","y"]
    fill_coor_df.columns = ["x", "y"]

    return coor_df.to_numpy().copy(), fill_coor_df.to_numpy().copy()

def setToArray(
        setInput,
        dtype='int64'
):
    """ This function transfer set to array.
        Args:
            setInput: set need to be trasnfered to array.
            dtype: data type.

        Return:
            arrayOutput: trasnfered array.
    """
    arrayOutput = np.zeros(len(setInput), dtype=dtype)
    index = 0
    for every in setInput:
        arrayOutput[index] = every
        index += 1
    return arrayOutput

def recovery_coord(
        adata,
        name='spatial',
        mask_ratio=0.5,
):
    """ This function generates spatial location for Spatial Transcriptomics data.
        Args:
            adata: AnnData object storing original data.
            name: Item in adata.obsm used for get spatial location. Default is "coord".
            mask_ratio: Down-sampling ratio. Default is 0.5.
        Return:
            coor_df: Spatial location of dowm-sampled data.
            fill_coor_df: Spatial location of recovered data.
            sample_index: Index of downsampled data.
            sample_barcode: Barcode of downsampled data.
    """
    coor_df = pd.DataFrame(adata.obsm[name])
    coor_df.index = adata.obs.index
    coor_df.columns = ["x", "y"]
    sample_index=np.random.choice(range(coor_df.shape[0]), size=round(mask_ratio*coor_df.shape[0]), replace=False)
    sample_index = setToArray(set(sample_index))
    sample_coor_df = coor_df.iloc[sample_index]
    sample_barcode = coor_df.index[sample_index]

    del_index = setToArray(set(range(coor_df.shape[0])) - set(sample_index))

    return sample_coor_df.to_numpy().copy(), coor_df.to_numpy().copy(), sample_index, sample_barcode

from scipy.sparse import csr_matrix, csc_matrix, coo_matrix
from PIL import Image
import anndata as ad
def ms2anndata(ms_org,ms_sp,HE_path,in_tissue=True):
    spot=[f"mspx_{i}" for i in range(len(ms_org))]
    ms_name=[f"{i}" for i in ms_org.iloc[:,2:].columns]
    
    ms_cons=pd.concat([pd.Series(spot),ms_sp.loc[:,["he_x","he_y"]],ms_org],axis=1)
    # 计算每一行的总和（从第3列开始）
    row_sums = ms_cons.iloc[:, 5:].sum(axis=1)
    # 删除总和为0的行
    ms_n = ms_cons[row_sums >= 1e6]
    ms_n.iloc[:, 5:] = ms_n.iloc[:, 5:].apply(lambda x: x - x.min(), axis=0)
    df_ms = ms_n.iloc[:, 5:].T
    ms_intensity = csr_matrix(ms_n.iloc[:,5:], dtype=np.float32)
    ms_adata = ad.AnnData(ms_intensity)
    ms_adata.var_names=ms_n.iloc[:,5:].columns
    ms_adata.obs_names=ms_n.iloc[:,0]
    ms_n.index = ms_n.iloc[:,0]
    ms_adata.obs["array_row"]=ms_n.iloc[:,3] ##raw x
    ms_adata.obs["array_col"]=ms_n.iloc[:,4] ##raw y
    ms_adata.obsm["spatial"]=ms_n.loc[:,["he_x","he_y"]].to_numpy()
    image_path = HE_path
    image = Image.open(image_path)
    image_array = np.array(image)
    spatial_key = "spatial"
    library_id = "tissue"  # 你可以自定义这个 ID
    ms_adata.uns[spatial_key] = {library_id: {}}
    ms_adata.uns[spatial_key][library_id]["images"] = {"hires": image_array}
    ms_adata.uns[spatial_key][library_id]["scalefactors"] = {
        "tissue_hires_scalef": 1,  # 图像像素和空间坐标的比例因子
        "spot_diameterres_full": 0.5,  # 每个观测点的直径
        'fiducial_diameter_fullres': 609.8565193216596,
        'spot_diameter_fullres': 377.5302262467417
    }
    return ms_adata