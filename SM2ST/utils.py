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
    # int_result = int(pixel_step)  # trans int
    # float_result = float(int_result)  # trans float
    coords_c[:,0] = (coords_c[:,0]-skew_step_x+pixel_step)/pixel_step
    coords_c[:,1] = (coords_c[:,1]-skew_step_y+pixel_step)/pixel_step
    full_coor_c[:,0] = (full_coor_c[:,0]-skew_step_x+pixel_step)/pixel_step
    full_coor_c[:,1] = (full_coor_c[:,1]-skew_step_y+pixel_step)/pixel_step
    
    encoding_df = pd.DataFrame(coords_c)
    full_encoding_df = pd.DataFrame(full_coor_c)
    return encoding_df, full_encoding_df

def masked_anndata(adata = None, mask_ratio=0.5):
    total_numbers = adata.X.shape[0]
    numbers_to_pick = int(total_numbers * mask_ratio)
    # Generates a list from 0 to total_numbers
    number_list = list(range(total_numbers))
    # Random selection using the sample function
    masked_index = random.sample(number_list, numbers_to_pick)

    masked_adata = adata.copy()
    masked_adata.X[masked_index] = 0
    set_number_list = set(number_list)
    set_masked_index = set(masked_index)

    # Obtain the result using the difference set operation
    remaining_index = list(set_number_list.difference(set_masked_index))
    # Create a Boolean array for filtering spots
    # Initialize to False, indicating that no selection is made by default
    filter_mask = np.zeros(total_numbers, dtype=bool)

    # Set the positions corresponding to masked_index to True, indicating the selection of these spots
    filter_mask[masked_index] = True

    # Filter spots in adata using Boolean indexes
    # Here we select those spots that were not randomly chosen
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
        keep_ratio=0.5,
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
    sample_index=np.random.choice(range(coor_df.shape[0]), size=round(keep_ratio*coor_df.shape[0]), replace=False)
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
    # Calculate the sum of each row (starting from the third column)
    row_sums = ms_cons.iloc[:, 5:].sum(axis=1)
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
    library_id = "tissue"  # You can customize this ID
    ms_adata.uns[spatial_key] = {library_id: {}}
    ms_adata.uns[spatial_key][library_id]["images"] = {"hires": image_array}
    ms_adata.uns[spatial_key][library_id]["scalefactors"] = {
        "tissue_hires_scalef": 1,  # The scale factor of image pixels and spatial coordinates
        "spot_diameterres_full": 0.5,  # The diameter of each observation point
        'fiducial_diameter_fullres': 609.8565193216596,
        'spot_diameter_fullres': 377.5302262467417
    }
    return ms_adata

import numpy as np
import anndata as ad
from skimage.metrics import structural_similarity as ssim
from scipy import sparse
import matplotlib.pyplot as plt
import seaborn as sns

def adata_to_feature_images(adata, spatial_key='spatial', image_shape=None):
    """
    Convert each feature in an AnnData object into a 2D spatial image.
    FIX: Ensures all arrays passed to np.add.at are standard numpy arrays, not AnnData Views.
    """
    # 1. Retrieve spatial coordinates
    if spatial_key not in adata.obsm:
        raise KeyError(f"Spatial key '{spatial_key}' not found in adata.obsm")
    
    spatial_coords = adata.obsm[spatial_key].copy()
    
    # Ensure coordinates are integers
    if not np.issubdtype(spatial_coords.dtype, np.integer):
        spatial_coords = np.round(spatial_coords).astype(int)
    
    x_coords = spatial_coords[:, 0]
    y_coords = spatial_coords[:, 1]
    
    # 2. Determine image dimensions
    if image_shape is None:
        max_x, max_y = x_coords.max(), y_coords.max()
        image_shape = (int(max_y) + 1, int(max_x) + 1)
    
    height, width = image_shape
    
    # Filter out points outside the image boundaries
    valid_mask = (x_coords >= 0) & (x_coords < width) & (y_coords >= 0) & (y_coords < height)
    x_valid = x_coords[valid_mask]
    y_valid = y_coords[valid_mask]
    
    # Access the expression matrix
    X = adata.X
    is_sparse = sparse.issparse(X)
    
    n_features = adata.n_vars
    feature_names = adata.var_names
    
    print(f"Generating {n_features} feature images (shape: {image_shape})...")
    
    feature_images = {}
    
    for i, fname in enumerate(feature_names):
        # Create fresh zero arrays (these are standard numpy arrays, safe to use)
        img_sum = np.zeros(image_shape, dtype=np.float64)
        img_count = np.zeros(image_shape, dtype=np.int32)
        
        # Extract expression values for the current feature
        if is_sparse:
            col_data = X[valid_mask, i]
            # .toarray() returns a standard numpy array, but let's be explicit with .flatten()
            vals = col_data.toarray().flatten() if sparse.issparse(col_data) else np.asarray(col_data).flatten()
        else:
            # CRITICAL FIX HERE: 
            # When slicing adata[:, gene], X might be an AnnData View.
            # We must convert it to a standard numpy array explicitly.
            raw_vals = X[valid_mask, i]
            vals = np.asarray(raw_vals).flatten()
        
        # Flatten 2D coordinates to 1D indices
        flat_indices = y_valid * width + x_valid
        
        # CRITICAL FIX HERE:
        # img_sum is already a standard numpy array created by np.zeros, so img_sum.ravel() 
        # should be fine. However, to be absolutely safe against any upstream weirdness,
        # we ensure the target buffer is a standard numpy array.
        target_buffer = np.asarray(img_sum).ravel()
        count_buffer = np.asarray(img_count).ravel()
        
        # Now np.add.at will work because target_buffer is a pure numpy array
        np.add.at(target_buffer, flat_indices, vals)
        np.add.at(count_buffer, flat_indices, 1)
        
        # Calculate mean
        with np.errstate(divide='ignore', invalid='ignore'):
            img_final = img_sum / img_count
            img_final[np.isnan(img_final)] = 0.0
            
        feature_images[fname] = img_final.astype(np.float32)
        
        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{n_features} features...")
            
    return feature_images
    

def calculate_spatial_feature_ssim(adata1, adata2, spatial_key='spatial', 
                                   image_shape=None, win_size=7):
    """
    Calculate SSIM for each feature between two spatial datasets.
    """
    # 1. 检查共同特征
    common_features = np.intersect1d(adata1.var_names, adata2.var_names)
    if len(common_features) == 0:
        raise ValueError("No common features found between the two datasets.")
    
    # 2. 检查坐标一致性 (可选，但推荐)
    # 注意：这里只检查前几个点或形状，完全比较可能很慢
    if adata1.obsm[spatial_key].shape != adata2.obsm[spatial_key].shape:
        print("⚠️ Warning: Spatial coordinate shapes differ.")
    else:
        # 简单采样检查
        if not np.array_equal(adata1.obsm[spatial_key][:10], adata2.obsm[spatial_key][:10]):
            print("⚠️ Warning: Spatial coordinates seem inconsistent. Results may be unreliable.")

    # 3. 自动推断图像大小 (如果未提供)
    if image_shape is None:
        coords = np.vstack([adata1.obsm[spatial_key], adata2.obsm[spatial_key]])
        max_x, max_y = coords[:, 0].max(), coords[:, 1].max()
        image_shape = (int(max_y) + 1, int(max_x) + 1)
        print(f"Auto-determined image shape: {image_shape}")

    # 4. 生成图像
    print("Generating images for dataset 1...")
    images1 = adata_to_feature_images(adata1, spatial_key, image_shape)
    
    print("Generating images for dataset 2...")
    images2 = adata_to_feature_images(adata2, spatial_key, image_shape)
    
    # 5. 计算 SSIM
    ssim_scores = {}
    print(f"Calculating SSIM for {len(common_features)} features...")
    
    # 调整 win_size 必须是奇数且小于图像最小维度
    min_dim = min(image_shape)
    if win_size >= min_dim:
        win_size = min_dim - 1 if (min_dim - 1) % 2 == 1 else min_dim - 2
        print(f"⚠️ Adjusted win_size to {win_size} to fit image dimensions.")
    if win_size % 2 == 0:
        win_size += 1

    for idx, feature in enumerate(common_features):
        img1 = images1[feature]
        img2 = images2[feature]
        
        # 动态计算 data_range
        # skimage 要求 data_range 是浮点数，通常是图像的最大可能值或实际最大值
        max_val = max(img1.max(), img2.max())
        min_val = min(img1.min(), img2.min())
        data_range = max_val - min_val
        
        if data_range == 0:
            # 如果两幅图都是常数且相同，SSIM 定义为 1
            ssim_scores[feature] = 1.0
            continue
        
        try:
            # 关键修复：
            # 1. 使用 data_range 参数
            # 2. 使用 win_size 参数 (skimage >= 0.19)
            # 3. multichannel=False (或 channel_axis=None 在新版中不需要显式设为None，默认就是灰度)
            # 4. full=False 确保只返回 score 标量
            score = ssim(
                img1, img2,
                data_range=data_range,
                win_size=win_size,
                full=False,
                channel_axis=None # 显式指定无通道轴，兼容新版 skimage
            )
            ssim_scores[feature] = float(score)
            
        except Exception as e:
            print(f"Error calculating SSIM for {feature}: {e}")
            ssim_scores[feature] = np.nan
        
        if (idx + 1) % 100 == 0:
            print(f"  Finished {idx + 1}/{len(common_features)} features")
    
    valid_scores = [v for v in ssim_scores.values() if not np.isnan(v)]
    avg_ssim = np.mean(valid_scores) if valid_scores else 0.0
    print(f"Finished! Average SSIM: {avg_ssim:.4f}")
    
    return ssim_scores

def visualize_feature_spatial(adata1, adata2, feature_name, spatial_key='spatial', 
                              image_shape=None, ssim_scores=None):
    """
    Visualize spatial distribution and differences.
    """
    # 如果 image_shape 未提供，尝试从数据推断
    if image_shape is None:
        coords = adata1.obsm[spatial_key]
        max_x, max_y = coords[:, 0].max(), coords[:, 1].max()
        image_shape = (int(max_y) + 1, int(max_x) + 1)

    images1 = adata_to_feature_images(adata1, spatial_key, image_shape)
    images2 = adata_to_feature_images(adata2, spatial_key, image_shape)
    
    if feature_name not in images1 or feature_name not in images2:
        raise ValueError(f"Feature {feature_name} not found in generated images.")

    img1 = images1[feature_name]
    img2 = images2[feature_name]
    diff = img1 - img2
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1
    im1 = axes[0].imshow(img1, cmap='magma')
    axes[0].set_title(f'Dataset 1: {feature_name}')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Plot 2
    im2 = axes[1].imshow(img2, cmap='magma')
    axes[1].set_title(f'Dataset 2: {feature_name}')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Plot 3 (Difference)
    ssim_val = ssim_scores.get(feature_name, np.nan) if ssim_scores else np.nan
    title_diff = f'Difference\n(SSIM: {ssim_val:.4f})' if not np.isnan(ssim_val) else 'Difference\n(SSIM: N/A)'
    
    im3 = axes[2].imshow(diff, cmap='RdBu_r')
    axes[2].set_title(title_diff)
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()

import scipy
def calculate_median_rmse(adata_1, adata_2):
    rmse_values = []
    if scipy.sparse.issparse(adata_1.X):
        A = adata_1.X.A
    else:
        A = adata_1.X
    if scipy.sparse.issparse(adata_2.X):
        B = adata_2.X.A
    else:
        B = adata_2.X
    for row_a, row_b in zip(A, B):
        mse = np.mean((row_a - row_b) ** 2)
        rmse = np.sqrt(mse)
        rmse_values.append(rmse)
    median_rmse = np.median(rmse_values)
    print(f"Median RMSE across all rows: {median_rmse}")
    
    mean_rmse = np.mean(rmse_values)
    print(f"Mean RMSE across all rows: {mean_rmse}")
    return rmse_values

import scipy
from scipy.stats import pearsonr
def calculate_median_pearsonr(adata_1, adata_2):
    if scipy.sparse.issparse(adata_1.X):
        A = adata_1.X.A
    else:
        A = adata_1.X
    if scipy.sparse.issparse(adata_2.X):
        B = adata_2.X.A
    else:
        B = adata_2.X
    r_values = []
    p_values = []
    for col1, col2 in zip(A, B):
        r_value, p_value = pearsonr(col1, col2)
        r_values.append(r_value)
        p_values.append(p_value)
    median_r = np.median(r_values)
    print(f"Median r across all rows: {median_r}")
    
    mean_r = np.mean(r_values)
    print(f"Mean r across all rows: {mean_r}")
    return r_values,p_values