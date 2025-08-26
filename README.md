# STMGraph

## Overview
 ![Image text](https://github.com/binbin-coder/SM2ST/blob/main/overview.png)
   Spatial transcriptomics, leveraging in situ sequencing or in situ hybridization, reveals the spatial distribution of gene expression at single-cell or sub-cellular resolution within tissue sections.  Concurrently, spatial metabolomics employs mass-spectrometry imaging (MSI), specifically MALDI-MS, to map metabolite distributions and provide molecular insight into metabolic activity.  Together, these techniques furnish complementary information on gene expression and metabolic dynamics for interrogating the tissue microenvironment.  However, the drift of MSI during the instrument’s long-term acquisition and the heterogeneity in its spatial chemical matrix coverage introduce noise that complicates cross-modal spatial alignment. Here we introduce an innovative multimodal registration framework that first adopts hematoxylin-and-eosin (H&E) stained images as a bridge to construct adaptive affine transformation matrices, aligning MSI data with the H&E reference.  This process enables spatial transcriptomics and spatial metabolomics to be projected into a unified coordinate system. Building upon this, we propose a novel architectural framework that integrates generative adversarial networks (GANs) with autoencoders.  This innovative approach enables effective denoising of metabolic ion signals and accurate remapping onto spatial transcriptomic loci, thereby achieving precise point-to-point co-registration between the two modalities. The proposed method effectively addresses spatial heterogeneity across diverse data types, thereby significantly enhancing the elucidation of mouse brain regions through refined latent embeddings of gene expression and metabolite distributions. Furthermore, via a self-supervised super-resolution model termed STMGraph, we compensate for the limited sensitivity of high-mass molecules in conventional high-resolution MALDI-MS, effectively enhancing spatial metabolomic resolution.

## Software dependencies
numpy ==1.26.4  
squidpy == 1.6.1  
scanpy == 1.9.8  
r-base  == 4.2.2  
rpy2 ==3.5.9  
torch-cluster ==1.6.1+pt113cu117  
torch-geometric == 2.5.3  
torch-scatter== 2.1.1+pt113cu117  
torch-sparse == 0.6.17+pt113cu117  
torch-spline-conv ==1.2.2+pt113cu117  
pytorch == 1.13.1  


## Installation
conda env create -f environment.yaml  
pip install sm2st

## Tutorial
### Benchmark Testing
```https://github.com/binbin-coder/SM2ST/blob/main/Tutorial/Normal_Resolution_SM2ST_masked_pearsonr_30o_rec.ipynb```  

### Integrate spatial multi-omics
```https://github.com/binbin-coder/SM2ST/blob/main/Tutorial/rectification20t2.ipynb```  
```https://github.com/binbin-coder/SM2ST/blob/main/Tutorial/rectification21t2.ipynb```  
```https://github.com/binbin-coder/SM2ST/blob/main/Tutorial/rectification22t2.ipynb```   

### manual rectification
```https://github.com/binbin-coder/SM2ST/blob/main/Tutorial/manual_rectification.ipynb```  
```https://github.com/binbin-coder/SM2ST/blob/main/Tutorial/Multi_omics_Cluster.ipynb```  

### Multi-omics clustering
```https://github.com/binbin-coder/SM2ST/blob/main/Tutorial/Muliti_omics_SpatialGlue_tutorial_smst.ipynb```  

### super resolution
```https://github.com/binbin-coder/SM2ST/blob/main/Tutorial/super_Resolution_STMGraph_30rex.ipynb```   


## Reference the software tutorial
* [ShinyCardinal](https://github.com/YonghuiDong/ShinyCardinal)
* [SCANPY](https://github.com/scverse/scanpy-tutorials)
* [spatialglue](https://spatialglue-tutorials.readthedocs.io/en/latest/index.html)
* [squidpy](https://squidpy.readthedocs.io/en/stable/)
* [STAGE](https://github.com/zhanglabtools/STAGE)
* [STMGraph](https://github.com/binbin-coder/STMGraph_pyg)


## Download test datasets used in SM2ST:
The datasets used in this paper can be downloaded from the following websites. Specifically,

(1) The  spatial multimodal analysis (SMA) protocol dataset https://data.mendeley.com/datasets/w7nw4km7xd/1

(2) MSI of the mouse hemisphere https://drive.google.com/drive/folders/1DT5NJrNumVVC8o43LmMNTPB5Klxl7lyz


## Contact
Feel free to submit an issue or contact us at llx_1910@163.com for problems about the packages.
