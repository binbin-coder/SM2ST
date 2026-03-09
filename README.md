# SM2ST

## Overview
 ![Image text](https://github.com/binbin-coder/SM2ST/blob/main/overview.png)
   Spatial transcriptomics (ST) and spatial metabolomics (SM) offer complementary insights into the tissue microenvironment by mapping gene expression and metabolic dynamics, respectively. However, integrating these modalities is hindered by instrumental drift during long-term acquisition and spatial heterogeneity in chemical matrix coverage inherent to matrix-assisted laser desorption/ionization mass spectrometry imaging (MALDI-MSI), which collectively complicate cross-modal alignment. Here we introduce an innovative multimodal registration framework that adopts hematoxylin-and-eosin (H&E) stained image as a bridging modality to compute adaptive affine transformations, aligning MSI with histological references and achieving landmark registration errors below 10 pixels. This process enables spatial transcriptomics and spatial metabolomics to be projected into a unified coordinate system. Building upon this, we propose a novel architectural framework that integrates generative adversarial networks (GANs) with autoencoders. This innovative approach enables effective denoising of metabolic ion signals and remapping onto spatial transcriptomic loci, thereby achieving point-to-point co-registration between the two modalities. The proposed method effectively addresses spatial heterogeneity across diverse data types, thereby enabling the development of a unified spatial multi-omics analytical framework. By integrating SM into the public ST development platform, we enable the sharing of established resources, such as multimodal clustering, while simultaneously addressing the limited sensitivity of high-mass molecules in conventional high-resolution MALDI-MS. Specifically, we introduce STMGraph, a self-supervised super-resolution model that compensates for these limitations.

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
```https://sm2st-tutorial.readthedocs.io/en/latest/``` 

### Benchmark Testing
```https://github.com/binbin-coder/SM2ST/blob/main/Tutorial/Normal_Resolution_SM2ST_masked_pearsonr_30o_rec.ipynb```  
```https://github.com/binbin-coder/SM2ST/blob/main/Tutorial/Normal_Resolution_SM2ST_masked_pearsonr_30o_rec_STAGE.ipynb```  
```https://github.com/binbin-coder/SM2ST/blob/main/Tutorial/Normal_Resolution_SM2ST_masked_pearsonr_30_rec_mask_03.ipynb```  

### Integrate spatial multi-omics
```https://github.com/binbin-coder/SM2ST/blob/main/Tutorial/rectification31t2_SM2ST_test.ipynb```  
```https://github.com/binbin-coder/SM2ST/blob/main/Tutorial/rectification31t2_STAGE_test.ipynb```  
```https://github.com/binbin-coder/SM2ST/blob/main/Tutorial/rectification41t2_SM2ST_test.ipynb```   

### manual rectification
```https://github.com/binbin-coder/SM2ST/blob/main/Tutorial/manual_rectification.ipynb```  
```https://github.com/binbin-coder/SM2ST/blob/main/Tutorial/Multi_omics_Cluster.ipynb```  

### Multi-omics clustering
```https://github.com/binbin-coder/SM2ST/blob/main/Tutorial/Muliti_omics_SpatialGlue_tutorial_smst.ipynb```  

### super resolution
```https://github.com/binbin-coder/SM2ST/blob/main/Tutorial/super_Resolution_STMGraph_pyG1_30rex.ipynb```   


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
