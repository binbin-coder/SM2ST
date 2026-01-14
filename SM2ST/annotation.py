from molmass import Formula
import pandas as pd
import anndata
from scipy.spatial import cKDTree
from pathlib import Path
MODULE_PATH = Path(__file__).parent
import scanpy as sc
from anndata import AnnData

class AnnDataSM(AnnData):
    """
    Anndata object for Spatial Metabolomics data
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._spatialtk_type = 'AnnDataSM'

    @classmethod
    def from_anndata(cls, adata, *args, **kwargs):
        if not isinstance(adata, sc.AnnData):
            raise ValueError("Input must be a scanpy AnnData object")
        return cls(adata, *args, **kwargs)

    def _gen_repr(self, n_obs, n_vars) -> str:
        if self.isbacked:
            backed_at = f" backed at {str(self.filename)!r}"
        else:
            backed_at = ""
        descr = f"Spatial Metabolomics AnnData object with n_obs × n_vars = {n_obs} × {n_vars}{backed_at}"
        for attr in [
            "obs",
            "var",
            "uns",
            "obsm",
            "varm",
            "layers",
            "obsp",
            "varp",
        ]:
            keys = getattr(self, attr).keys()
            if len(keys) > 0:
                descr += f"\n    {attr}: {str(list(keys))[1:-1]}"
        return descr

    def __repr__(self) -> str:
        if self.is_view:
            return "View of " + self._gen_repr(self.n_obs, self.n_vars)
        else:
            return self._gen_repr(self.n_obs, self.n_vars)

class AnnDataST(AnnData):
    """
    Anndata object for Spatial Transcriptomics data
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._spatialtk_type = 'AnnDataST'

    @classmethod
    def from_anndata(cls, adata, *args, **kwargs):
        if not isinstance(adata, sc.AnnData):
            raise ValueError("Input must be a scanpy AnnData object")
        return cls(adata, *args, **kwargs)
    
    def _gen_repr(self, n_obs, n_vars) -> str:
        if self.isbacked:
            backed_at = f" backed at {str(self.filename)!r}"
        else:
            backed_at = ""
        descr = f"Spatial Transcriptomics AnnData object with n_obs × n_vars = {n_obs} × {n_vars}{backed_at}"
        for attr in [
            "obs",
            "var",
            "uns",
            "obsm",
            "varm",
            "layers",
            "obsp",
            "varp",
        ]:
            keys = getattr(self, attr).keys()
            if len(keys) > 0:
                descr += f"\n    {attr}: {str(list(keys))[1:-1]}"
        return descr

    def __repr__(self) -> str:
        if self.is_view:
            return "View of " + self._gen_repr(self.n_obs, self.n_vars)
        else:
            return self._gen_repr(self.n_obs, self.n_vars)
        
class AnnDataJointSMST(AnnData):
    """
    Anndata object for Joint Spatial Metabolomics and Transcriptomics data
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._spatialtk_type = 'AnnDataJointSTSM'

    def _gen_repr(self, n_obs, n_vars) -> str:
        if self.isbacked:
            backed_at = f" backed at {str(self.filename)!r}"
        else:
            backed_at = ""
        descr = f"Joint Spatial Transcriptomics and Metabolomics\n" + "AnnData object with n_obs × n_vars = {n_obs} × {n_vars}{backed_at}"
        for attr in [
            "obs",
            "var",
            "uns",
            "obsm",
            "varm",
            "layers",
            "obsp",
            "varp",
        ]:
            keys = getattr(self, attr).keys()
            if len(keys) > 0:
                descr += f"\n    {attr}: {str(list(keys))[1:-1]}"
        return descr

    def __repr__(self) -> str:
        if self.is_view:
            return "View of " + self._gen_repr(self.n_obs, self.n_vars)
        else:
            return self._gen_repr(self.n_obs, self.n_vars)

def _calculate_ppm_range(
    observed_mz, 
    ppm_tolerance=5
):
    ppm_range = observed_mz * (ppm_tolerance / 1e6)
    lower_limit = observed_mz - ppm_range
    upper_limit = observed_mz + ppm_range
    return pd.Series([ppm_range, lower_limit, upper_limit], index=['ppm_range', 'lower_limit', 'upper_limit'])

def metabolite_annotation(
    adata_SM:AnnDataSM,
    adduct_type: str,
    adduct_method: str,
    tolerance_ppm: float = 5,
    inplace: bool = True
) -> pd.DataFrame:
    """
    Annotate metabolites based on the m/z values.
    :param adata_SM: AnnDataSM. The AnnDataSM object.
    :param adduct_type: str. The adduct type.
    :param adduct_method: str. The adduct method, 'add' or 'sub'.
    :param tolerance_ppm: float, default 5. The tolerance in ppm.
    :param inplace: bool, default True. Whether to modify the AnnDataSM object inplace.
    :return: pd.DataFrame. The annotated metabolites.
    """
    sf_results_df = pd.read_csv(MODULE_PATH / "./data/hmdb.csv", index_col=0)
    if adduct_type==None:
        adduct_mz = 0
    else:
        f_adduct = Formula(adduct_type)
        adduct_mz = f_adduct.monoisotopic_mass
    if adduct_method == 'add':
        sf_results_df['m/z_addion'] = sf_results_df['monisotopic_molecular_weight'] + adduct_mz
    elif adduct_method == 'sub':
        sf_results_df['m/z_addion'] = sf_results_df['monisotopic_molecular_weight'] - adduct_mz
    else:
        raise ValueError("adduct_method should be 'add' or 'sub'")
    sf_results_df[['ppm_range', 'lower_limit', 'upper_limit']] = sf_results_df['m/z_addion'].apply(
        lambda x: _calculate_ppm_range(x, ppm_tolerance=tolerance_ppm))
    var_df = adata_SM.var.copy()
    var_df['name'] = var_df['name'].astype(float)
    var_df['accession'] = None
    var_df['metabolite_name'] = None
    var_df['iupac_name']=None
    var_df["chemical_formula"]=None
    var_df['kegg']=None
    var_df['bigg']=None
    var_df['direct_parent']=None
    var_df['class']=None
    var_df['sub_class']=None
    sf_results_df['lower_limit'] = sf_results_df['lower_limit'].astype(float)
    sf_results_df['upper_limit'] = sf_results_df['upper_limit'].astype(float)
    #sf_results_df = sf_results_df.sort_values('lower_limit')
    kdtree = cKDTree(sf_results_df.loc[:,['lower_limit','upper_limit']])
    for index,row in var_df.iterrows():
        #print(index)
        mz_target = row['name']
        query_point=[mz_target,mz_target]
        distance, index2 = kdtree.query(query_point)
        lower_limit = float(sf_results_df.loc[index2]['lower_limit'])  
        upper_limit = float(sf_results_df.loc[index2]['upper_limit'])
        mz_target = float(mz_target)
        if lower_limit <= mz_target <= upper_limit:
            sf_accession = sf_results_df.loc[index2]['accession']
            sf_name = sf_results_df.loc[index2]['name']
            sf_iupac_name = sf_results_df.loc[index2]['iupac_name']
            sf_chemical_formula = sf_results_df.loc[index2]['chemical_formula']
            sf_kegg = sf_results_df.loc[index2]['kegg']
            sf_bigg = sf_results_df.loc[index2]['bigg']
            sf_direct_parent = sf_results_df.loc[index2]['direct_parent']
            sf_class = sf_results_df.loc[index2]['class']
            sf_subclass = sf_results_df.loc[index2]['sub_class']
        else:
            sf_accession = None
            sf_name = None
            sf_iupac_name = None
            sf_chemical_formula = None
            sf_kegg = None
            sf_bigg = None
            sf_direct_parent = None
            sf_class = None
            sf_subclass = None
        var_df.loc[index,'accession'] = sf_accession
        var_df.loc[index,'metabolite_name'] = sf_name
        var_df.loc[index,'iupac_name'] = sf_iupac_name
        var_df.loc[index,'chemical_formula'] = sf_chemical_formula
        var_df.loc[index,'kegg'] = sf_kegg
        var_df.loc[index,'bigg'] = sf_bigg
        var_df.loc[index,'direct_parent'] = sf_direct_parent
        var_df.loc[index,'class'] = sf_class
        var_df.loc[index,'sub_class'] = sf_subclass
    if inplace:
        adata_SM.var['accession'] = var_df['accession'].values
        adata_SM.var['metabolite_name'] = var_df['metabolite_name'].values
        adata_SM.var['iupac_name'] = var_df['iupac_name'].values
        adata_SM.var['chemical_formula'] = var_df['chemical_formula'].values
        adata_SM.var['kegg'] = var_df['kegg'].values
        adata_SM.var['bigg'] = var_df['bigg'].values
        adata_SM.var['direct_parent'] = var_df['direct_parent'].values
        adata_SM.var['class'] = var_df['class'].values
        adata_SM.var['sub_class'] = var_df['sub_class'].values
    return var_df