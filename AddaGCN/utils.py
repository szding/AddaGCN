import scipy.sparse as sp
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import scipy
import random
from keras.utils import to_categorical
from scipy.sparse import issparse
import anndata as ad
import itertools

def preprocess(adata,
               HVG_method='wilcoxon',
               sp_=False,
               sc_=False,
               celltype='cellType',
               num_markers = 20):


    adata.var['mt'] = adata.var_names.str.startswith('Mt-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

    # raw
    adata.layers['raw'] = adata.X.copy()
    print(np.sum(adata.layers['raw']), adata.layers['raw'].shape)

    # normalize
    sc.pp.normalize_total(adata)
    adata.layers['normalized'] = adata.X.copy()
    print(np.sum(adata.layers['normalized']), adata.layers['normalized'].shape)

    # log1p
    sc.pp.log1p(adata)
    adata.layers['log1p'] = adata.X.copy()
    print(np.sum(adata.layers['log1p']), adata.layers['log1p'].shape)

    # scale
    sc.pp.scale(adata, max_value=10) #
    adata.layers['scaled'] = adata.X.copy()
    print(np.sum(adata.layers['scaled']), adata.layers['scaled'].shape)

    if sp_:
        return adata

    if sc_:
        if HVG_method == 'wilcoxon':
            sc.tl.rank_genes_groups(adata, celltype, method=HVG_method)
            genelists = adata.uns['rank_genes_groups']['names']
            df_genelists = pd.DataFrame.from_records(genelists)
            df_genelists.head(5)

            res_genes = []
            for column in df_genelists.head(num_markers):
                res_genes.extend(df_genelists.head(num_markers)[column].tolist())
            res_genes_ = list(set(res_genes))
            len(res_genes_)
        if HVG_method == 'seurat':
            sc.pp.highly_variable_genes(adata, flavor=HVG_method, n_top_genes=2000, inplace=True)
            res_genes_ = list(adata.var['highly_variable'][adata.var['highly_variable'].values].index)


        return adata, res_genes_


def CT_num(ct_num, target_sum):
    if ct_num == 1:
        valid_pairs = [[target_sum]]
    elif ct_num == 2:
        valid_pairs = []
        for a in range(1, target_sum):
            b = target_sum - a
            if b >= 1:
                valid_pairs.append((a, b))
    elif ct_num == 3:
        valid_pairs = []
        for a in range(1, target_sum):
            for b in range(1, target_sum - a):
                c = target_sum - a - b
                if c > 0:
                    valid_pairs.append((a, b, c))
    elif ct_num == 4:
        valid_pairs = []
        for a in range(1, target_sum):
            for b in range(1, target_sum - a):
                for c in range(1, target_sum - a - b):
                    d = target_sum - a - b - c
                    if d > 0:
                        valid_pairs.append((a, b, c, d))

    return valid_pairs


def pseudo_num(adata, mat_sc, lab_sc_sub, ct_num, target_sum, ct_list):
    sample_exp = []
    sample_prod = []

    pairs = list(itertools.combinations(ct_list, ct_num))
    valid_pairs = CT_num(ct_num, target_sum)
    for pair in pairs:
        for ppair in valid_pairs:

            row_indices = []
            sample_pro = np.zeros((len(ct_list),)).tolist()
            for i in range(len(pair)):
                ct = pair[i]
                cell_index = lab_sc_sub[lab_sc_sub == ct].index.tolist()
                sampled_items = random.choices(cell_index, k=ppair[i])
                ct_indices = [np.where(adata.obs_names == query)[0][0] if query in adata.obs_names else None for query
                              in sampled_items]
                row_indices = row_indices + ct_indices

                ct_index = ct_list.index(ct)
                sample_pro[ct_index] = ppair[i]

            row_means = np.mean(mat_sc[row_indices], axis=0).tolist()  # pseudo
            sample_exp.append(row_means)

            sample_pro = list(sample_pro / np.sum(sample_pro))
            sample_prod.append(sample_pro)

    return sample_exp, sample_prod


def Resampling_sc(adata, mat_sc, lab_sc_sub, ct_keys='cellType'):
    mat_sc_new = mat_sc.copy()
    lab_sc_sub_new = lab_sc_sub.copy()
    ct_num = adata.obs.cellType.value_counts()
    ct_name = adata.obs.cellType.value_counts().index.tolist()
    max_cell_num = max(ct_num)

    for ct in ct_name:
        if ct_num.loc[ct] < max_cell_num:
            sample_num = max_cell_num - ct_num.loc[ct]
            cell_id = lab_sc_sub[lab_sc_sub == ct].index.tolist()
            sampled_items = random.choices(cell_id, k=sample_num)

            formatted_list = [f"{item}-{index + 1}" for index, item in enumerate(sampled_items)]
            row_indices = [np.where(adata.obs_names == query)[0][0] if query in adata.obs_names else None for query in
                           sampled_items]

            mat_sc_new = np.concatenate((mat_sc_new, mat_sc[row_indices]), axis=0)
            lab_sc_sub_new_ = pd.DataFrame([ct] * len(formatted_list), index=formatted_list, columns=[ct_keys])
            lab_sc_sub_new = pd.concat([lab_sc_sub_new, lab_sc_sub_new_.cellType])
    mat_sc = mat_sc_new.copy()
    lab_sc_sub = lab_sc_sub_new.copy()

    return mat_sc, lab_sc_sub


def pseudo_build(adata,
                 nmix=8,
                 iter=1,
                 ct_unm=3,
                 n_samples=1000):
    sparsemode = issparse(adata.layers['raw'])
    if sparsemode:
        mat_sc = adata.layers['raw'].toarray()
    else:
        mat_sc = adata.layers['raw']
    lab_sc_sub = adata.obs.cellType
    sc_sub_dict = dict(zip(range(len(set(lab_sc_sub))), set(lab_sc_sub)))
    sc_sub_dict2 = dict((y, x) for x, y in sc_sub_dict.items())

    ct_list = list(sc_sub_dict.values())
    if nmix > ct_unm:
        target_num = ct_unm
    else:
        target_num = nmix

    sample_exp_ = []
    sample_pro_ = []
    for j in range(iter):
        sample_exp_z = []
        sample_prod_z = []
        for i in range(1, target_num + 1):
            sample_exp0, sample_pro0 = pseudo_num(adata, mat_sc, lab_sc_sub, i, nmix, ct_list)
            sample_exp_z = sample_exp_z + sample_exp0
            sample_prod_z = sample_prod_z + sample_pro0
        sample_exp_ = sample_exp_ + sample_exp_z
        sample_pro_ = sample_pro_ + sample_prod_z
    sample_exp = np.array(sample_exp_)
    sample_pro = np.array(sample_pro_)
    n_samples_ = n_samples - sample_pro.shape[0]
    print('\033[1;33m... Pseudo number ...\033[0m')
    print(f"Fixed number : {sample_pro.shape[0]}")
    print(f"Unfixed number : {n_samples_}")

    mat_sc, lab_sc_sub = Resampling_sc(adata, mat_sc, lab_sc_sub, ct_keys='cellType')

    lab_sc_num = [sc_sub_dict2[ii] for ii in lab_sc_sub]
    lab_sc_num = np.asarray(lab_sc_num, dtype='int')

    sc_mix_, lab_mix_ = random_mix(mat_sc, lab_sc_num, nmix=nmix, n_samples=n_samples_)

    sc_mix = np.concatenate((sample_exp, sc_mix_), axis=0)
    lab_mix = np.concatenate((sample_pro, lab_mix_), axis=0)

    n_name = ['mix_' + str(i) for i in range(sc_mix.shape[0])]
    sc_mix = pd.DataFrame(sc_mix, columns=adata.var_names, index=n_name)
    pseudo_sp = ad.AnnData(sc_mix)
    pseudo_sp.obsm['pseudo_label'] = pd.DataFrame(lab_mix, columns=list(sc_sub_dict.values()),
                                                  index=pseudo_sp.obs_names)

    return pseudo_sp


def random_mix(Xs, ys, nmix=5, n_samples=10000, seed=0):
    Xs_new, ys_new = [], []
    ys_ = to_categorical(ys)

    rstate = np.random.RandomState(seed)
    fraction_all = rstate.rand(n_samples, nmix)
    randindex_all = rstate.randint(len(Xs), size=(n_samples, nmix))

    for i in range(n_samples):
        fraction = fraction_all[i]
        fraction = fraction / np.sum(fraction)
        fraction = np.reshape(fraction, (nmix, 1))

        randindex = randindex_all[i]
        ymix = ys_[randindex]

        yy = np.sum(ymix * fraction, axis=0)

        XX = np.asarray(Xs[randindex]) * fraction
        XX_ = np.sum(XX, axis=0)

        ys_new.append(yy)
        Xs_new.append(XX_)

    Xs_new = np.asarray(Xs_new)
    ys_new = np.asarray(ys_new)

    return Xs_new, ys_new




def ad_to_df(adata, da_l = 'raw'):
    if isinstance(adata.layers[da_l], anndata._core.views.SparseCSRView) or \
            isinstance(adata.layers[da_l], scipy.sparse._csr.csr_matrix):
        if da_l == 'raw':
            df_ = pd.DataFrame(adata.layers['raw'].todense(), index=adata.obs_names, columns=adata.var_names)
        if da_l == 'normalized':
            df_ = pd.DataFrame(adata.layers['normalized'].todense(), index=adata.obs_names, columns=adata.var_names)
        if da_l == 'scaled':
            df_ = pd.DataFrame(adata.layers['scaled'].todense(), index=adata.obs_names, columns=adata.var_names)

    else:
        if da_l == 'raw':
            df_ = pd.DataFrame(adata.layers['raw'], index=adata.obs_names, columns=adata.var_names)
        if da_l == 'normalized':
            df_ = pd.DataFrame(adata.layers['normalized'], index=adata.obs_names, columns=adata.var_names)
        if da_l == 'scaled':
            df_ = pd.DataFrame(adata.layers['scaled'], index=adata.obs_names, columns=adata.var_names)

    return df_


def intersect(lst1, lst2):
    """
    Gets and returns intersection of two lists.

    """

    temp = set(lst2)
    lst3 = [value for value in lst1 if value in temp]
    return lst3

def normalize_A(adj, symmetric=True):
    if symmetric:
        d = sp.diags(np.power(np.array(adj.sum(axis=1)), -1/2).flatten(), 0)
        a_norm = d.dot(adj).dot(d).tocsr()
    else:
        d = sp.diags(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj).tocsr()
    return a_norm

def preprocess_A(adj, symmetric=True):
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize_A(adj, symmetric)
    return adj

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def categorical_crossentropy(preds, labels):
    return np.mean(-np.log(np.extract(labels, preds)))

def accuracy(preds, labels):
    return np.mean(np.equal(np.argmax(labels,axis=1), np.argmax(preds,axis=1)))

def evaluate_preds(preds, labels, indices):
    split_loss = list()
    split_acc = list()
    for y_split, idx_split in zip(labels, indices):
        split_loss.append(categorical_crossentropy(preds[idx_split],y_split[idx_split]))
        split_acc.append(accuracy(preds[idx_split], y_split[idx_split]))
    return split_loss, split_acc
