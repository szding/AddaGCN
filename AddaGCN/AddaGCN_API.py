import warnings
warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.WARNING)
from .gutils import A_matrix
from .utils import pseudo_build,preprocess
from .train_model import train

import pandas as pd
import numpy as np


class AddaGCN_API:
    def __init__(self, adata_sp, adata_sc, num_markers=20,HVG_method = 'wilcoxon',
                 n_samples=1000, nmix=8, k_filter = 100, k_slg = 6,
                 sample_iter  = 1,ct_unm_fix = 3,
                 celltype = 'cellType', sp= 'spatial'):

        self.num_markers = num_markers
        self.celltype = celltype
        self.sp = sp


        print('... marker gene ...')
        adata_sc, inter_genes_ = preprocess(adata_sc, sc_=True, num_markers=self.num_markers,
                                            celltype=self.celltype, HVG_method = HVG_method)
        inter_genes = [val for val in inter_genes_ if val in adata_sp.var.index]
        self.adata_sc = adata_sc[:, inter_genes]
        self.adata_sp = adata_sp[:, inter_genes]


        print('... pseudo ...')
        pseudo_sp = pseudo_build(self.adata_sc, nmix=nmix, iter=sample_iter, ct_unm=ct_unm_fix, n_samples=n_samples)


        self.adata_sp = preprocess(self.adata_sp, sp_=True, celltype=self.celltype)
        self.adata_pseudo = preprocess(pseudo_sp, sp_= True)


        self.lab_sc_sub = self.adata_sc.obs.cellType
        del self.adata_sc


        print('... Graph ...')
        self.X, self.X_label, self.A, self.mask_, self.idx, self.Y_real = A_matrix(self.adata_pseudo, adata_real=self.adata_sp,
                                                                                    k_filter = k_filter, k_slg = k_slg)


    def Train_(self, loss_weight = [0.1,1], epochs = 2500, lr = [0.001, 0.001], initial_train_epochs=50, emb_dim=64,
               initial_train = True,enable_dann = True):

        pred, embedding = train(X=self.X, A=self.A, y=self.X_label, yt=self.Y_real, idx=self.idx,
                                emb_dim=emb_dim,  enable_dann=enable_dann,
                                n_iterations=epochs, loss_weight = loss_weight,  alpha_lr=lr,
                                initial_train=initial_train,
                                initial_train_epochs=initial_train_epochs)

        sc_sub_dict = dict(zip(range(len(set(self.lab_sc_sub))), set(self.lab_sc_sub)))
        cell_type = list(sc_sub_dict.values())

        self.pred_real = pd.DataFrame(pred[self.mask_[1], :], index=self.adata_sp.obs_names, columns=cell_type)
        emb_real = pd.DataFrame(embedding[self.mask_[1], :], index=self.adata_sp.obs_names,
                                columns=["emb" + str(i) for i in range(1, embedding.shape[1] + 1)])
        self.adata_sp.obs[self.pred_real.columns] = self.pred_real
        self.adata_sp.uns['celltypes'] = list(self.pred_real.columns)
        self.adata_sp.obsm['emb'] = emb_real

        pseudo_mask = np.array([not elem for elem in self.mask_[1]])
        pred_pseudo = pd.DataFrame(pred[pseudo_mask, :], index=self.adata_pseudo.obs_names,
                                   columns=cell_type)
        emb_pseudo = pd.DataFrame(embedding[pseudo_mask, :], index=self.adata_pseudo.obs_names,
                                  columns=["emb" + str(i) for i in range(1, embedding.shape[1] + 1)])
        self.adata_pseudo.obsm['pred_label'] = pred_pseudo
        self.adata_pseudo.obsm['emb'] = emb_pseudo




