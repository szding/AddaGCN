from sklearn.neighbors import KDTree
from sklearn import preprocessing
import sklearn.neighbors
from sklearn.model_selection import train_test_split
from collections import defaultdict
import networkx as nx
from .utils import *

def Graph(adata_pseudo, adata_real, k_filter = 100, k_slg = 6):

    # pseudo
    data1 = ad_to_df(adata_pseudo, da_l = 'raw').T
    data1_norm = ad_to_df(adata_pseudo, da_l = 'normalized').T
    data1_scale = ad_to_df(adata_pseudo, da_l = 'scaled').T

    # real
    data2 = ad_to_df(adata_real, da_l='raw').T
    data2_norm = ad_to_df(adata_real, da_l='normalized').T
    data2_scale = ad_to_df(adata_real, da_l='scaled').T

    count_list_pseudo = [data1, data1]    # pseudo-pseudo
    norm_list_pseudo = [data1_norm, data1_norm]
    scale_list_pseudo = [data1_scale, data1_scale]

    count_list_pseudo_real = [data1, data2] # pseudo-real
    norm_list_pseudo_real = [data1_norm, data2_norm]
    scale_list_pseudo_real = [data1_scale, data2_scale]

    count_list_real = [data2, data2] #real-real
    norm_list_real = [data2_norm, data2_norm]
    scale_list_real = [data2_scale, data2_scale]

    # pseudo-pseudo
    link_pseudo = Link_graph(count_list_pseudo, norm_list_pseudo, scale_list_pseudo,
                             features=adata_real.var_names, k_filter=k_filter)
    # pseudo-real
    link_pseudo_real = Link_graph(count_list_pseudo_real, norm_list_pseudo_real, scale_list_pseudo_real,
                                  features=adata_real.var_names, k_filter=k_filter, num_c=30)
    # real-real
    link_real = Link_graph(count_list_real, norm_list_real, scale_list_real, features=adata_real.var_names, k_filter=k_filter)  

    # SLG
    link3 = Cal_Spatial_Net(adata_real.obsm["spatial"], k_cutoff=k_slg, model='KNN', verbose=True)  

    def dup(graph):
        graph.drop_duplicates(inplace=True)
        graph = graph.reset_index(drop=True)
        return graph

    graph0 = dup(link_pseudo[0].iloc[:, 0:2])   # pseudo-pseudo
    graph1 = dup(link_pseudo_real[0].iloc[:, 0:2])  # pseudo-real
    graph2 = dup(link_real[0].iloc[:, 0:2])  # real-real exp
    graph3 = dup(link3.iloc[:, 0:2])  # # real -real location

    graph4 = dup(pd.concat([graph2, graph3], axis=0, join='inner'))

    return graph0, graph1, graph2, graph3, graph4


def A_matrix(adata_pseudo, adata_real,  k_filter = 100, k_slg = 6):

    graph0, graph1, graph2, graph3, graph4 = Graph(adata_pseudo, adata_real,  k_filter, k_slg)

    Xc_pseudo = ad_to_df(adata_pseudo, da_l='raw')
    Xc_pseudo = Xc_pseudo.reset_index(drop=True)
    Xc_label = adata_pseudo.obsm['pseudo_label']
    Xc_label = Xc_label.reset_index(drop=True)

    Xt_real = ad_to_df(adata_real, da_l='raw')
    Xt_real = Xt_real.reset_index(drop=True)
    Xt_label = pd.DataFrame(np.zeros(shape=(Xt_real.shape[0], Xc_label.shape[1])), columns=Xc_label.columns)

    temD_train, temd_test, temL_train, teml_test = train_test_split(
        Xc_pseudo, Xc_label, test_size=0.1, random_state=1)
    temd_train, temd_val, teml_train, teml_val = train_test_split(
        temD_train, temL_train, test_size=0.1, random_state=1)

    train = pd.concat([temd_train, Xt_real])  # train data
    lab_train = pd.concat([teml_train, Xt_label])  # train label

    X = pd.concat([pd.concat([train, temd_val]), temd_test])
    L = pd.concat([pd.concat([lab_train, teml_val]), teml_test])

    M = len(temd_train)
    idx_train = range(M)
    idx_pred = range(M, len(lab_train))
    idx_val = range(len(lab_train), len(lab_train) + len(teml_val))
    idx_test = range(len(lab_train) + len(teml_val), len(lab_train) + len(teml_val) + len(teml_test))

    train_mask = sample_mask(idx_train, L.shape[0])
    pred_mask = sample_mask(idx_pred, L.shape[0])
    val_mask = sample_mask(idx_val, L.shape[0])
    test_mask = sample_mask(idx_test, L.shape[0])
    pseudo_mask = np.array([bool(x) for x in 1 - pred_mask])

    y_train = np.zeros(np.array(L).shape)
    y_val = np.zeros(np.array(L).shape)
    y_test = np.zeros(np.array(L).shape)
    y_train[train_mask, :] = np.array(L)[train_mask, :]
    y_val[val_mask, :] = np.array(L)[val_mask, :]
    y_test[test_mask, :] = np.array(L)[test_mask, :]

    # graph
    fake1 = np.array([-1] * len(Xt_real.index))
    index1 = np.concatenate((teml_train.index, fake1, teml_val.index, teml_test.index)).flatten()

    fake2 = np.array([-1] * len(teml_train))
    fake3 = np.array([-1] * (len(teml_val) + len(teml_test)))
    find1 = np.concatenate((fake2, Xt_real.index, fake3)).flatten()

    # # pseudo-pseudo
    row0 = [np.where(index1 == graph0.iloc[i, 1])[0][0] for i in range(len(graph0))]
    col0 = [np.where(index1 == graph0.iloc[i, 0])[0][0] for i in range(len(graph0))]

    # pseudo-real
    # i = 1
    row1 = [np.where(find1 == graph1.iloc[i, 1])[0][0] for i in range(len(graph1))] #real
    col1 = [np.where(index1 == graph1.iloc[i, 0])[0][0] for i in range(len(graph1))]  # pseudo

    # real-real
    row2 = [np.where(find1 == graph4.iloc[i, 1])[0][0] for i in range(len(graph4))]
    col2 = [np.where(find1 == graph4.iloc[i, 0])[0][0] for i in range(len(graph4))]

    # total  A
    print('... Calculating A matrix ...')
    row = row0 + row1 + row2
    col = col0 + col1 + col2

    adj = defaultdict(list)  
    for i in range(adata_pseudo.shape[0] + adata_real.shape[0]): 
        adj[i].append(i)
    for i in range(len(row)):
        adj[row[i]].append(col[i])
        adj[col[i]].append(row[i])
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(adj))  
    A = preprocess_A(adj)  
    A = normalize_A(A)

    X = np.asmatrix(np.array(X), dtype='float64')
    X = np.divide(X, X.sum(1).reshape(-1, 1), dtype='float64')  
    X_label = np.array(L)
    print(type(X), type(A), type(X_label))

    Y_real = [y_train, y_val, y_test]
    idx = [idx_train, idx_pred, idx_val, idx_test]
    mask_ = [train_mask, pred_mask, pseudo_mask]

    return X, X_label, A, mask_, idx, Y_real

def SVD(mat, num_cc):
    U, s, V = np.linalg.svd(mat)
    d = s[0:int(num_cc)]
    u = U[:, 0:int(num_cc)]
    v = V[0:int(num_cc), :].transpose()
    return u, v, d

def Scale(x):
    y = preprocessing.scale(x)
    return y

def l2norm(mat):
    stat = np.sqrt(np.sum(mat**2, axis=1))
    cols = mat.columns
    mat[cols] = mat[cols].div(stat, axis=0)
    mat[np.isinf(mat)] = 0
    return mat

def topGenes(Loadings, dim, numG):
    data = Loadings.iloc[:, dim]
    num = np.round(numG / 2).astype('int')
    data1 = data.sort_values(ascending=False)
    data2 = data.sort_values(ascending=True)
    posG = np.array(data1.index[0:num])
    negG = np.array(data2.index[0:num])
    topG = np.concatenate((posG, negG))
    return topG

def TopGenes(Loadings, dims, DimGenes, maxGenes):
    maxG = max(len(dims) * 2, maxGenes)
    gens = [None] * DimGenes
    idx = -1
    for i in range(1, DimGenes + 1):
        idx = idx + 1
        selg = []
        for j in dims:
            selg.extend(set(topGenes(Loadings, dim=j, numG=i)))
        gens[idx] = set(selg)
    lens = np.array([len(i) for i in gens])
    lens = lens[lens < maxG]
    maxPer = np.where(lens == np.max(lens))[0][0] + 1
    selg = []
    for j in dims:
        selg.extend(set(topGenes(Loadings, dim=j, numG=maxPer)))
    selgene = np.array(list(set(selg)), dtype=object)
    return (selgene)


def embed(data1, data2, num_cc=20):
    random.seed(123)
    object1 = Scale(data1)
    object2 = Scale(data2)
    mat3 = np.matmul(np.matrix(object1).transpose(), np.matrix(object2))
    a = SVD(mat=mat3, num_cc=int(num_cc))
    embeds_data = np.concatenate((a[0], a[1])) 
    ind = np.where(
        [embeds_data[:, col][0] < 0 for col in range(embeds_data.shape[1])])[0]
    embeds_data[:, ind] = embeds_data[:, ind] * (-1)

    embeds_data = pd.DataFrame(embeds_data)
    embeds_data.index = np.concatenate(
        (np.array(data1.columns), np.array(data2.columns)))
    embeds_data.columns = ['D_' + str(i) for i in range(num_cc)] 
    d = a[2]
    return embeds_data, d

def Embed(data_use1, data_use2, features, count_names, num_cc):
    #
    features = checkFeature(data_use1, features) 
    features = checkFeature(data_use2, features) 
    data1 = data_use1.loc[features, ] 
    data2 = data_use2.loc[features, ] 

    embed_results = embed(data1=data1, data2=data2, num_cc=num_cc)
    cell_embeddings = np.matrix(embed_results[0])
    combined_data = data1.merge(data2,
                                left_index=True,
                                right_index=True,
                                how='inner') 
    count_names = intersect(count_names, features)
    new_data1 = combined_data.loc[count_names, ].dropna()  

    loadings = pd.DataFrame(np.matmul(np.matrix(new_data1), cell_embeddings))
    loadings.index = new_data1.index
    return embed_results, loadings


def checkFeature(data_use, features):
    data1 = data_use.loc[features, ]
    feature_var = data1.var(1)
    Var_features = np.array(features)[np.where(feature_var != 0)[0]]
    return Var_features

def kNN(data, k, query=None): 

    dim = data.shape[0]
    if dim < k:
        k = int(dim/2)
    data[data.isnull().T.any()] = 0

    tree = KDTree(data) #最近邻检索
    if query is None:
        query = data
    query[query.isnull().T.any()] = 0
    dist, ind = tree.query(query, k)
    return dist, ind


def KNN(cell_embedding, spots1, spots2, k):
    embedding_spots1 = cell_embedding.loc[spots1, ]
    embedding_spots2 = cell_embedding.loc[spots2, ]
    nnaa = kNN(embedding_spots1, k=k + 1)
    nnbb = kNN(embedding_spots2, k=k + 1)
    nnab = kNN(data=embedding_spots2, k=k, query=embedding_spots1)
    nnba = kNN(data=embedding_spots1, k=k, query=embedding_spots2)
    return nnaa, nnab, nnba, nnbb, spots1, spots2

def MNN(neighbors, colnames, num):
    max_nn = np.array([neighbors[1][1].shape[1], neighbors[2][1].shape[1]]) 
    if ((num > max_nn).any()):
        num = np.min(max_nn)
    spots1 = colnames 
    spots2 = colnames
    nn_spots1 = neighbors[4] 
    nn_spots2 = neighbors[5] 
    cell1_index = [
        list(nn_spots1).index(i) for i in spots1 if (nn_spots1 == i).any()
    ]
    cell2_index = [
        list(nn_spots2).index(i) for i in spots2 if (nn_spots2 == i).any()
    ]
    ncell = range(neighbors[1][1].shape[0])
    ncell = np.array(ncell)[np.in1d(ncell, cell1_index)]
    
    mnn_cell1 = [None] * (len(ncell) * 5)
    mnn_cell2 = [None] * (len(ncell) * 5)
    idx = -1

    for cell in ncell:
        neighbors_ab = neighbors[1][1][cell, 0:5]
        mutual_neighbors = np.where(
            neighbors[2][1][neighbors_ab, 0:5] == cell)[0]
        for i in neighbors_ab[mutual_neighbors]:
            idx = idx + 1
            mnn_cell1[idx] = cell
            mnn_cell2[idx] = i
    mnn_cell1 = mnn_cell1[0:(idx + 1)]
    mnn_cell2 = mnn_cell2[0:(idx + 1)]
    import pandas as pd
    mnns = pd.DataFrame(np.column_stack((mnn_cell1, mnn_cell2)))
    mnns.columns = ['spot1', 'spot2']
    return mnns

def filterEdge(edges, neighbors, mats, features, k_filter):
    nn_spots1 = neighbors[4] 
    nn_spots2 = neighbors[5] 
    mat1 = mats.loc[features, nn_spots1].transpose() 
    mat2 = mats.loc[features, nn_spots2].transpose() 
    cn_data1 = l2norm(mat1)
    cn_data2 = l2norm(mat2)

    nn = kNN(data=cn_data2.loc[nn_spots2, ], 
             query=cn_data1.loc[nn_spots1, ], 
             k=k_filter) 
    position = [
        np.where(
            edges.loc[:, "spot2"][x] == nn[1][edges.loc[:, 'spot1'][x], ])[0] 
        for x in range(edges.shape[0])
    ]
    nps = np.concatenate(position, axis=0)
    fedge = edges.iloc[nps, ]
    return (fedge)

def Link_graph(count_list,
               norm_list,
               scale_list,
               features,
               k_filter=200,
               num_c = 30):
    all_edges = []
    i = 0
    j = 1
    counts1 = count_list[i].copy()
    counts2 = count_list[j].copy()
    counts1.columns = counts1.columns+'new'
    norm_data1 = norm_list[i].copy()
    norm_data2 = norm_list[j].copy()
    norm_data1.columns = norm_data1.columns + 'new'
    scale_data1 = scale_list[i].copy()
    scale_data2 = scale_list[j].copy()
    scale_data1.columns = scale_data1.columns + 'new'
    rowname = counts1.index 
    cell_embedding, loading = Embed(data_use1=scale_data1,
                                    data_use2=scale_data2,
                                    features=features,
                                    count_names=rowname,
                                    num_cc=num_c)  
    norm_embedding = l2norm(mat=cell_embedding[0])
    spots1 = counts1.columns
    spots2 = counts2.columns
    neighbor = KNN(cell_embedding=norm_embedding,
                   spots1=spots1,
                   spots2=spots2,
                   k=30)
    mnn_edges = MNN(neighbors=neighbor,
                    colnames=cell_embedding[0].index, 
                    num=5)
    select_genes = TopGenes(Loadings=loading,
                            dims=range(num_c),  
                            DimGenes=100,
                            maxGenes=200) 
    Mat = pd.concat([norm_data1, norm_data2], axis=1) 
    
    final_edges = filterEdge(edges=mnn_edges,
                             neighbors=neighbor,
                             mats=Mat,
                             features=select_genes,
                             k_filter=k_filter)   
    final_edges['Dataset1'] = [i + 1] * final_edges.shape[0]
    final_edges['Dataset2'] = [j + 1] * final_edges.shape[0]
    all_edges.append(final_edges)

    return all_edges

def Cal_Spatial_Net(sp_loc, rad_cutoff=None, k_cutoff=None, model='Radius', verbose=True):

    assert (model in ['Radius', 'KNN'])
    if verbose:
        print('------Calculating spatial graph...')
    coor = pd.DataFrame(sp_loc)  
    coor.columns = ['imagerow', 'imagecol']

    if model == 'Radius':
        nbrs = sklearn.neighbors.NearestNeighbors(radius=rad_cutoff).fit(coor)
        distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it] * indices[it].shape[0], indices[it], distances[it])))

    if model == 'KNN':
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k_cutoff + 1).fit(coor)
        distances, indices = nbrs.kneighbors(coor)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it] * indices.shape[1], indices[it, :], distances[it, :])))

    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['spot1', 'spot2', 'Distance']

    Spatial_Net = KNN_df.copy()
    Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance'] > 0,]

    del Spatial_Net["Distance"]
    Spatial_Net['Dataset1'] = [1] * Spatial_Net.shape[0]
    Spatial_Net['Dataset2'] = [2] * Spatial_Net.shape[0]
    return Spatial_Net


