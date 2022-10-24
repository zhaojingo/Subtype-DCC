import torch
import numpy as np
import pandas as pd
from os.path import splitext, basename
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset

def get_feature(cancer_type, batch_size, training):

    fea_CN_file = '../subtype_file/fea/' + cancer_type + '/CN.fea'
    fea_CN = pd.read_csv(fea_CN_file, header=0, index_col=0, sep=',')

    fea_meth_file = '../subtype_file/fea/' + cancer_type + '/meth.fea'
    fea_meth = pd.read_csv(fea_meth_file, header=0, index_col=0, sep=',')

    fea_mirna_file = '../subtype_file/fea/' + cancer_type + '/miRNA.fea'
    fea_mirna = pd.read_csv(fea_mirna_file, header=0, index_col=0, sep=',')

    fea_rna_file = '../subtype_file/fea/' + cancer_type + '/rna.fea'
    fea_rna = pd.read_csv(fea_rna_file, header=0, index_col=0, sep=',')

    feature = np.concatenate((fea_CN, fea_meth, fea_mirna, fea_rna), axis=0).T
    
    minmaxscaler = MinMaxScaler()
    feature = minmaxscaler.fit_transform(feature)
    feature = torch.tensor(feature)
    
    dataloader = DataLoader(feature, batch_size=batch_size, shuffle=training)

    return dataloader



