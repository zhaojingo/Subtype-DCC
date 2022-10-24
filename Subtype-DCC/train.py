import os
import numpy as np
import torch
import torchvision
import argparse
from modules import ae, network, contrastive_loss
from utils import yaml_config_hook, save_model
from torch.utils import data
from dataloader import *
import copy
import pandas as pd
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
 


def inference(loader, model, device):
    model.eval()
    cluster_vector = []
    feature_vector = []
    for step, x in enumerate(loader):
        x = x.float().to(device)
        with torch.no_grad():
            c,h = model.forward_cluster(x)
        c = c.detach()
        h = h.detach()
        cluster_vector.extend(c.cpu().detach().numpy())
        feature_vector.extend(h.cpu().detach().numpy())
    cluster_vector = np.array(cluster_vector)
    feature_vector = np.array(feature_vector)
    print("Features shape {}".format(feature_vector.shape))
    return cluster_vector,feature_vector


def train():
    loss_epoch = 0
    for step, x in enumerate(DL):
        optimizer.zero_grad()
        x_i = (x + torch.normal(0, 1, size=(x.shape[0], x.shape[1]))).float().to(device)
        x_j = (x + torch.normal(0, 1, size=(x.shape[0], x.shape[1]))).float().to(device)
        z_i, z_j, c_i, c_j = model(x_i, x_j)
        batch = x_i.shape[0]
        criterion_instance = contrastive_loss.DCL(temperature=0.5, weight_fn=None)
        criterion_cluster = contrastive_loss.ClusterLoss(cluster_number, args.cluster_temperature, loss_device).to(loss_device)
        loss_instance = criterion_instance(z_i, z_j)+criterion_instance(z_j, z_i)
        loss_cluster = criterion_cluster(c_i, c_j)
        loss = loss_instance + loss_cluster
        loss.backward()
        optimizer.step()
        loss_epoch += loss.item()
    return loss_epoch

def draw_fig(list,name,epoch):
    x1 = range(0, epoch+1)
    print(x1)
    y1 = list
    save_file = './results/' + name + 'Train_loss.png'
    plt.cla()
    plt.title('Train loss vs. epoch', fontsize=20)
    plt.plot(x1, y1, '.-')
    plt.xlabel('epoch', fontsize=20)
    plt.ylabel('Train loss', fontsize=20)
    plt.grid()
    plt.savefig(save_file)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cancer_type", '-c', type=str, default="BRCA")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--cluster_number', type=int,default=5)
    args = parser.parse_args()# 参数实例化
    cancer_dict = {'BRCA': 5, 'BLCA': 5, 'KIRC': 4,
                   'LUAD': 3, 'PAAD': 2, 'SKCM': 4,
                   'STAD': 3, 'UCEC': 4, 'UVM': 4, 'GBM': 2}
    
    cluster_number = cancer_dict[args.cancer_type]  # 按照癌症种类选择 
    print(cluster_number)

    config = yaml_config_hook("config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    model_path = './save/' + args.cancer_type
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    logger = SummaryWriter(log_dir="./log")
    
    #load data
    DL=get_feature(args.cancer_type, args.batch_size, True)
    
    # initialize model
    ae = ae.AE()
    model = network.Network(ae, args.feature_dim, cluster_number)
    model = model.to(device)
    
    # optimizer / loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_device = device
    
    # train
    loss=[]
    for epoch in range(args.start_epoch, args.epochs+1):
        lr = optimizer.param_groups[0]["lr"]
        loss_epoch = train()
        loss.append(loss_epoch)
        logger.add_scalar("train loss", loss_epoch)
        if epoch % 100 == 0:
            print(f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch}")
    save_model(model_path, model, optimizer, args.epochs)
    draw_fig(loss,args.cancer_type,epoch)
    
    #inference
    dataloader=get_feature(args.cancer_type,args.batch_size,False)
    
    # load model
    model = network.Network(ae, args.feature_dim, cluster_number)
    model_fp = os.path.join(model_path, "checkpoint_{}.tar".format(args.epochs))
    model.load_state_dict(torch.load(model_fp, map_location=device.type)['net'])
    model.to(device)

    print("### Creating features from model ###")
    X,h = inference(dataloader, model, device)
    output = pd.DataFrame(columns=['sample_name', 'dcc'])  # 建立新的DataFrame
    fea_tmp_file = '../subtype_file/fea/' + args.cancer_type + '/rna.fea'
    sample_name = list(pd.read_csv(fea_tmp_file).columns)[1:]
    output['sample_name'] = sample_name
    output['dcc'] = X+1
    out_file = './results/' + args.cancer_type +'.dcc'
    output.to_csv(out_file, index=False, sep='\t')
            
    fea_out_file = './results/' + args.cancer_type +'.fea'
    fea = pd.DataFrame(data=h, index=sample_name,
                               columns=map(lambda x: 'v' + str(x), range(h.shape[1])))
    fea.to_csv(fea_out_file, header=True, index=True, sep='\t')
