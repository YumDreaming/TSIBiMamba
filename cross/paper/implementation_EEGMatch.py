# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 14:19:41 2022

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May 26 11:22:29 2022

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May 18 16:34:46 2022

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 17:20:11 2021

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 15:54:21 2021

@author: user
"""

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
from torch.nn import init
import os
import random
from torch.optim import RMSprop
from typing import Optional
import scipy.io as scio
from torch.optim.optimizer import Optimizer
from sklearn import preprocessing
from Adversarial_DG import TripleDomainAdversarialLoss
from model_EEGMatch import Domain_adaption_model,discriminator_DG
def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class StepwiseLR_GRL:
    """
    A lr_scheduler that update learning rate using the following schedule:

    .. math::
        \text{lr} = \text{init_lr} \times \text{lr_mult} \times (1+\gamma i)^{-p},

    where `i` is the iteration steps.

    Parameters:
        - **optimizer**: Optimizer
        - **init_lr** (float, optional): initial learning rate. Default: 0.01
        - **gamma** (float, optional): :math:`\gamma`. Default: 0.001
        - **decay_rate** (float, optional): :math:`p` . Default: 0.75
    """
    def __init__(self, optimizer: Optimizer, init_lr: Optional[float] = 0.01,
                 gamma: Optional[float] = 0.001, decay_rate: Optional[float] = 0.75,max_iter: Optional[float] = 1000):
        self.init_lr = init_lr
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.optimizer = optimizer
        self.iter_num = 0
        self.max_iter=max_iter
    def get_lr(self) -> float:
        lr = self.init_lr / (1.0 + self.gamma * (self.iter_num/self.max_iter)) ** (self.decay_rate)
        return lr

    def step(self):
        """Increase iteration number `i` by 1 and update learning rate in `optimizer`"""
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            if 'lr_mult' not in param_group:
                param_group['lr_mult'] = 1.
            param_group['lr'] = lr * param_group['lr_mult']
        self.iter_num += 1

def weigth_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)
        init.constant_(m.bias.data,0.3)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()    
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0,0.03)
#        torch.nn.init.kaiming_normal_(m.weight.data,a=0,mode='fan_in',nonlinearity='relu')
        m.bias.data.zero_()

def augmentation(feature_seqence,label_seqence,video_time,alpha=0.5):
    augment_data=[]
    augment_label=[]
    flag=0
    if len(feature_seqence)==0:
        return feature_seqence,label_seqence
    for i in range(len(video_time)):
        video_feature=feature_seqence[flag:flag+video_time[i],:]
        video_label=label_seqence[flag:flag+video_time[i],:]
        for j in range(len(video_feature)):
            index=np.random.randint(0,len(video_feature),2)
            weight_sequence=0.5*np.ones(2).reshape((1,2))
            lam=np.random.beta(alpha,alpha)
            weight_sequence[0,0]=lam
            weight_sequence[0,1]=1-lam
            augment_data.append(np.dot(weight_sequence,video_feature[index,:]))
            augment_label.append(video_label[j,:])
        flag+=video_time[i]
    # np.vstack(augment_data) → (W_i, D) -> (N, D)
    # np.vstack(augment_label) → (W_i, C) -> (N, C)
    return np.vstack(augment_data),np.vstack(augment_label)

def get_dataset_aug(test_id,session,video,parameter):
    alpha=parameter['alpha']
    path='F:\\zhourushuang\\transfer_learning\\feature_for_net_session'+str(session)+'_LDS_de'
    os.chdir(path)
    # 原始的“源域有标签”特征矩阵
    feature_list_source_labeled=[]
    # 原始的“源域有标签”标签向量（One - hot）
    label_list_source_labeled=[]
    # 原始的“源域无标签”特征矩阵
    feature_list_source_unlabeled=[]
    # 原始的“源域无标签”伪标签（One - hot，但训练时不使用）
    label_list_source_unlabeled=[]
    # 原始的“目标域”特征矩阵
    feature_list_target=[]
    # 原始的“目标域”标签向量（One - hot）
    label_list_target=[]
    # Mixup 增强后的“源域有标签”特征
    feature_list_source_labeled_aug=[]
    # Mixup 增强后的“源域有标签”标签
    label_list_source_labeled_aug=[]
    # Mixup 增强后的“源域无标签”特征
    feature_list_source_unlabeled_aug=[]
    # Mixup 增强后的“源域无标签”标签
    label_list_source_unlabeled_aug=[]
    # Mixup 增强后的“目标域”特征
    feature_list_target_aug=[]
    # Mixup 增强后的“目标域”标签
    label_list_target_aug=[]
    ## our label:0 negative, label:1 :neural,label:2:positive, seed original label: -1,0,1, our label= seed label+1
    min_max_scaler = preprocessing.MinMaxScaler(feature_range = (-1, 1))
    video_time=[235,233,206,238,185,195,237,216,265,237,235,233,235,238,206]
    index=0
    for info in os.listdir(path):
        domain = os.path.abspath(path)
        info_ = os.path.join(domain,info)
        if session==1:
            feature = scio.loadmat(info_)['dataset_session1']['feature'][0,0]
            label = scio.loadmat(info_)['dataset_session1']['label'][0,0]
        elif session==2:
            feature = scio.loadmat(info_)['dataset_session2']['feature'][0,0]
            label = scio.loadmat(info_)['dataset_session2']['label'][0,0]
        else:
            feature = scio.loadmat(info_)['dataset_session3']['feature'][0,0]
            label = scio.loadmat(info_)['dataset_session3']['label'][0,0]
        feature=min_max_scaler.fit_transform(feature).astype('float32')
        one_hot_label_mat=np.zeros((len(label),3))
        for i in range(len(label)):
            if label[i]==0:
                one_hot_label=[1,0,0]
                one_hot_label=np.hstack(one_hot_label).reshape(1,3)
                one_hot_label_mat[i,:]=one_hot_label
            if label[i]==1:
                one_hot_label=[0,1,0]
                one_hot_label=np.hstack(one_hot_label).reshape(1,3)
                one_hot_label_mat[i,:]=one_hot_label
            if label[i]==2:
                one_hot_label=[0,0,1]
                one_hot_label=np.hstack(one_hot_label).reshape(1,3)
                one_hot_label_mat[i,:]=one_hot_label
        if index!=test_id:
            ## source labeled data
            feature_labeled=feature[0:np.cumsum(video_time[0:video])[-1],:] # (M, 310)
            label_labeled=one_hot_label_mat[0:np.cumsum(video_time[0:video])[-1],:] # (M, 3)
            feature_list_source_labeled.append(feature_labeled)
            label_list_source_labeled.append(label_labeled)
            ## the origin EEG data for augmentation
            feature_labeled_origin,label_labeled_origin=np.copy(feature_labeled),np.copy(label_labeled)
            ## source labeled data and the augdata
            feature_labeled_aug,label_labeled_aug=augmentation(feature_labeled_origin,label_labeled_origin,video_time[0:video],alpha)
            feature_labeled=np.row_stack((feature_labeled,feature_labeled_aug)).astype('float32')
            label_labeled=np.row_stack((label_labeled,label_labeled_aug)).astype('float32')
            feature_list_source_labeled_aug.append(feature_labeled)
            label_list_source_labeled_aug.append(label_labeled)
            ## source unlabeled data
            feature_unlabeled=feature[np.cumsum(video_time[0:video])[-1]:len(feature),:]
            label_unlabeled=one_hot_label_mat[np.cumsum(video_time[0:video])[-1]:len(feature),:]
            feature_list_source_unlabeled.append(feature_unlabeled)
            label_list_source_unlabeled.append(label_unlabeled)
            ## source unlabeled data and aug data
            ## the origin EEG data for augmentation
            feature_unlabeled_origin,label_unlabeled_origin=np.copy(feature_unlabeled),np.copy(label_unlabeled)
            feature_unlabeled_aug,label_unlabeled_aug=augmentation(feature_unlabeled_origin,label_unlabeled_origin,video_time[video:len(video_time)],alpha)
            feature_unlabeled=np.row_stack((feature_unlabeled,feature_unlabeled_aug)).astype('float32')
            label_unlabeled=np.row_stack((label_unlabeled,label_unlabeled_aug)).astype('float32')
            feature_list_source_unlabeled_aug.append(feature_unlabeled)
            label_list_source_unlabeled_aug.append(label_unlabeled)         
        else:
            ## target labeled data
            feature_list_target.append(feature)
            label_list_target.append(one_hot_label_mat)
            label=one_hot_label_mat
            ## target labeled data and aug data
            feature_origin,label_origin=np.copy(feature),np.copy(label)
            feature_aug,label_aug=augmentation(feature_origin,label_origin,video_time,alpha)
            feature=np.row_stack((feature,feature_aug)).astype('float32')
            label=np.row_stack((label,label_aug)).astype('float32')
            feature_list_target_aug.append(feature)
            label_list_target_aug.append(label)
        index+=1

    source_feature_labeled,source_label_labeled=np.vstack(feature_list_source_labeled),np.vstack(label_list_source_labeled)
    source_feature_unlabeled,source_label_unlabeled=np.vstack(feature_list_source_unlabeled),np.vstack(label_list_source_unlabeled)
    target_feature=feature_list_target[0]
    target_label=label_list_target[0]
    
    source_feature_labeled_aug,source_label_labeled_aug=np.vstack(feature_list_source_labeled_aug),np.vstack(label_list_source_labeled_aug)
    source_feature_unlabeled_aug,source_label_unlabeled_aug=np.vstack(feature_list_source_unlabeled_aug),np.vstack(label_list_source_unlabeled_aug)
    target_feature_aug=feature_list_target_aug[0]
    target_label_aug=label_list_target_aug[0]
    
    target_set={'feature':target_feature,'label':target_label,'feature_aug':target_feature_aug,'label_aug':target_label_aug}
    source_set_labeled={'feature':source_feature_labeled,'label':source_label_labeled,'feature_aug':source_feature_labeled_aug,'label_aug':source_label_labeled_aug}
    source_set_unlabeled={'feature':source_feature_unlabeled,'label':source_label_unlabeled,'feature_aug':source_feature_unlabeled_aug,'label_aug':source_label_unlabeled_aug}
    return target_set,source_set_labeled,source_set_unlabeled
def get_generated_targets(model,x_s,x_un,x_t,labels_s,semi):
        with torch.no_grad():
            model.eval()
            un_predict =model.predict(x_un)  
            t_predict  =model.predict(x_t)
            if semi==1:
                X,Y=torch.cat((x_s,x_un)),torch.cat((labels_s,un_predict.to(labels_s)))
            else:
                X,Y=x_s,labels_s
            _,_,_,_,dist_matrix_t = model(X,x_t,Y.to(X))
            sim_matrix = model.get_cos_similarity_distance(Y.to(X))
            sim_matrix_target = model.get_cos_similarity_by_threshold(dist_matrix_t)
            return sim_matrix,sim_matrix_target,un_predict,t_predict

def checkpoint(model,checkpoint_PATH,flag):
    if flag=='load':
        model_CKPT = torch.load(checkpoint_PATH)
        model.load_state_dict(model_CKPT['state_dict'])
        model.P=model_CKPT['P']
        model.stored_mat=model_CKPT['stored_mat']
        model.cluster_label=model_CKPT['cluster_label']
        model.upper_threshold=model_CKPT['upper_threshold']
        model.lower_threshold=model_CKPT['lower_threshold']
        model.threshold=model_CKPT['threshold']
    elif flag=='save':
        torch.save({'P': model.P, 'stored_mat':model.stored_mat,'cluster_label':model.cluster_label,'threshold':model.threshold,
                    'upper_threshold':model.upper_threshold,'lower_threshold':model.lower_threshold,'state_dict': model.state_dict()},checkpoint_PATH)


def train_model(loader_train_labeled,loader_train_unlabeled, loader_test,model,dann_loss, optimizer,hidden_4,epoch,batch_size,parameter,threshold_update=True):
    # switch to train mode
    model.train()
    dann_loss.train()
    train_source_iter_labeled,train_source_iter_unlabeled,train_target_iter=enumerate(loader_train_labeled),enumerate(loader_train_unlabeled),enumerate(loader_test)
    T =2*3394//batch_size
    cls_loss_sum=0
    transfer_loss_sum=0
    if parameter['boost_type']=='linear':
        boost_factor=parameter['cluster_weight']*(epoch/model.max_iter)
    elif parameter['boost_type']=='exp':
        boost_factor=parameter['cluster_weight']*(2.0 / (1.0 + np.exp(-1 * epoch / model.max_iter))- 1)
    elif parameter['boost_type']=='constant':
        boost_factor=parameter['cluster_weight']
    for i in range(T):
        model.train()
        _,(x_s,labels_s) = next(train_source_iter_labeled)
        _,(x_un,_) = next(train_source_iter_unlabeled)
        _,(x_t,_) = next(train_target_iter)
        x_t=Variable(x_t.cuda())
        x_s,labels_s,x_un=Variable(x_s.cuda()), Variable(labels_s.cuda()),Variable(x_un.cuda()) 
        with torch.no_grad():
            estimated_sim_truth,estimated_sim_truth_target,un_predict,t_predict= get_generated_targets(model,x_s,x_un,x_t,labels_s,1)
            X,Y=torch.cat((x_s,x_un)),torch.cat((labels_s,un_predict.to(labels_s)))
        source_predict,feature_source_f,feature_target_f,sim_matrix,sim_matrix_target = model(X,x_t,Y.to(X))
        eta=0.00001
        bce_loss=-(torch.log(sim_matrix+eta)*estimated_sim_truth)-(1-estimated_sim_truth)*torch.log(1-sim_matrix+eta)
        bce_loss_target=-(torch.log(sim_matrix_target+eta)*estimated_sim_truth_target)-(1-estimated_sim_truth_target)*torch.log(1-sim_matrix_target+eta)
        indicator,nb_selected=model.compute_indicator(sim_matrix_target)
        cls_loss = torch.mean(bce_loss)
        cluster_loss=torch.sum(indicator*bce_loss_target)/nb_selected
        P_loss=torch.norm(torch.matmul(model.P.T,model.P)-torch.eye(hidden_4).cuda(),'fro')
        transfer_loss = dann_loss(feature_source_f[0:len(x_s),:],feature_target_f,feature_source_f[len(x_s):len(feature_source_f),:])
        cls_loss_sum+=cls_loss.data
        transfer_loss_sum+=transfer_loss.data
        loss = cls_loss+transfer_loss+0.01*P_loss+boost_factor*cluster_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('transfer_loss:',str(transfer_loss_sum/T))
    print('cls_loss:',str(cls_loss_sum/T))
    if threshold_update==True:
        model.update_threshold(epoch)  
    return cls_loss_sum.cpu().detach().numpy(),transfer_loss_sum.cpu().detach().numpy()

def train_and_test_GAN(test_id,max_iter,parameter,session,threshold_update=True):
    setup_seed(20)
    hidden_1,hidden_2,hidden_3,hidden_4,num_of_class,low_rank,upper_threshold,lower_threshold,temp=parameter['hidden_1'],parameter['hidden_2'], parameter['hidden_1'],parameter['hidden_2'],parameter['num_of_class'],parameter['low_rank'],parameter['upper_threshold'],parameter['lower_threshold'],parameter['temp']
    BATCH_SIZE = parameter['batch_size']
    video=parameter['video']
    target_set,source_set_labeled,source_set_unlabeled=get_dataset_aug(test_id,session,video,parameter)
    torch_dataset_source_labeled = Data.TensorDataset(torch.from_numpy(source_set_labeled['feature_aug']),torch.from_numpy(source_set_labeled['label_aug']))
    torch_dataset_source_unlabeled = Data.TensorDataset(torch.from_numpy(source_set_unlabeled['feature_aug']),torch.from_numpy(source_set_unlabeled['label_aug']))
    torch_dataset_test = Data.TensorDataset(torch.from_numpy(target_set['feature_aug']),torch.from_numpy(target_set['label_aug']))
    test_features,test_labels=torch.from_numpy(target_set['feature']),torch.from_numpy(target_set['label'])
    source_features,source_labels=torch.from_numpy(source_set_labeled['feature']),torch.from_numpy(source_set_labeled['label'])
    valid_features,valid_labels=torch.from_numpy(source_set_unlabeled['feature']),torch.from_numpy(source_set_unlabeled['label'])
    BATCH_SIZE_UN=BATCH_SIZE
    loader_train_labeled = Data.DataLoader(
            dataset=torch_dataset_source_labeled,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0
            )
    loader_train_unlabeled = Data.DataLoader(
        dataset=torch_dataset_source_unlabeled,
        batch_size=BATCH_SIZE_UN,
        shuffle=True,
        num_workers=0
        )
    BATCH_SIZE_Target=BATCH_SIZE
    loader_test = Data.DataLoader(
            dataset=torch_dataset_test,
            batch_size=BATCH_SIZE_Target,
            shuffle=True,
            num_workers=0
            ) 
    setup_seed(20)
    model=Domain_adaption_model(hidden_1,hidden_2,hidden_3,hidden_4,num_of_class,low_rank,max_iter,upper_threshold,lower_threshold,temp).cuda(0)
    model.apply(weigth_init)
    domain_discriminator = discriminator_DG(hidden_2).cuda()
    domain_discriminator.apply(weigth_init)
    dann_loss = TripleDomainAdversarialLoss(domain_discriminator).cuda()
    optimizer = RMSprop(model.get_parameters() + domain_discriminator.get_parameters(),lr=1e-3, weight_decay=1e-5)
    best_acc = 0.
    target_acc_list=np.zeros(max_iter)
    target_nmi_list=np.zeros(max_iter)
    source_acc_list=np.zeros(max_iter)
    source_nmi_list=np.zeros(max_iter)
    cls_loss_list=np.zeros(max_iter)
    transfer_loss_list=np.zeros(max_iter)
    for epoch in range(max_iter):
        # train for one epoch
        if len(np.unique(model.cluster_label))!=3:
            model.cluster_label=np.hstack([0,1,2])
        model.train()
        cls_loss_sum,transfer_loss_sum=train_model(loader_train_labeled,loader_train_unlabeled,loader_test,model,dann_loss,optimizer,hidden_4,epoch,BATCH_SIZE_Target,parameter,threshold_update)
        source_acc,source_nmi=model.cluster_label_update(source_features.cuda(),source_labels.cuda())
        model.eval()
        target_acc,target_nmi=model.target_domain_evaluation(test_features.cuda(),test_labels.cuda())
        valid_acc,valid_nmi=model.target_domain_evaluation(valid_features.cuda(),valid_labels.cuda())
        target_acc_list[epoch]=target_acc
        source_acc_list[epoch]=source_acc
        target_nmi_list[epoch]=target_nmi
        source_nmi_list[epoch]=source_nmi
        cls_loss_list[epoch]=cls_loss_sum
        transfer_loss_list[epoch]=transfer_loss_sum
        print('src:','epoch:',epoch,'acc=',source_acc,'nmi=',source_nmi)
        print('tar:','epoch:',epoch,'acc=',target_acc,'nmi=',target_nmi)
        print('val:','epoch:',epoch,'acc=',valid_acc,'nmi=',valid_nmi)
        best_acc = max(target_acc, best_acc)
        
    return best_acc,cls_loss_list,source_acc_list,source_nmi_list,target_acc_list,target_nmi_list,transfer_loss_list

def main(update_threshold,parameter,session):
    """
    :param update_threshold(bool): 是否在每个 epoch 后动态更新相似度阈值
    :param parameter(dict): 前面定义好的所有超参数
    :param session(int:1,2,3): 指定使用 SEED 的第几次会话数据
    :return:
    """
    # 固定随机种子
    setup_seed(20)
    # 训练轮数
    max_iter=1000
    # 最佳准确率数组
    best_acc_mat=np.zeros(15)
    # 损失曲线矩阵
    transfer_loss_curve=np.zeros((15,max_iter))
    cls_loss_curve=np.zeros((15,max_iter))
    # 准确率曲线
    source_acc_curve=np.zeros((15,max_iter))
    target_acc_curve=np.zeros((15,max_iter))
    # NMI 曲线
    source_nmi_curve=np.zeros((15,max_iter))
    target_nmi_curve=np.zeros((15,max_iter))
    """
    循环 15 次，对应 SEED 数据集的 15 名受试者
    每一轮 i 的取值分别是 0,1,2,…,14，代表“第 i+1 个受试者”在做“留一受试者做目标域”的实验折（Fold）
    """
    for i in range(15):
        """
        best_acc：该受试者的最佳目标域分类准确率（scalar）
        cls_loss_list：分类损失随 epoch 的变化，length 1000
        source_acc_list：源域聚类准确率随 epoch 变化
        source_nmi_list：源域 NMI 随 epoch 变化
        target_acc_list：目标域准确率随 epoch 变化
        target_nmi_list：目标域 NMI 随 epoch 变化
        transfer_loss_list：对抗/转移损失随 epoch 变化
        """
        best_acc,cls_loss_list,source_acc_list,source_nmi_list,target_acc_list,target_nmi_list,transfer_loss_list=train_and_test_GAN(i,max_iter,parameter,session,update_threshold)
        best_acc_mat[i]=best_acc
        source_acc_curve[i,:]=source_acc_list
        target_acc_curve[i,:]=target_acc_list
        source_nmi_curve[i,:]=source_nmi_list
        target_nmi_curve[i,:]=target_nmi_list
        transfer_loss_curve[i,:]=transfer_loss_list
        cls_loss_curve[i,:]=cls_loss_list
    return best_acc_mat,cls_loss_curve,transfer_loss_curve,source_acc_curve,source_nmi_curve,target_acc_curve,target_nmi_curve


"""
low_rank = 32: 在模型里先把特征从 hidden_2/hidden_4 维度映射到一个 32 维的子空间，再和原型做内积，这是一种降噪、降维的思想
upper_threshold = 0.9, lower_threshold = 0.5:
    · 相似度 ≥ upper_threshold（0.9） 的对，被标记为正对（标签 1）
    · 相似度 < lower_threshold（0.5） 的对，被标记为负对（标签 0）
cluster_weight = 2: 伪标签对比损失（cluster loss）的权重超参 γ
boost_type = 'linear': 决定 boost_factor 随 epoch 的变化方式('linear'：按 epoch 线性增大、'exp'：按指数曲线增大)
video = 3: 在 get_dataset_aug 里用来划分“源域有标签数据” vs “源域无标签数据”
temp = 0.9: 伪标签“锐化”时的温度系数，调用 sharpen(p, t),当 t<1（比如 0.9）时，会使分布更“尖锐”，增强模型对高置信度类别的信心
alpha = 0.5: EEG-Mixup 和后面 augmentation（Mixup）里的 β 分布参数，控制插值权重的抖动范围。
"""
parameter={'hidden_1':64,'hidden_2':64,'num_of_class':3,'cluster_weight':2,'low_rank':32,'upper_threshold':0.9,'lower_threshold':0.5,
           'boost_type':'linear','video':3,'temp':0.9,'batch_size':48,'alpha':0.5}
"""
best_acc_mat (15,) 每个受试者（Fold）的 最佳 目标域准确率
cls_loss_curve (15, 1000) 分类损失 随 1000 个 epoch 变化（每行对应一个受试者）
transfer_loss_curve (15, 1000) 域对抗/转移损失 随 epoch 变化
source_acc_curve (15, 1000) 源域聚类准确率 随 epoch 变化
source_nmi_curve (15, 1000) 源域 NMI 随 epoch 变化
target_acc_curve (15, 1000) 目标域准确率 随 epoch 变化
target_nmi_curve (15, 1000) 目标域 NMI 随 epoch 变化

NMI 指的是： Normalized Mutual Information（归一化互信息），
是一种用来评估聚类结果与“真实”标签（或另外一种聚类划分）之间一致性的指标，
常用于无监督聚类或半监督聚类任务中。
"""
best_acc_mat,cls_loss_curve,transfer_loss_curve,source_acc_curve,source_nmi_curve,target_acc_curve,target_nmi_curve=main(True,parameter,1)
result_list={'best_acc_mat':best_acc_mat,
            'cls_loss_curve':cls_loss_curve,
            'source_acc_curve':source_acc_curve,
            'source_nmi_curve':source_nmi_curve,
            'target_acc_curve':target_acc_curve,
            'target_nmi_curve':target_nmi_curve}