# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from PIL import Image
import json
import numpy as np
import torchvision.transforms as transforms
import os
from torch.utils.data import Dataset
import random
import copy
import cv2
from .transforms import *
import matplotlib.pyplot as plt
import pandas as pd
from skimage import transform as skTr
import scipy.io


identity = lambda x:x


class SetDataset:
    def __init__(self, data_file, n_way, n_query, A_attr_per_class, image_size, transform, simple_mode=False, is_train=True, n_episode=100, debug_mode=False, return_label=False):
        with open(data_file, 'r') as f:
            self.meta = json.load(f)
        
        
        self.n_episode=n_episode
        self.attr_list = np.unique(self.meta['attribute_no']).tolist()
        #self.attr_to_idx = {item:key for key, item in enumerate(self.attr_list)}
        self.A_attr_per_class = A_attr_per_class
        self.CA_mat = torch.from_numpy(pd.read_table('/home/user/tzuyin/ra_project/dataset/CUB_200_2011/attributes/class_attribute_labels_continuous.txt',sep=' ',header=None).values/100.)[...,self.attr_list]
        self.CA_mat = torch.where(self.CA_mat>=0.5, 1., 0.)
        self.n_way = n_way
        self.simple_mode = simple_mode
        self.debug_mode=debug_mode
        self.cl_list = np.unique(self.meta['image_labels']).tolist()
        self.selection_prior_based = (self.CA_mat[self.cl_list]@((1/torch.sum(self.CA_mat,0))[...,None])).reshape([-1])
        self.return_label=return_label
        self.tr_dict={'_shape':0,
                      '_color':1,
                      '_pattern':2}
        
        attribute_name = self.attribute_name = pd.read_table('/home/user/tzuyin/ra_project/dataset/CUB_200_2011/attributes.txt',sep=' ',header=None).values[...,1]
        
        part_name = {'back':[0],
                    'beak':[1],
                    'belly':[2],
                    'breast':[3],
                    'crown':[4],
                    'forehead':[5],
                    'eye':[6,10],
                    'head':[4,5,6,10],
                    'leg':[7,11],
                    'wing':[8,12],
                    'nape':[9],
                    'tail':[13],
                    'throat':[14],
                    'upperparts':[0,3,4,5,6,10,14],
                    'underparts':[2,8,12,13],
                    'primary':[0,2,3,4,5,6,8,9,10,12,13,14]}
          
        attribute_part_label = np.zeros([312,15])
    
        for key_attr, attr_n in enumerate(attribute_name):
            for key_part, part_n in enumerate(part_name):

                if('head' in attr_n and 'forehead' not in attr_n):
                    for counter in part_name['head']:
                        attribute_part_label[key_attr, counter]=1
                
                elif('forehead' in attr_n):
                    for counter in part_name['forehead']:
                        attribute_part_label[key_attr, counter]=1
                        
                elif(part_n in attr_n):
                    for counter in part_name[part_n]:
                        attribute_part_label[key_attr, counter]=1
            

        self.attr_sub_meta = {}
        for attr in self.attr_list:
            self.attr_sub_meta[attr] = []

        for x,l,A,z in zip(self.meta['image_names'],self.meta['image_labels'],self.meta['image_attributes'], self.meta['part']):
            for y in A:
                idx = np.where(attribute_part_label[y]==1)[0][0]
                if(np.array(z)[idx][2]!=0):
                    self.attr_sub_meta[y].append({'path':x, 'part': z, 'mask': attribute_part_label[y], 'class':l})


        self.sub_attr_dataloader = [] 
        sub_data_loader_params = dict(batch_size = 3,
                                  shuffle = True,
                                  num_workers = 0, #use main thread only or may receive multiple batches
                                  pin_memory = False)  

        for attr in self.attr_list:
            print(attr)
            sub_dataset = SubAttributeDataset(self.attr_sub_meta[attr], attr, image_size, transform = transform, is_train = is_train )
            self.sub_attr_dataloader += [torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params)]


        self.sub_meta = {}
        for cl in self.cl_list:
            self.sub_meta[cl] = []

        if 'part' in self.meta:
            for x,y,z,attrs in zip(self.meta['image_names'],self.meta['image_labels'], self.meta['part'],self.meta['image_attributes']):
                self.sub_meta[y].append({'path':x, 'part': z, 'image_attributes':attrs})
        else:
            for x,y in zip(self.meta['image_names'],self.meta['image_labels']):
                self.sub_meta[y].append({'path':x})

        self.sub_dataloader = [] 
        sub_data_loader_params = dict(batch_size = n_query,
                                  shuffle = True,
                                  num_workers = 0, #use main thread only or may receive multiple batches
                                  pin_memory = False)   

        if 'part' in self.meta:
            for cl in self.cl_list:
                sub_dataset = SubPartsDataset(self.sub_meta[cl], cl, image_size, transform = transform, is_train = is_train )
                self.sub_dataloader.append( torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params) )
        else:     
            for cl in self.cl_list:
                sub_dataset = SubDataset(self.sub_meta[cl], cl, transform = transform )
                self.sub_dataloader.append( torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params) )


    def __getitem__(self,i):

        #C_selected_idx = torch.randperm(len(self.cl_list))[:self.n_way]
        C_selected_idx = torch.multinomial(self.selection_prior_based,self.n_way)
        C_selected = np.array(self.cl_list)[C_selected_idx.numpy().tolist()]

        #C_selected = random_choice_class(n_way=5)
        Attr_count = torch.sum(self.CA_mat[C_selected],0)
        Attr_candidate = torch.where(Attr_count!=0)
        
        CA_priority_mat = ((self.CA_mat[C_selected].permute(1,0)[Attr_candidate].permute(1,0))*(Attr_count[None].permute(1,0)[Attr_candidate].permute(1,0))).clone()
        CA_priority_mat[torch.where(CA_priority_mat!=0)] = 1/(CA_priority_mat[torch.where(CA_priority_mat!=0)])**2
        #CA_priority_mat[torch.where(CA_priority_mat==0)]=10000
        #CA_priority_mat[torch.where(CA_priority_mat>3)]=10000
        #CA_priority_mat = torch.argsort(CA_priority_mat,-1)
        #selected_attr_ori = Attr_candidate[0][CA_priority_mat[...,:self.A_attr_per_class]]
        sample_attr_idx = torch.multinomial(CA_priority_mat,self.A_attr_per_class)

        selected_attr_ori = Attr_candidate[0][sample_attr_idx]
        selected_attr = torch.unique(selected_attr_ori.reshape([-1]))
        
        ##check if over sample
           
        proxy_mat = self.CA_mat[C_selected].permute(1,0)[selected_attr].permute(1,0)
        
        sup_img=[]
        sup_part=[]
        sup_mask=[]
        for A in selected_attr:
            
            self.sub_attr_dataloader[A.item()].dataset.ban_class(C_selected)
            
            check_len = len(self.sub_attr_dataloader[A.item()])
        
            if(check_len<3):
                _a, _b, _c = [], [], []
                for counter in range(3):
                    _1, _2, _3 = next(iter(self.sub_attr_dataloader[A.item()]))
                    _a.append(_1)
                    _b.append(_2)
                    _c.append(_3)
                _1, _2, _3 = torch.cat(_a,0)[:3],torch.cat(_b,0)[:3],torch.cat(_c,0)[:3]
                
                
            else:
                _1, _2, _3 = next(iter(self.sub_attr_dataloader[A.item()]))
            sup_img+=[_1]
            sup_part+=[_2]
            sup_mask+=[_3]
            
            self.sub_attr_dataloader[A.item()].dataset.unban_class()
            
        support = [torch.cat(sup_img,0), torch.cat(sup_part,0), torch.cat(sup_mask,0)]

        query_img=[]
        query_target=[]
        query_part=[]
        
        if(self.simple_mode):
            for C, SA in zip(C_selected_idx, selected_attr_ori.numpy().astype(int)):
                self.sub_dataloader[C.item()].dataset.attribute_requirement(np.array(self.attr_list)[SA])
                _1, _2, _3 = next(iter(self.sub_dataloader[C.item()]))
                query_img+=[_1]
                query_target+=[_2]
                query_part+=[_3]
        else:
            for C in C_selected_idx:
                _1, _2, _3 = next(iter(self.sub_dataloader[C.item()]))
                query_img+=[_1]
                query_target+=[_2]
                query_part+=[_3]
        query = [torch.cat(query_img,0), torch.cat(query_target,0), torch.cat(query_part,0)]
        
               
        if(self.debug_mode==True):
            return support, query, proxy_mat, C_selected, np.array(self.attr_list)[selected_attr], selected_attr
        
        else:
            if(self.return_label==True):
                attr_type_label = []
                AN = self.attribute_name[np.array(self.attr_list)[selected_attr]]
                for attr_name in AN:
                    for N in self.tr_dict:
                        if(N in attr_name):
                            attr_type_label += [self.tr_dict[N]]
                            
                assert len(attr_type_label) == len(AN), 'length of attr_type_label:'+str(attr_type_label)+' / length of AN:'+str(AN)+'\n Error names:'+str(AN)
                
                return support, query, proxy_mat, torch.from_numpy(np.array(attr_type_label).astype(np.int64)).long()
        
            else:
                return support, query, proxy_mat

    def __len__(self):
        return self.n_episode


class AttributeDataset(SetDataset):
    def __init__(self, data_file, n_way, n_query, A_attr_per_class, image_size, transform, simple_mode=False, is_train=True, n_episode=100, debug_mode=False, return_label=False):
        super().__init__(data_file, n_way, n_query, A_attr_per_class, image_size, transform, simple_mode=False, is_train=True, n_episode=100, debug_mode=False)
        self.return_label=return_label
        self.tr_dict={'shape':0,
                      'color':1,
                      'pattern':2}
    def __getitem__(self,i):
        check_len = len(self.sub_attr_dataloader[i])
        
        if(check_len<3):
            _a, _b, _c = [], [], []
            for counter in range(3):
                _1, _2, _3 = next(iter(self.sub_attr_dataloader[i]))
                _a.append(_1)
                _b.append(_2)
                _c.append(_3)
            ret = tuple([torch.cat(_a,0)[:3],torch.cat(_b,0)[:3],torch.cat(_c,0)[:3]])
            
            
        else:
            ret = next(iter(self.sub_attr_dataloader[i]))
            
        
        if(self.return_label==True):
            
            attr_type_label = -1
            attr_name = self.attribute_name[self.attr_list[i]]
            for N in self.tr_dict:
                if(N in attr_name):
                    attr_type_label = self.tr_dict[N]
            assert attr_type_label != -1
            return ret, attr_name, self.attr_list[i], attr_type_label
        
        else:
            return ret, self.attribute_name[self.attr_list[i]], self.attr_list[i]

    def __len__(self):
        return len(self.sub_attr_dataloader)




class CUB_dataloader(torch.utils.data.Dataset):
    def __init__(self, split, transform, size=84, renumber=False, dataset_root_path='/home/user/tzuyin/ra_project/dataset'):
        self.corresponding_path = pd.read_table(dataset_root_path+'/CUB_200_2011/images.txt',sep=' ',header=None).values[...,1:]
        self.image_path = np.stack([dataset_root_path+'/CUB_200_2011/images/'+item for item in self.corresponding_path],axis=0)
        self.attribute = pd.read_table(dataset_root_path+'/CUB_200_2011/attributes/class_attribute_labels_continuous.txt',sep=' ',header=None).values/100.
        self.img_label = pd.read_table(dataset_root_path+'/CUB_200_2011/image_class_labels.txt',sep=' ',header=None).values[...,1]-1        
        self.attribute_name = pd.read_table(dataset_root_path+'/CUB_200_2011/attributes.txt',sep=' ',header=None).values[...,1]
        self.object_xywh = pd.read_table(dataset_root_path+'/CUB_200_2011/bounding_boxes.txt',sep=' ',header=None).values[...,1:]
        self.split_mat = scipy.io.loadmat(dataset_root_path+'/CUB_200_2011/att_splits.mat')  
        self.transform = transform
        self.renumber = renumber
        self.size = size
        if(split=='train'):
            self.img_id_list = self.split_mat['train_loc']-1
        elif(split=='train_val'):
            self.img_id_list = self.split_mat['trainval_loc']-1
        elif(split=='val'):
            self.img_id_list = self.split_mat['val_loc']-1
        elif(split=='test_seen'):
            self.img_id_list = self.split_mat['test_seen_loc']-1
        elif(split=='test_unseen'):
            self.img_id_list = self.split_mat['test_unseen_loc']-1
        elif(split=='all'):
            self.img_id_list = np.array([[i] for i in range(self.img_label.shape[0])])
        else:
            raise "The selection must in the set of {'train','train_val','val','test_seen','test_unseen'}"

        self.renumbered_label = np.sort(np.unique(self.img_label[self.img_id_list]))
        self.label_transform_list = (np.ones([200])*-1).astype(int)
        for key, value in enumerate(self.renumbered_label):
            self.label_transform_list[value] = key
        
        try:
            self.image_attribute_labels = pd.read_csv(dataset_root_path+'/CUB_200_2011/attributes/processed_image_attribute_labels.csv',header=None).values
        except:
            
            df = pd.read_table(dataset_root_path+'/CUB_200_2011/attributes/image_attribute_labels.txt',header=None)
            pd.DataFrame(pd.DataFrame(df[0].str.split(' ',expand=True)).values[...,2].astype(float).reshape([-1,312])).to_csv('/home/user/tzuyin/ra_project/dataset/CUB_200_2011/attributes/processed_image_attribute_labels.csv',
                                                                                                                                                    header=False,
                                                                                                                                                    index=False)
            self.image_attribute_labels = pd.read_csv(dataset_root_path+'/CUB_200_2011/attributes/processed_image_attribute_labels.csv',header=None).values
    
    def get_class_attribute(self, return_tensor=False):
        if(self.renumber == True):
            if(return_tensor):
                return torch.from_numpy(self.attribute[self.renumbered_label].astype(np.float32))
            else:
                return self.attribute[self.renumbered_label].astype(np.float32)
        else:
            if(return_tensor):
                return torch.from_numpy(self.attribute.astype(np.float32))
            else:
                return self.attribute.astype(np.float32)
        
    def renumber_index(self, label):
        if self.renumber==True:
            return self.label_transform_list[label]
        else:
            return label

    def load_image(self, idx):
        ipaths = self.image_path[idx][...,0]
        selected_xywh = self.object_xywh[idx]

        img_vec = []        
        for ip, oxywh in zip(ipaths, selected_xywh):
            ori_image = plt.imread(ip)/255.
            oxywh = oxywh.astype(int)
            w = h = max([oxywh[2], oxywh[3]])
            
            try:
                croped_image = ori_image[oxywh[1]:oxywh[1]+h, oxywh[0]:oxywh[0]+w, :]
            except:
                ori_image = np.tile(np.expand_dims(ori_image,axis=-1),(1,1,3))
                croped_image = ori_image[oxywh[1]:oxywh[1]+h, oxywh[0]:oxywh[0]+w, :]
            
            img_vec.append(self.transform(skTr.resize(croped_image, (self.size, self.size))))
   
        return np.stack(img_vec,axis=0)
    
    def __len__(self):
        return len(self.img_id_list)

    def __getitem__(self, index):
        
        idx = self.img_id_list[index]

        qur_img = self.load_image(idx)
        img = qur_img.astype(np.float32)
        img_attribute_label = self.image_attribute_labels[idx]
        label = np.array(self.img_label[idx])
        #class_attribute_label = self.attribute[label]
        cls_w = self.get_class_attribute()

        return img[0].astype(np.float32), img_attribute_label[0].astype(np.float32), self.renumber_index(label)[0].astype(np.int64), cls_w.astype(np.float32)


class SubDataset:
    def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), target_transform=identity):
        self.sub_meta = sub_meta
        self.cl = cl 
        self.transform = transform
        self.target_transform = target_transform
        self.image_attribute_label = np.zeros([len(sub_meta),312])
        self.avaliable_idx = np.array([_ for _ in range(len(self.sub_meta))])
        
        for counter in range(len(self.sub_meta)):
            self.image_attribute_label[counter, self.sub_meta[counter]['image_attributes'].numpy()] = 1
        
        
    def attribute_requirement(self, attr):
        hits = np.sum(self.image_attribute_label[...,attr],-1)
        self.avaliable_idx = np.where(hits==len(attr))[0]

    def __getitem__(self,i):
        #print( '%d -%d' %(self.cl,i))
        image_path = os.path.join(self.sub_meta[self.avaliable_idx[i]]['path'])
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        target = self.target_transform(self.cl)
        return img, target

    def __len__(self):
        return len(self.sub_meta)
    
class SubAttributeDataset(Dataset):

    def __init__(self, sub_meta, attr, image_size, transform=transforms.ToTensor(), target_transform=identity, is_train=True):
        self.num_joints = 15

        self.is_train = is_train
        self.sub_meta = sub_meta
        self.attr = attr 
        self.transform = transform
        self.target_transform = target_transform

        self.flip = is_train

        self.image_size = image_size
       
        self.transform = transform
        self.target_transform = target_transform
        self.ban_class_idx = None

    def ban_class(self,cls_num):
        self.ban_class_idx=cls_num
    
    def unban_class(self):
        self.ban_class_idx=None

    def __len__(self,):
        return len(self.sub_meta)

    def __getitem__(self, idx):
        if(isinstance(self.ban_class, type(None))):
            while(self.sub_meta[idx]['class'] in self.ban_class_idx):
                idx = (idx + 1)%len(self.sub_meta)
        image_file = os.path.join(self.sub_meta[idx]['path'])
        
        data_numpy = cv2.imread(image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

        if data_numpy is None:
            raise ValueError('Fail to read {}'.format(image_file))

        part_mask = self.sub_meta[idx]['mask']
        
        joints_vis = self.sub_meta[idx]['part']
        joints_vis = np.array(joints_vis)

        r = 0
        #c = np.array([data_numpy.shape[1], data_numpy.shape[0]]) // 2
        c = np.mean(joints_vis[np.where(part_mask!=0)][...,:-1],0)
        s = np.array([data_numpy.shape[1], data_numpy.shape[0]]) // 224

        if self.is_train:
            sf = 0.25
            rf = 30
            sd = np.random.uniform(0.,0.3)
            #s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
            s = np.random.uniform(0.95,1.05)*s
            c = (np.array([data_numpy.shape[1], data_numpy.shape[0]]) // 2)*sd + c*(1-sd)
            r = np.clip(np.random.randn()*rf, -rf*2, rf*2) \
                if random.random() <= 0.6 else 0

            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                for i in range(self.num_joints):
                    if joints_vis[i, 2] > 0.0:
                        joints_vis[i, 0] = data_numpy.shape[1] - joints_vis[i, 0]
                c[0] = data_numpy.shape[1] - c[0] - 1
            
        trans = get_affine_transform(c, s, r, [self.image_size, self.image_size])
        input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(self.image_size), int(self.image_size)),
            flags=cv2.INTER_LINEAR)
        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        input = Image.fromarray(input.transpose((1,0,2)))

        for i in range(self.num_joints):
            if joints_vis[i, 2] > 0.0:
                joints_vis[i, 0:2] = affine_transform(joints_vis[i, 0:2], trans)
                

        if self.transform:
            input = self.transform(input)


        joints_vis = self.target_transform(joints_vis)

        return input, joints_vis, part_mask



class SubPartsDataset(Dataset):

    def __init__(self, sub_meta, cl, image_size, transform=transforms.ToTensor(), target_transform=identity, is_train=True):
        self.num_joints = 15

        self.is_train = is_train
        self.sub_meta = sub_meta
        self.cl = cl 
        self.transform = transform
        self.target_transform = target_transform

        self.flip = is_train

        self.image_size = image_size
       
        self.transform = transform
        self.target_transform = target_transform
        self.image_attribute_label = np.zeros([len(sub_meta),312])
        self.avaliable_idx = np.array([_ for _ in range(len(self.sub_meta))])
        for counter in range(len(self.sub_meta)):
            self.image_attribute_label[counter, self.sub_meta[counter]['image_attributes']] = 1
        
        
    def attribute_requirement(self, attr):
        hits = np.sum(self.image_attribute_label[...,attr],-1)
        self.avaliable_idx = np.where(hits!=0)[0]


    def __len__(self,):
        return len(self.sub_meta)

    def __getitem__(self, idx):

        modified_idx = idx%len(self.avaliable_idx)
        image_file = os.path.join(self.sub_meta[self.avaliable_idx[modified_idx]]['path'])
        
        data_numpy = cv2.imread(image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

        if data_numpy is None:
            raise ValueError('Fail to read {}'.format(image_file))

        joints_vis = self.sub_meta[self.avaliable_idx[modified_idx]]['part']
        joints_vis = np.array(joints_vis)

        r = 0
        c = np.array([data_numpy.shape[1], data_numpy.shape[0]]) // 2
        s = np.array([data_numpy.shape[1], data_numpy.shape[0]]) // 160

        if self.is_train:
            sf = 0.25
            rf = 30
            s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn()*rf, -rf*2, rf*2) \
                if random.random() <= 0.6 else 0

            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                for i in range(self.num_joints):
                    if joints_vis[i, 2] > 0.0:
                        joints_vis[i, 0] = data_numpy.shape[1] - joints_vis[i, 0]
                c[0] = data_numpy.shape[1] - c[0] - 1
            
        trans = get_affine_transform(c, s, r, [self.image_size, self.image_size])
        input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(self.image_size), int(self.image_size)),
            flags=cv2.INTER_LINEAR)
        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        input = Image.fromarray(input.transpose((1,0,2)))

        for i in range(self.num_joints):
            if joints_vis[i, 2] > 0.0:
                new_joints = affine_transform(joints_vis[i, 0:2], trans)
                if(new_joints[0]>0 and new_joints[1]>0 and new_joints[0]<self.image_size-1 and new_joints[1]<self.image_size-1):
                    joints_vis[i, 0:2] = new_joints
                else:
                    joints_vis[i, 2] = 0
                

        if self.transform:
            input = self.transform(input)

        target = self.target_transform(self.cl)

        joints_vis = self.target_transform(joints_vis)

        return input, target, joints_vis

class TransformLoader:
    def __init__(self, image_size, 
                 normalize_param    = dict(mean= [0.485, 0.456, 0.406] , std=[0.229, 0.224, 0.225]),
                 jitter_param       = dict(Brightness=0.4, Contrast=0.4, Color=0.4)):
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param
    
    def parse_transform(self, transform_type):
        if transform_type=='ImageJitter':
            method = add_transforms.ImageJitter( self.jitter_param )
            return method
        method = getattr(transforms, transform_type)
        if transform_type=='RandomSizedCrop':
            return method(self.image_size) 
        elif transform_type=='CenterCrop':
            return method(self.image_size) 
        elif transform_type=='Scale':
            return method([int(self.image_size*1.15), int(self.image_size*1.15)])
        elif transform_type=='Normalize':
            return method(**self.normalize_param )
        else:
            return method()

    def get_composed_transform(self, aug = False):
        if aug:
            transform_list = ['ImageJitter', 'ToTensor', 'Normalize']
        else:
            transform_list = ['ToTensor', 'Normalize']

        transform_funcs = [ self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform


if __name__ == '__main__':
    trans_loader = TransformLoader(84)
    transform = trans_loader.get_composed_transform(False)
    test = SetDataset(data_file='/home/user/tzuyin/ra_project/dataset/CUB_200_2011/base.json', 
                      n_way=5, 
                      n_query=16, 
                      A_attr_per_class=1, 
                      image_size=84, transform=transform, 
                      is_train=True)
    ret = test[0]

