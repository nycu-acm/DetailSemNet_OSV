"""
taken and modified from https://github.com/pranv/ARC
"""

from builtins import print
import os
import numpy as np
from numpy.random import choice
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from skimage.transform import resize
import torchvision
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image, ImageFilter, ImageOps
from sklearn.model_selection import train_test_split
import cv2
import scipy.ndimage as ndimage
from tqdm import tqdm

from module.preprocess import preprocess_signature, normalize_image

use_cuda = True

def imread_tool(img_path):
    image = np.asarray(Image.open(img_path).convert('L'))
    inputSize = image.shape
    normalized_image, cropped_image = normalize_image(image.astype(np.uint8), inputSize) # RETURN: normalized, cropped
    #image = normalized_image
    #normalized_image = preprocess_signature(image.astype(np.uint8))
    return Image.fromarray(normalized_image) , Image.fromarray(cropped_image)

class SigDataset_BH(Dataset):
    def __init__(self, opt, path, train=True, image_size=256, mode='normalized'):
        self.path = path
        self.image_size = image_size
        if torchvision.__version__ == '0.8.2':
            print(torchvision.__version__, "old version")
            trans_list = [transforms.Resize((image_size,image_size)), # interpolation=transforms.InterpolationMode.NEAREST
            transforms.ToTensor(),]
        else:
            print(torchvision.__version__, "new version")
            trans_list = [transforms.RandomInvert(1.0),
            transforms.Resize((image_size,image_size)), # interpolation=transforms.InterpolationMode.NEAREST
            transforms.ToTensor(),]
        
        self.basic_transforms = transforms.Compose(trans_list)
        
        if torchvision.__version__ == '0.8.2':
            print(torchvision.__version__, "old version")
            class to_3channels(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                def forward(self, img):
                    return img.expand(-1, 3,*img.shape[2:])
            
            trans_aug_list = [to_3channels(),
                            transforms.ColorJitter(0.4, 0.4, 0.4, 0.2),
                            transforms.Grayscale(num_output_channels=1),
                            transforms.RandomErasing(),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomVerticalFlip(),
                            transforms.RandomResizedCrop((image_size,image_size)),
                            ]
        else:
            print(torchvision.__version__, "new version")
            trans_aug_list = [transforms.ColorJitter(0.4, 0.4, 0.4, 0.2),
                            transforms.RandomErasing(),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomVerticalFlip(),
                            transforms.RandomResizedCrop((image_size,image_size)),
                            ]
        
        self.augment_transforms = transforms.Compose(trans_aug_list)
        '''
        self.basic_transforms = transforms.Compose([transforms.RandomInvert(1.0),
                                                    transforms.Resize((image_size,image_size)),
                                                    transforms.ToTensor(),
                                                    #transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                                                    #transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5]),
                                                    #transforms.Grayscale(num_output_channels=1)
                                                    ])
        self.augment_transforms = transforms.Compose([transforms.ColorJitter(0.4, 0.4, 0.4, 0.2),
                                                    transforms.RandomErasing(),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.RandomVerticalFlip(),
                                                    transforms.RandomResizedCrop((image_size,image_size)),
                                                    #transforms.RandomInvert(),
                                                    ])
        '''
        self.shift_transform = transforms.RandomAffine(degrees=0, scale=(1.0, 1.5))
        # translate: transforms.RandomAffine(degrees=0, translate=(0.5, 0.5))
        # rotate: transforms.RandomAffine(degrees=45)
        # scale: transforms.RandomAffine(degrees=0, scale=(0.5, 1.5))
        # all: transforms.RandomAffine(degrees=45, translate=(0.5, 0.5), scale=(0.5, 1.5))
        data_root = path  # ./../BHSig260/Bengali
        # 24 pos, 30 neg
        pos_set = set()
        for x in range(1,24+1):
            for y in range(1,24+1):
                if x != y and (y,x) not in pos_set:
                    pos_set.add((x, y))
        neg_set = set((x, y) for x in range(1,24+1) for y in range(1,30+1))

        print(len(pos_set), len(neg_set))
        
        self.img_dict = {}
        data_df = pd.DataFrame(columns=['index','img_path', 'label', 'writer_id'])
        FLAG = os.path.exists(os.path.join(data_root, "pairs.csv"))
        data_df_pair = pd.DataFrame(columns=['item_0','item_1', 'label', 'writer_id'])
        if FLAG:
            data_df_pair = pd.read_csv(os.path.join(data_root, "pairs.csv"))
        for dir in tqdm(os.listdir(data_root)):
            dir_path = os.path.join(data_root, dir)
            if os.path.isdir(dir_path):
                single_dict = {}
                for img in os.listdir(dir_path):
                    img_path = os.path.join(dir_path, img)
                    img_split = img.split('-')
                    label = None
                    if img_split[3] == 'G':
                        label = 1
                    elif img_split[3] == 'F':
                        label = 0
                    assert label is not None
                    index = int(img_split[4][:-4])
                    #data_df = data_df.append({'index': index, 'img_path': img_path, 'label': label, 'writer_id': int(dir)}, ignore_index=True)
                    single_dict[(label, index)] = img_path
                    
                    # normalized, cropped
                    #'''
                    if mode == 'normalized':
                        sig_image, _ = imread_tool(img_path)
                    elif mode == 'cropped':
                        _, sig_image = imread_tool(img_path)
                        sig_image = sig_image#.filter(ImageFilter.MinFilter(5))
                    elif mode == 'centered':
                        _, sig_image = imread_tool(img_path)
                        trans = transforms.CenterCrop(np.asarray(sig_image).shape[0])
                        sig_image = trans(sig_image)
                    elif mode == 'left':
                        sig_image, _ = imread_tool(img_path)
                        width, height = sig_image.size
                        left = 0
                        top = 0
                        right = width / 2
                        bottom = height
                        sig_image = sig_image.crop((left, top, right, bottom)).filter(ImageFilter.MinFilter(5))
                    else:
                        return NotImplementedError
                    if torchvision.__version__ == '0.8.2':
                        sig_image = ImageOps.invert(sig_image)
                    sig_image = self.basic_transforms(sig_image)
                    self.img_dict[img_path] = sig_image
                    #'''
                if not FLAG:
                    for item in list(pos_set):
                        first, second = item
                        path_first = single_dict[(1, first)]
                        path_second = single_dict[(1, second)]
                        assert path_first is not None
                        assert path_second is not None
                        data_df_pair = data_df_pair.append({'item_0': path_first, 'item_1': path_second, 'label': 1, 'writer_id': int(dir)}, ignore_index=True)
                        #data_df_pair = data_df_pair.append({'item_0': path_second, 'item_1': path_first, 'label': 1, 'writer_id': int(dir)}, ignore_index=True)
                    for item in list(neg_set):
                        first, second = item
                        path_first = single_dict[(1, first)]
                        path_second = single_dict[(0, second)]
                        assert path_first is not None
                        assert path_second is not None
                        data_df_pair = data_df_pair.append({'item_0': path_first, 'item_1': path_second, 'label': 0, 'writer_id': int(dir)}, ignore_index=True)
                        #data_df_pair = data_df_pair.append({'item_0': path_second, 'item_1': path_first, 'label': 0, 'writer_id': int(dir)}, ignore_index=True)

        #print(f'total {len(data_df)} images !!')
        print(f'total {len(data_df_pair)} pairs !!')
        #self.total_data_df = data_df
        self.total_data_df = data_df_pair
        if 'demo' in path:
            self.train_df = data_df_pair
            self.test_df = data_df_pair
        elif 'Bengali' in path:
            #'''
            if opt.part:
                with open("test_B.json", "r") as fp:
                    import json
                    test_list = json.load(fp)
                    self.train_df = data_df_pair[~data_df_pair['writer_id'].isin(test_list)]
            else:
                self.train_df = data_df_pair[data_df_pair['writer_id'] > 50]
            
            if opt.part:
                with open("test_B.json", "r") as fp:
                    import json
                    test_list = json.load(fp)
                    self.test_df = data_df_pair[data_df_pair['writer_id'].isin(test_list)]
            else:
                self.test_df = data_df_pair[data_df_pair['writer_id'] <= 50]
            # old version
            #self.train_df = data_df_pair[data_df_pair['writer_id'] >= 50]
            #self.test_df = data_df_pair[data_df_pair['writer_id'] < 50]
            '''
            self.train_df = pos_df[pos_df['writer_id'] >= 20]
            self.test_df = pos_df[pos_df['writer_id'] < 20]
            '''
            #self.train_df, self.test_df = train_test_split(pos_df, test_size=0.5, shuffle=False, random_state=1)
        elif 'Hindi' in path: # 100 train, 60 test
            if opt.part:
                with open("test_H.json", "r") as fp:
                    import json
                    test_list = json.load(fp)
                    self.train_df = data_df_pair[~data_df_pair['writer_id'].isin(test_list)]
            else:
                self.train_df = data_df_pair[data_df_pair['writer_id'] > 60]
            if opt.part:
                with open("test_H.json", "r") as fp:
                    import json
                    test_list = json.load(fp)
                    self.test_df = data_df_pair[data_df_pair['writer_id'].isin(test_list)]
            else:
                self.test_df = data_df_pair[data_df_pair['writer_id'] <= 60]
            # old version
            #self.train_df = data_df_pair[data_df_pair['writer_id'] >= 60]
            #self.test_df = data_df_pair[data_df_pair['writer_id'] < 60]
            #self.train_df, self.test_df = train_test_split(pos_df, test_size=0.5, shuffle=False, random_state=1)
        #data_df.to_csv("data.csv")
        if not FLAG:
            data_df_pair.to_csv(os.path.join(data_root, "pairs.csv"))

        self.train = train

        self.shift = opt.shift

    def __len__(self):
        if self.train:
            return len(self.train_df)
        else:
            return len(self.test_df)
    
    def __getitem__(self, index):
        if self.train:
            self.data_df = self.train_df
        else:
            self.data_df = self.test_df
        
        # Anchor
        img_path_0 = self.data_df.iloc[index]['item_0']
        sig_image_0 = self.img_dict[img_path_0]
        img_path_1 = self.data_df.iloc[index]['item_1']
        sig_image_1 = self.img_dict[img_path_1]
        label = self.data_df.iloc[index]['label']
        writer_id = self.data_df.iloc[index]['writer_id']
        #"""
        if self.shift:
            sig_image_0 = self.shift_transform(sig_image_0)
            sig_image_1 = self.shift_transform(sig_image_1)
        #"""
        image_pair = torch.cat((sig_image_0, sig_image_1), dim=0)
        '''
        if self.train:
            image_pair = torch.squeeze(self.augment_transforms(torch.unsqueeze(image_pair, dim=1)))
        '''
        #return image_pair, torch.tensor([[label]])
        return image_pair, torch.tensor([[label]]), writer_id

class SigDataset_CEDAR(Dataset):
    def __init__(self, opt, path, train=True, image_size=256, mode='normalized'):
        self.path = path
        self.image_size = image_size
        trans_list = [transforms.RandomInvert(1.0),
                    transforms.Resize((image_size,image_size)), # interpolation=transforms.InterpolationMode.NEAREST
                    transforms.ToTensor(),]
        
        self.basic_transforms = transforms.Compose(trans_list)
    
        trans_aug_list = [transforms.ColorJitter(0.4, 0.4, 0.4, 0.2),
                        transforms.RandomErasing(),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        transforms.RandomResizedCrop((image_size,image_size)),
                        #transforms.RandomPerspective(),
                        ]
        
        self.augment_transforms = transforms.Compose(trans_aug_list)
        
        data_root = path  # ./../BHSig260/Bengali
        if train:
            path = os.path.join(data_root, "gray_train.txt")
            if opt.part:
                path = os.path.join(data_root, "gray_train_part.txt")
        else:
            path = os.path.join(data_root, "gray_test.txt")
            #path = os.path.join(data_root, "gray_all.txt")
            if opt.part:
                path = os.path.join(data_root, "gray_test_part.txt")
        
        self.img_dict = {}
        for dir in os.listdir(data_root):
            dir_path = os.path.join(data_root, dir)
            if os.path.isdir(dir_path):
                for img in tqdm(os.listdir(dir_path)):
                    if img[-4:] == '.png':
                        img_path = os.path.join(dir_path, img)
                        sig_image, _ = imread_tool(img_path)
                        sig_image = self.basic_transforms(sig_image)
                        self.img_dict[img_path] = sig_image
        
        with open(path, 'r') as f:
            lines = f.readlines()

        self.labels = []
        self.datas = []
        self.writer_id = []
        for line in lines:
            refer, test, label = line.split()
            
            refer_img = self.img_dict[os.path.join(data_root, refer)]
            test_img = self.img_dict[os.path.join(data_root, test)]
            
            refer_test = torch.cat((refer_img, test_img), dim=0)
            self.datas.append(refer_test)
            self.labels.append(int(label))
            self.writer_id.append(int(str(refer).split('_')[2]))
        
        self.train = train
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        image_pair = self.datas[index]
        
        if self.train:
            image_pair = torch.squeeze(self.augment_transforms(torch.unsqueeze(image_pair, dim=1)))

        return image_pair, torch.tensor([[self.labels[index]]])
        #return image_pair, torch.tensor([[self.labels[index]]]), self.writer_id[index]

class single_test_dataset(Dataset):
    def __init__(self, opt, anchor_path, ref_path, label, image_size=224, mode='normalized'):
        self.anchor_path = anchor_path
        self.ref_path = ref_path
        if torchvision.__version__ == '0.8.2':
            print(torchvision.__version__, "old version")
            trans_list = [transforms.Resize((image_size,image_size)), # interpolation=transforms.InterpolationMode.NEAREST
            transforms.ToTensor(),]
        else:
            print(torchvision.__version__, "new version")
            trans_list = [transforms.RandomInvert(1.0),
            transforms.Resize((image_size,image_size)), # interpolation=transforms.InterpolationMode.NEAREST
            transforms.ToTensor(),]
        
        self.basic_transforms = transforms.Compose(trans_list)
        '''
        self.basic_transforms = transforms.Compose([transforms.RandomInvert(1.0),
                                                    transforms.Resize((image_size,image_size)),
                                                    transforms.ToTensor(),
                                                    #transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                                                    #transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5]),
                                                    #transforms.Grayscale(num_output_channels=1)
                                                    ])
        '''
        self.img_dict = {}
        path_dict = {'Anchor':anchor_path, 'ref':ref_path}
        print(path_dict)
        for item in ['Anchor', 'ref']:
            # normalized, cropped
            if mode == 'normalized':
                sig_image, _ = imread_tool(path_dict[item])
                sig_image.save('img_pre_{}.png'.format(item))
            elif mode == 'cropped':
                _, sig_image = imread_tool(path_dict[item])
            elif mode == 'centered':
                _, sig_image = imread_tool(path_dict[item])
                trans = transforms.CenterCrop(np.asarray(sig_image).shape[0])
                sig_image = trans(sig_image)
            elif mode == 'left':
                sig_image, _ = imread_tool(path_dict[item])
                width, height = sig_image.size
                left = 0
                top = 0
                right = width / 2
                bottom = height
                sig_image = sig_image.crop((left, top, right, bottom)).filter(ImageFilter.MinFilter(5))
            else:
                return NotImplementedError
            if torchvision.__version__ == '0.8.2':
                sig_image = ImageOps.invert(sig_image)
            sig_image = self.basic_transforms(sig_image)
            self.img_dict[item] = sig_image
            self.label = label
            self.shift = opt.shift
            self.shift_transform = transforms.RandomAffine(degrees=40, translate=(0.2, 0.2))
            # translate: transforms.RandomAffine(degrees=0, translate=(0.5, 0.5))
            # rotate: transforms.RandomAffine(degrees=40)
            # scale: transforms.RandomAffine(degrees=0, scale=(0.5, 1.5))
    def __len__(self):
        return 1
    
    def __getitem__(self, index):
        # 'Anchor', 'ref'
        sig_image = self.img_dict['Anchor']
        ref_sig_image = self.img_dict['ref']
        if self.shift:
            sig_image = self.shift_transform(sig_image)
            ref_sig_image = self.shift_transform(ref_sig_image)
        image_pair_0 = torch.cat((sig_image, ref_sig_image), dim=0)

        return image_pair_0, torch.tensor([[self.label]]), {'Anchor':self.anchor_path, 'ref':self.ref_path}

class SigDataset_real(Dataset):
    def __init__(self, path, image_size=256, mode='normalized'):
        self.path = path
        self.image_size = image_size
        trans_list = [transforms.RandomInvert(1.0),
                      transforms.Resize((image_size,image_size)), # interpolation=transforms.InterpolationMode.NEAREST
                      transforms.ToTensor(),]
        
        self.basic_transforms = transforms.Compose(trans_list)
        
        data_root = path  # ./../BHSig260/Bengali
        
        self.img_dict = {}
        data_df_pair = pd.DataFrame(columns=['item_0','item_1', 'label'])

        self.img_dict = {}
        for img in os.listdir(data_root):
            img_path = os.path.join(data_root, img)
            sig_image, _ = imread_tool(img_path)
            sig_image = self.basic_transforms(sig_image)
            self.img_dict[img_path] = sig_image

        for img_1 in os.listdir(data_root):
            for img_2 in os.listdir(data_root):
                img_split_1 = img_1.split('-')
                img_split_2 = img_2.split('-')
                path_first = os.path.join(data_root, img_1)
                path_second = os.path.join(data_root, img_2)
                if img_split_1[0] == img_split_2[0] and img_split_1[1] != img_split_2[1]:
                    data_df_pair = data_df_pair.append({'item_0': path_first, 'item_1': path_second, 'label': 1}, ignore_index=True)
                elif img_split_1[0] != img_split_2[0]:
                    data_df_pair = data_df_pair.append({'item_0': path_first, 'item_1': path_second, 'label': 0}, ignore_index=True)
        self.data_df = data_df_pair

    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, index):
        img_path_0 = self.data_df.iloc[index]['item_0']
        sig_image_0 = self.img_dict[img_path_0]
        
        img_path_1 = self.data_df.iloc[index]['item_1']
        sig_image_1 = self.img_dict[img_path_1]
        
        label = self.data_df.iloc[index]['label']
        
        image_pair = torch.cat((sig_image_0, sig_image_1), dim=0)
        
        return image_pair, torch.tensor([[label]]), {'Anchor':img_path_0, 'ref':img_path_1}

if __name__ == '__main__':
    '''
    path = './../seg_demo/real'

    sigdataset_train = SigDataset_real(path, image_size=64, mode='normalized')
    train_loader = DataLoader(sigdataset_train, batch_size=1, shuffle=False)
    '''
    '''
    path = './../BHSig260/Bengali'

    sigdataset_train = SigDataset_BH(path, train=True, image_size=64, mode='normalized')
    train_loader = DataLoader(sigdataset_train, batch_size=2, shuffle=False)

    path = './../BHSig260/Hindi'

    sigdataset_train = SigDataset_BH(path, train=True, image_size=64, mode='normalized')
    train_loader = DataLoader(sigdataset_train, batch_size=2, shuffle=False)
    '''
    path = './../CEDAR'
    sigdataset_train = SigDataset_CEDAR(opt=None, path=path, train=True, image_size=64, mode='normalized')
    train_loader = DataLoader(sigdataset_train, batch_size=2, shuffle=False)
    
    for X, Y in tqdm(train_loader):
        X = X.view(-1,2,64,64)
        Y = Y.view(-1,1)