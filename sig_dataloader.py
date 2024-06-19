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

def imread_tool(img_path, convet_type = 'L', binarization = False):
    image = np.asarray(Image.open(img_path).convert(convet_type))
    inputSize = image.shape
    normalized_image, cropped_image = normalize_image(image.astype(np.uint8), inputSize) # RETURN: normalized, cropped
    #image = normalized_image
    if binarization:
        normalized_image[normalized_image < 255] = 0
        cropped_image[cropped_image < 255] = 0
    return Image.fromarray(normalized_image) , Image.fromarray(cropped_image)

class SigDataset(Dataset):
    def __init__(self, path, train=True, image_size=256, convert_type = 'L'):
        self.path = path
        self.image_size = image_size
        self.basic_transforms = transforms.Compose([transforms.RandomInvert(1.0),
                                                    transforms.Resize((image_size,image_size)),
                                                    transforms.ToTensor(),
                                                    #transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                                                    #transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5]),
                                                    #transforms.Normalize(mean=0.5,std=0.25),
                                                    #transforms.Grayscale(num_output_channels=1)
                                                    ])
        self.augment_transforms = transforms.Compose([transforms.ColorJitter(0.4, 0.4, 0.4, 0.2),
                                                    transforms.RandomErasing(),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.RandomVerticalFlip(),
                                                    transforms.RandomResizedCrop((image_size,image_size)),
                                                    #transforms.RandomPerspective(),
                                                    #transforms.RandomInvert(),
                                                    ])
        self.shift_transform = transforms.RandomAffine(degrees=0, translate=(0.2, 0.2))
        data_root = path  # ./../ChiSig
        data_df = pd.DataFrame(columns=['img_path', 'label', 'writer_id'])

        self.img_path_dict = {}
        for img in os.listdir(data_root):
            img_split = img.split('-')
            self.img_path_dict.setdefault(img_split[0], []).append(img)

        self.img_path_dict = dict(sorted(self.img_path_dict.items()))
        self.img_dict = {}
        for writer_id, key in tqdm(enumerate(self.img_path_dict)):
            if writer_id >= 100:
                continue
            pos_label = self.img_path_dict[key][0].split('-')[1]
            pos_flag = False
            for img in self.img_path_dict[key]:
                img_split = img.split('-')
                if img_split[1] != pos_label:
                    pos_flag = True
            if pos_flag:
                for img in self.img_path_dict[key]:
                    img_path = os.path.join(data_root, img)
                    img_split = img.split('-')
                    label = int(img_split[1])
                    data_df = data_df.append({'img_path': img_path, 'label': label, 'writer_id': writer_id}, ignore_index=True)
                    sig_image, _ = imread_tool(img_path, convet_type=convert_type) # normalized
                    sig_image = self.basic_transforms(sig_image)
                    self.img_dict[img_path] = sig_image
            
        print(f'total {len(data_df)} images !!')
        # self.data_df = self.train_df = self.test_df = data_df
        self.data_df = data_df
        #self.train_df, self.test_df = train_test_split(data_df, test_size=0.3, shuffle=False, random_state=1)
        #data_df.to_csv("data.csv")
        #self.train_df = data_df[data_df['writer_id'] >= 100]
        #self.test_df = data_df[data_df['writer_id'] < 100]
        self.train_df = data_df[data_df['writer_id'] >= 20]
        self.test_df = data_df[data_df['writer_id'] < 20]
        #self.train_df.to_csv("data_train.csv")
        #self.test_df.to_csv("data_test.csv")

        self.train = train

    def __len__(self):
        if self.train:
            return len(self.train_df)
        else:
            return len(self.test_df)

    def __getitem__(self, index):
        if self.train:
            self.group = self.train_df.groupby('writer_id')
            self.data_df = self.train_df
        else:
            self.group = self.test_df.groupby('writer_id')
            self.data_df = self.test_df
        
        group_writer_id = self.data_df.iloc[index]['writer_id']
        in_class_df = self.group.get_group(group_writer_id)
        # Anchor
        img_path = self.data_df.iloc[index]['img_path']
        #sig_image = Image.open(img_path).convert('RGB')
        #sig_image = imread_tool(img_path)
        #sig_image = self.basic_transforms(sig_image)
        sig_image = self.img_dict[img_path]
        #if self.train:
        #    sig_image = self.augment_transforms(sig_image)
        writer_id = self.data_df.iloc[index]['writer_id']
        label = self.data_df.iloc[index]['label']

        positive_path, negative_path = None, None

        # positive
        while True:
            #print(img_path, len(in_class_df[in_class_df['label'] == label]))
            if len(in_class_df[in_class_df['label'] == label]) == 1:
                positive_path = img_path
                break
            else:
                sample_df = in_class_df.sample()
                if sample_df['label'].item() == self.data_df.iloc[index]['label'] and sample_df['img_path'].item() != img_path:
                    positive_path = sample_df['img_path'].item()
                    #print(positive_path)
                    break
        #positive_sig_image = Image.open(positive_path).convert('RGB')
        #positive_sig_image = imread_tool(positive_path)
        #positive_sig_image = self.basic_transforms(positive_sig_image)
        positive_sig_image = self.img_dict[positive_path]
        #if self.train:
        #    positive_sig_image = self.augment_transforms(positive_sig_image)

        # negative
        while True:
            sample_df = in_class_df.sample()
            if sample_df['label'].item() != self.data_df.iloc[index]['label']:
                negative_path = sample_df['img_path'].item()
                #print(negative_path)
                break
        #negative_sig_image = Image.open(negative_path).convert('RGB')
        #negative_sig_image = imread_tool(negative_path)
        #negative_sig_image = self.basic_transforms(negative_sig_image)
        negative_sig_image = self.img_dict[negative_path]
        #if self.train:
        #    negative_sig_image = self.augment_transforms(negative_sig_image)
        '''
        while True:
            sample_df = self.data_df.sample()
            if sample_df['writer_id'].item() != self.data_df.iloc[index]['writer_id']:
                negative_path = sample_df['img_path'].item()
                break
        negative_sig_image = self.img_dict[negative_path]
        '''
        
        image_pair_0 = torch.cat((sig_image, positive_sig_image), dim=0)
        if self.train: image_pair_0 = torch.squeeze(self.augment_transforms(torch.unsqueeze(image_pair_0, dim=1)))
        image_pair_1 = torch.cat((sig_image, negative_sig_image), dim=0)
        if self.train: image_pair_1 = torch.squeeze(self.augment_transforms(torch.unsqueeze(image_pair_1, dim=1)))
        image_pairs = torch.cat((torch.unsqueeze(image_pair_0, 0), torch.unsqueeze(image_pair_1, 0)), dim=0)
        #print(image_pairs.shape)

        if self.train:
            return image_pairs, torch.tensor([[1],[0]])
        else:
            return image_pairs, torch.tensor([[1],[0]])#, {'Anchor':img_path, 'positive':positive_path, 'negative':negative_path}


class SigDataset_BH(Dataset):
    def __init__(self, opt, path, train=True, image_size=256, mode='normalized', save=False, partition=False):
        self.path = path
        self.image_size = image_size
        #torchvision.transforms.functional.invert
        if torchvision.__version__ == '0.8.2':
            print(torchvision.__version__, "old version")
            trans_list = [transforms.Resize((image_size,image_size),), # interpolation=transforms.InterpolationMode.NEAREST
            transforms.ToTensor(),]
        else:
            print(torchvision.__version__, "new version")
            trans_list = [transforms.RandomInvert(1.0),
            transforms.Resize((image_size,image_size),), # interpolation=transforms.InterpolationMode.NEAREST
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
                            #transforms.RandomPerspective(),
                            ]
        
        self.augment_transforms = transforms.Compose(trans_aug_list)
        
        class pixel_trans(torch.nn.Module):
            def __init__(self):
                super().__init__()
            def forward(self, img):
                _, h, w = img.shape
                direct_shift = torch.clone(img)
                shift_value_x = 8
                shift_value_y = 8
                direct_shift[:, shift_value_y:, :] = torch.clone(direct_shift[:, :h-shift_value_y, :])
                direct_shift[:, :, shift_value_x:] = torch.clone(direct_shift[:, :, :w-shift_value_x])
                return direct_shift

        self.shift_transform = transforms.Compose([transforms.RandomAffine(degrees=(1,1), interpolation=transforms.InterpolationMode.BILINEAR),
                                                    transforms.RandomAffine(degrees=(-1,-1),),
        ])#, interpolation=transforms.InterpolationMode.BILINEAR)
        # translate: transforms.RandomAffine(degrees=0, translate=(0.2, 0.2))
        # rotate: transforms.RandomAffine(degrees=(1,1))
        # scale: transforms.RandomAffine(degrees=0, scale=(0.5, 1.5))
        # all: transforms.RandomAffine(degrees=(1,1), translate=(0.2, 0.2), scale=(0.5, 1.5))
        data_root = path  # ./../BHSig260/Bengali
        data_df = pd.DataFrame(columns=['img_path', 'label', 'writer_id'])

        self.img_dict = {}
        for dir in tqdm(os.listdir(data_root)):
            dir_path = os.path.join(data_root, dir)
            if os.path.isdir(dir_path):
                for img in os.listdir(dir_path):
                    img_path = os.path.join(dir_path, img)
                    img_split = img.split('-')
                    label = None
                    if img_split[3] == 'G':
                        label = 1
                    elif img_split[3] == 'F':
                        label = 0
                    assert label is not None
                    data_df = data_df.append({'img_path': img_path, 'label': label, 'writer_id': int(dir)}, ignore_index=True)
                    #sig_image = Image.open(img_path).convert('1')
                    # normalized, cropped
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
                    #sig_image = self.__get_com_cropped__(sig_image)
                    if torchvision.__version__ == '0.8.2':
                        sig_image = ImageOps.invert(sig_image)
                    sig_image = self.basic_transforms(sig_image)
                    self.img_dict[img_path] = sig_image

        print(f'total {len(data_df)} images !!')
        self.total_data_df = data_df
        pos_df = data_df.groupby('label').get_group(1)
        if 'demo' in path:
            self.train_df = pos_df
            self.test_df = pos_df
        elif 'Bengali' in path:
            #'''
            # load training part
            if opt.fs:
                self.train_df = pos_df[pos_df['writer_id'] == 50]
            elif opt.part:
                with open("test_B.json", "r") as fp:
                    import json
                    test_list = json.load(fp)
                    self.train_df = pos_df[~pos_df['writer_id'].isin(test_list)]
            else:    
                self.train_df = pos_df[pos_df['writer_id'] > 50]
            
            # load testing part
            if opt.part:
                with open("test_B.json", "r") as fp:
                    import json
                    test_list = json.load(fp)
                    self.test_df = pos_df[pos_df['writer_id'].isin(test_list)]
            else:
                self.test_df = pos_df[pos_df['writer_id'] <= 50]
            '''
            # old version
            else:    
                self.train_df = pos_df[pos_df['writer_id'] >= 50]
            self.test_df = pos_df[pos_df['writer_id'] < 50]
            '''
            '''
            self.train_df = pos_df[pos_df['writer_id'] >= 20]
            self.test_df = pos_df[pos_df['writer_id'] < 20]
            '''
            #self.train_df, self.test_df = train_test_split(pos_df, test_size=0.5, shuffle=False, random_state=1)
        elif 'Hindi' in path: # 100 train, 60 test
            if opt.fs:
                self.train_df = pos_df[pos_df['writer_id'] == 60]
            elif opt.part:
                with open("test_H.json", "r") as fp:
                    import json
                    test_list = json.load(fp)
                    self.train_df = pos_df[~pos_df['writer_id'].isin(test_list)]
            else:    
                self.train_df = pos_df[pos_df['writer_id'] > 60]
            if opt.part:
                with open("test_H.json", "r") as fp:
                    import json
                    test_list = json.load(fp)
                    self.test_df = pos_df[pos_df['writer_id'].isin(test_list)]
            else:
                self.test_df = pos_df[pos_df['writer_id'] <= 60]
            '''
            # old version
            else:    
                self.train_df = pos_df[pos_df['writer_id'] >= 60]
            self.test_df = pos_df[pos_df['writer_id'] < 60]
            '''
            #self.train_df, self.test_df = train_test_split(pos_df, test_size=0.5, shuffle=False, random_state=1)
        #data_df.to_csv("data.csv")

        self.train = train

        self.shift = opt.shift

        self.save = save
        print(self.save)

    def __len__(self):
        if self.train:
            return len(self.train_df)
        else:
            return 2*len(self.test_df)
    
    def __get_com_cropped__(self, image):
        '''image is a binary PIL image'''
        image = np.asarray(image)
        com = ndimage.measurements.center_of_mass(image)
        com = np.round(com)
        com[0] = np.clip(com[0], 0, image.shape[0])
        com[1] = np.clip(com[1], 0, image.shape[1])
        X_center, Y_center = int(com[0]), int(com[1])
        c_row, c_col = image[X_center, :], image[:, Y_center]

        x_start, x_end, y_start, y_end = -1, -1, -1, -1

        for i, v in enumerate(c_col):
            v = np.sum(image[i, :])
            if v < 255*image.shape[1]: # there exists text pixel
                if x_start == -1:
                    x_start = i
                else:
                    x_end = i

        for j, v in enumerate(c_row):
            v = np.sum(image[:, j])
            if v < 255*image.shape[0]: # there exists text pixel
                if y_start == -1:
                    y_start = j
                else:
                    y_end = j

        crop_rgb = Image.fromarray(np.asarray(image[x_start:x_end, y_start:y_end]))
        return crop_rgb
    
    def __getitem__(self, index):
        if self.train:
            self.group = self.total_data_df.groupby('writer_id')
            self.data_df = self.train_df
            index_anchor = index
        else:
            self.group = self.total_data_df.groupby('writer_id')
            self.data_df = self.test_df
            index_anchor = index//2
        
        # Anchor
        img_path = self.data_df.iloc[index_anchor]['img_path']
        #print(img_path)
        #sig_image = imread_tool(img_path)
        sig_image = self.img_dict[img_path]
        #if self.train:
        #    sig_image = self.augment_transforms(sig_image)
        writer_id = self.data_df.iloc[index_anchor]['writer_id']
        label = self.data_df.iloc[index_anchor]['label']
        
        ###
        group_writer_id = self.data_df.iloc[index_anchor]['writer_id']
        in_class_df = self.group.get_group(group_writer_id)

        positive_path, negative_path = None, None

        # positive
        if self.train or index%2 == 0:
            while True:
                #print(img_path, len(in_class_df[in_class_df['label'] == label]))
                if len(in_class_df[in_class_df['label'] == label]) == 1:
                    positive_path = img_path
                    break
                else:
                    sample_df = in_class_df.sample()
                    if sample_df['label'].item() == label and sample_df['img_path'].item() != img_path:
                        positive_path = sample_df['img_path'].item()
                        break
            #print(positive_path)
            #positive_sig_image = Image.open(positive_path).convert('RGB')
            #positive_sig_image = imread_tool(positive_path)
            positive_sig_image = self.img_dict[positive_path]
            #if self.train:
            #    positive_sig_image = self.augment_transforms(positive_sig_image)

        # negative
        if self.train or index%2 != 0:
            while True:
                sample_df = in_class_df.sample()
                if sample_df['label'].item() != label:
                    negative_path = sample_df['img_path'].item()
                    break
            #print(negative_path)
            #negative_sig_image = Image.open(negative_path).convert('RGB')
            #negative_sig_image = imread_tool(negative_path)
            negative_sig_image = self.img_dict[negative_path]
            #if self.train:
            #    negative_sig_image = self.augment_transforms(negative_sig_image)

        # negative 2
        '''
        while True:
            sample_df = in_class_df.sample()
            if sample_df['label'].item() != self.data_df.iloc[index]['label'] and sample_df['img_path'].item() != negative_path:
                negative_path_2 = sample_df['img_path'].item()
                break
        negative_sig_image_2 = self.img_dict[negative_path_2]
        '''
        
        if self.train:
            if self.shift:
                sig_image = self.shift_transform(sig_image)
                positive_sig_image = self.shift_transform(positive_sig_image)
                negative_sig_image = self.shift_transform(negative_sig_image)
            
            #print(sig_image.shape, positive_sig_image.shape, negative_sig_image.shape)
            image_pair_0 = torch.cat((sig_image, positive_sig_image), dim=0)
            image_pair_0 = torch.squeeze(\
                self.augment_transforms(torch.unsqueeze(image_pair_0, dim=1)))
            image_pair_1 = torch.cat((sig_image, negative_sig_image), dim=0)
            image_pair_1 = torch.squeeze(\
                self.augment_transforms(torch.unsqueeze(image_pair_1, dim=1)))
            image_pair_2 = torch.cat((positive_sig_image, sig_image), dim=0)
            image_pair_2 = torch.squeeze(\
                self.augment_transforms(torch.unsqueeze(image_pair_2, dim=1)))
            image_pair_3 = torch.cat((negative_sig_image, sig_image), dim=0)
            image_pair_3 = torch.squeeze(\
                self.augment_transforms(torch.unsqueeze(image_pair_3, dim=1)))
            
            image_pairs = torch.cat((torch.unsqueeze(image_pair_0, 0),
                                    torch.unsqueeze(image_pair_1, 0),
                                    torch.unsqueeze(image_pair_2, 0),
                                    torch.unsqueeze(image_pair_3, 0)
                                    ), dim=0)
                
            return image_pairs, torch.tensor([[1],[0],[1],[0]])
        else:
            '''
            #print(positive_sig_image.shape)
            #positive_sig_image = self.shift_transform(positive_sig_image)
            #negative_sig_image = self.shift_transform(negative_sig_image)
            # original
            image_pair_0 = torch.cat((sig_image, positive_sig_image), dim=0)
            # test aug
            #image_pair_0 = torch.unsqueeze(image_pair_0, dim=1)
            #image_pair_0 = torch.squeeze(self.augment_transforms(image_pair_0))
            ###
            image_pair_1 = torch.cat((sig_image, negative_sig_image), dim=0)
            image_pairs = torch.cat((torch.unsqueeze(image_pair_0, 0), torch.unsqueeze(image_pair_1, 0)), dim=0)
        

            if self.save:
                return image_pairs, torch.tensor([[1],[0]]), {'Anchor':img_path, 'positive':positive_path, 'negative':negative_path}
            else:
                return image_pairs, torch.tensor([[1],[0]])
            '''
            if index%2 == 0:
                image_pair_final = torch.cat((sig_image, positive_sig_image), dim=0)
                label_final = torch.tensor([[1]])
            else:
                image_pair_final = torch.cat((sig_image, negative_sig_image), dim=0)
                label_final = torch.tensor([[0]])
            
            if self.save:
                #return image_pair_final, label_final, {'Anchor':img_path, 'positive':positive_path, 'negative':negative_path}
                return image_pair_final, label_final, {'Anchor':img_path, 'ref':positive_path if index%2 == 0 else negative_path}
            else:
                return image_pair_final, label_final

class SigDataset_different(Dataset):
    def __init__(self, path, train=True, image_size=256):
        self.path = path
        self.image_size = image_size
        self.basic_transforms = transforms.Compose([transforms.RandomInvert(1.0),
                                                    transforms.Resize((image_size,image_size)),
                                                    transforms.ToTensor(),
                                                    #transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                                                    #transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5]),
                                                    #transforms.Normalize(mean=0.5,std=0.25),
                                                    #transforms.Grayscale(num_output_channels=1)
                                                    ])
        self.augment_transforms = transforms.Compose([transforms.ColorJitter(0.4, 0.4, 0.4, 0.2),
                                                    transforms.RandomErasing(),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.RandomVerticalFlip(),
                                                    transforms.RandomResizedCrop((image_size,image_size)),
                                                    #transforms.RandomInvert(),
                                                    ])
        self.shift_transform = transforms.RandomAffine(degrees=0, translate=(0.2, 0.2))
        data_root = path  # ./../ChiSig
        self.img_dict = {}
        data_df = self.process_dataset(data_root=data_root)
        data_df_2 = self.process_dataset(data_root='/mnt/HDD1/shih/OSV/seg_demo/ChiSig_Synthetic/Test_result')
            
        print(f'total {len(data_df)} images !!')
        self.data_df = data_df
        self.train_df, self.test_df = train_test_split(data_df, test_size=0.3, shuffle=False, random_state=1)
        self.train_df_2, self.test_df_2 = train_test_split(data_df_2, test_size=0.3, shuffle=False, random_state=1)
        data_df.to_csv("data.csv")
        #self.train_df = data_df[data_df['writer_id'] >= 100]
        #self.test_df = data_df[data_df['writer_id'] < 100]
        self.train_df.to_csv("data_train.csv")
        self.test_df.to_csv("data_test.csv")

        self.train = train

    def process_dataset(self, data_root):
        data_df = pd.DataFrame(columns=['img_path', 'label', 'writer_id'])

        img_path_dict = {}
        for img in os.listdir(data_root):
            img_split = img.split('-')
            img_path_dict.setdefault(img_split[0], []).append(img)
        
        for writer_id, key in tqdm(enumerate(img_path_dict)):
            pos_label = img_path_dict[key][0].split('-')[1]
            pos_flag = False
            for img in img_path_dict[key]:
                img_split = img.split('-')
                if img_split[1] != pos_label:
                    pos_flag = True
            if pos_flag:
                for img in img_path_dict[key]:
                    img_path = os.path.join(data_root, img)
                    img_split = img.split('-')
                    label = int(img_split[1])
                    data_df = data_df.append({'img_path': img_path, 'label': label, 'writer_id': writer_id}, ignore_index=True)
                    sig_image, _ = imread_tool(img_path) # normalized
                    sig_image = self.basic_transforms(sig_image)
                    self.img_dict[img_path] = sig_image
        return data_df
        
    def __len__(self):
        if self.train:
            return len(self.train_df_2)
        else:
            return len(self.test_df_2)

    def __getitem__(self, index):
        if self.train:
            self.group = self.train_df.groupby('writer_id')
            self.data_df = self.train_df
            self.data_df_anchor = self.train_df_2
        else:
            self.group = self.test_df.groupby('writer_id')
            self.data_df = self.test_df
            self.data_df_anchor = self.test_df_2
        
        group_writer_id = self.data_df.iloc[index]['writer_id']
        in_class_df = self.group.get_group(group_writer_id)
        # Anchor
        img_path = self.data_df_anchor.iloc[index]['img_path']
        sig_image = self.img_dict[img_path]
        writer_id = self.data_df_anchor.iloc[index]['writer_id']
        label = self.data_df_anchor.iloc[index]['label']

        positive_path, negative_path = None, None

        # positive
        while True:
            #print(img_path, len(in_class_df[in_class_df['label'] == label]))
            if len(in_class_df[in_class_df['label'] == label]) == 1:
                positive_path = img_path
                break
            else:
                sample_df = in_class_df.sample()
                if sample_df['label'].item() == label and sample_df['img_path'].item().split('/')[-1] != img_path.split('/')[-1]:
                    positive_path = sample_df['img_path'].item()
                    #print(positive_path)
                    break
        #positive_sig_image = Image.open(positive_path).convert('RGB')
        #positive_sig_image = imread_tool(positive_path)
        #positive_sig_image = self.basic_transforms(positive_sig_image)
        positive_sig_image = self.img_dict[positive_path]
        #if self.train:
        #    positive_sig_image = self.augment_transforms(positive_sig_image)

        # negative
        while True:
            sample_df = in_class_df.sample()
            if sample_df['label'].item() != label:
                negative_path = sample_df['img_path'].item()
                #print(negative_path)
                break
        #negative_sig_image = Image.open(negative_path).convert('RGB')
        #negative_sig_image = imread_tool(negative_path)
        #negative_sig_image = self.basic_transforms(negative_sig_image)
        negative_sig_image = self.img_dict[negative_path]
        #if self.train:
        #    negative_sig_image = self.augment_transforms(negative_sig_image)
        
        image_pair_0 = torch.cat((sig_image, positive_sig_image), dim=0)
        if self.train: image_pair_0 = torch.squeeze(self.augment_transforms(torch.unsqueeze(image_pair_0, dim=1)))
        image_pair_1 = torch.cat((sig_image, negative_sig_image), dim=0)
        if self.train: image_pair_1 = torch.squeeze(self.augment_transforms(torch.unsqueeze(image_pair_1, dim=1)))
        image_pairs = torch.cat((torch.unsqueeze(image_pair_0, 0), torch.unsqueeze(image_pair_1, 0)), dim=0)
        #print(image_pairs.shape)

        if self.train:
            return image_pairs, torch.tensor([[1],[0]])
        else:
            return image_pairs, torch.tensor([[1],[0]])#, {'Anchor':img_path, 'positive':positive_path, 'negative':negative_path}

class SigDataset_BH_different(Dataset):    
    def __init__(self, opt, path, train=True, image_size=256, mode='normalized', save=False):
        self.path = path
        self.image_size = image_size
        self.basic_transforms = transforms.Compose([transforms.RandomInvert(1.0),
                                                    transforms.Resize((image_size,image_size)), # interpolation=transforms.InterpolationMode.NEAREST
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
        self.shift_transform = transforms.RandomAffine(degrees=0, translate=(0.2, 0.2))
        data_root = path  # ./../BHSig260/Bengali

        self.img_dict = {}
        data_df = self.process_dataset(data_root=data_root, mode=mode)
        data_df_2 = self.process_dataset(data_root='./../BHSig260_demo/Bengali', mode=mode)

        print(f'total {len(data_df)} images !!')
        self.total_data_df = data_df
        pos_df = data_df.groupby('label').get_group(1)
        pos_df_2 = data_df_2.groupby('label').get_group(1)
        
        self.train_df = pos_df
        self.test_df = pos_df
        
        self.train_df_2 = pos_df_2
        self.test_df_2 = pos_df_2

        self.train = train

        self.shift = opt.shift

        self.save = save

    def process_dataset(self, data_root, mode):
        data_df = pd.DataFrame(columns=['img_path', 'label', 'writer_id'])
        for dir in tqdm(os.listdir(data_root)):
            dir_path = os.path.join(data_root, dir)
            if os.path.isdir(dir_path):
                for img in os.listdir(dir_path):
                    img_path = os.path.join(dir_path, img)
                    img_split = img.split('-')
                    label = None
                    if img_split[3] == 'G':
                        label = 1
                    elif img_split[3] == 'F':
                        label = 0
                    assert label is not None
                    data_df = data_df.append({'img_path': img_path, 'label': label, 'writer_id': int(dir)}, ignore_index=True)
                    #sig_image = Image.open(img_path).convert('1')
                    # normalized, cropped
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
                    #sig_image = self.__get_com_cropped__(sig_image)
                    sig_image = self.basic_transforms(sig_image)
                    self.img_dict[img_path] = sig_image
        return data_df
    
    def __get_com_cropped__(self, image):
        '''image is a binary PIL image'''
        image = np.asarray(image)
        com = ndimage.measurements.center_of_mass(image)
        com = np.round(com)
        com[0] = np.clip(com[0], 0, image.shape[0])
        com[1] = np.clip(com[1], 0, image.shape[1])
        X_center, Y_center = int(com[0]), int(com[1])
        c_row, c_col = image[X_center, :], image[:, Y_center]

        x_start, x_end, y_start, y_end = -1, -1, -1, -1

        for i, v in enumerate(c_col):
            v = np.sum(image[i, :])
            if v < 255*image.shape[1]: # there exists text pixel
                if x_start == -1:
                    x_start = i
                else:
                    x_end = i

        for j, v in enumerate(c_row):
            v = np.sum(image[:, j])
            if v < 255*image.shape[0]: # there exists text pixel
                if y_start == -1:
                    y_start = j
                else:
                    y_end = j

        crop_rgb = Image.fromarray(np.asarray(image[x_start:x_end, y_start:y_end]))
        return crop_rgb
    
    def __len__(self):
        if self.train:
            return len(self.train_df_2)
        else:
            return len(self.test_df_2)
    
    def __getitem__(self, index):
        if self.train:
            self.group = self.total_data_df.groupby('writer_id')
            self.data_df = self.train_df
            self.data_df_anchor = self.train_df_2
        else:
            self.group = self.total_data_df.groupby('writer_id')
            self.data_df = self.test_df
            self.data_df_anchor = self.test_df_2
        
        # Anchor
        img_path = self.data_df_anchor.iloc[index]['img_path']
        #print(img_path)
        sig_image = self.img_dict[img_path]
        writer_id = self.data_df_anchor.iloc[index]['writer_id']
        label = self.data_df_anchor.iloc[index]['label']
        
        ###
        group_writer_id = self.data_df_anchor.iloc[index]['writer_id']
        in_class_df = self.group.get_group(group_writer_id)

        positive_path, negative_path = None, None

        # positive
        while True:
            #print(img_path, len(in_class_df[in_class_df['label'] == label]))
            if len(in_class_df[in_class_df['label'] == label]) == 1:
                positive_path = img_path
                break
            else:
                sample_df = in_class_df.sample()
                if sample_df['label'].item() == label and sample_df['img_path'].item().split('/')[-1] != img_path.split('/')[-1]:
                    positive_path = sample_df['img_path'].item()
                    break
        #print(positive_path)
        #positive_sig_image = Image.open(positive_path).convert('RGB')
        #positive_sig_image = imread_tool(positive_path)
        positive_sig_image = self.img_dict[positive_path]
        #if self.train:
        #    positive_sig_image = self.augment_transforms(positive_sig_image)

        # negative
        while True:
            sample_df = in_class_df.sample()
            if sample_df['label'].item() != label:
                negative_path = sample_df['img_path'].item()
                break
        #print(negative_path)
        #negative_sig_image = Image.open(negative_path).convert('RGB')
        #negative_sig_image = imread_tool(negative_path)
        negative_sig_image = self.img_dict[negative_path]
        #if self.train:
        #    negative_sig_image = self.augment_transforms(negative_sig_image)

        # negative 2
        '''
        while True:
            sample_df = in_class_df.sample()
            if sample_df['label'].item() != self.data_df.iloc[index]['label'] and sample_df['img_path'].item() != negative_path:
                negative_path_2 = sample_df['img_path'].item()
                break
        negative_sig_image_2 = self.img_dict[negative_path_2]
        '''
        
        if self.train:
            if self.shift:
                sig_image = self.shift_transform(sig_image)
                positive_sig_image = self.shift_transform(positive_sig_image)
                negative_sig_image = self.shift_transform(negative_sig_image)
            
            #print(sig_image.shape, positive_sig_image.shape, negative_sig_image.shape)
            image_pair_0 = torch.cat((sig_image, positive_sig_image), dim=0)
            image_pair_0 = torch.squeeze(self.augment_transforms(torch.unsqueeze(image_pair_0, dim=1)))
            image_pair_1 = torch.cat((sig_image, negative_sig_image), dim=0)
            image_pair_1 = torch.squeeze(self.augment_transforms(torch.unsqueeze(image_pair_1, dim=1)))
            image_pair_2 = torch.cat((positive_sig_image, sig_image), dim=0)
            image_pair_2 = torch.squeeze(self.augment_transforms(torch.unsqueeze(image_pair_2, dim=1)))
            image_pair_3 = torch.cat((negative_sig_image, sig_image), dim=0)
            image_pair_3 = torch.squeeze(self.augment_transforms(torch.unsqueeze(image_pair_3, dim=1)))
            image_pairs = torch.cat((torch.unsqueeze(image_pair_0, 0),
                                    torch.unsqueeze(image_pair_1, 0),
                                    torch.unsqueeze(image_pair_2, 0),
                                    torch.unsqueeze(image_pair_3, 0)
                                    ), dim=0)
                
            return image_pairs, torch.tensor([[1],[0],[1],[0]])
        else:
            image_pair_0 = torch.cat((sig_image, positive_sig_image), dim=0)
            image_pair_1 = torch.cat((sig_image, negative_sig_image), dim=0)
            image_pairs = torch.cat((torch.unsqueeze(image_pair_0, 0), torch.unsqueeze(image_pair_1, 0)), dim=0)
        

            if self.save:
                return image_pairs, torch.tensor([[1],[0]]), {'Anchor':img_path, 'positive':positive_path, 'negative':negative_path}
            else:
                return image_pairs, torch.tensor([[1],[0]])

class single_test_dataset(Dataset):
    def __init__(self, anchor_path, positive_path, negative_path, image_size=256, mode='normalized'):
        self.anchor_path = anchor_path
        self.positive_path = positive_path
        self.negative_path = negative_path
        self.basic_transforms = transforms.Compose([transforms.RandomInvert(1.0),
                                                    transforms.Resize((image_size,image_size)),
                                                    transforms.ToTensor(),
                                                    #transforms.Normalize(mean=0.5,std=0.25),
                                                    #transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                                                    #transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5]),
                                                    #transforms.Grayscale(num_output_channels=1)
                                                    ])
        self.img_dict = {}
        path_dict = {'Anchor':anchor_path, 'positive':positive_path, 'negative':negative_path}
        for item in ['Anchor', 'positive', 'negative']:
            print(path_dict[item])
            # normalized, cropped
            if mode == 'normalized':
                sig_image, _ = imread_tool(path_dict[item])
                #sig_image.save('img_pre_{}.png'.format(item))
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
            sig_image = self.basic_transforms(sig_image)
            self.img_dict[item] = sig_image
    def __len__(self):
        return 1
    
    def __getitem__(self, index):
        # 'Anchor', 'positive', 'negative'
        sig_image = self.img_dict['Anchor']
        positive_sig_image = self.img_dict['positive']
        negative_sig_image = self.img_dict['negative']
        image_pair_0 = torch.cat((sig_image, positive_sig_image), dim=0)
        image_pair_1 = torch.cat((sig_image, negative_sig_image), dim=0)
        image_pairs = torch.cat((torch.unsqueeze(image_pair_0, 0), torch.unsqueeze(image_pair_1, 0)), dim=0)

        return image_pairs, torch.tensor([[1],[0]]), {'Anchor':self.anchor_path, 'positive':self.positive_path, 'negative':self.negative_path}

class SigDataset_GPDS(Dataset):
    def __init__(self, opt, path, train=True, image_size=256, mode='normalized', save=False, partition=False, test_idx=None):
        self.mode = True # sample mode or not
        self.path = path
        self.image_size = image_size
        trans_list = [transforms.RandomInvert(1.0),
                    transforms.Resize((image_size,image_size),), # interpolation=transforms.InterpolationMode.NEAREST
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

        self.shift_transform = transforms.Compose([transforms.RandomAffine(degrees=(1,1), interpolation=transforms.InterpolationMode.BILINEAR),
                                                    transforms.RandomAffine(degrees=(-1,-1),),
        ])
        
        data_root = path  # ./../OSV/SignatureGPDSSyntheticSignaturesManuscriptsv/firmasSINTESISmanuscritas/
        
        # 24 pos, 30 neg
        pos_set = set()
        for x in range(1,24+1):
            for y in range(1,24+1):
                if x != y and (y,x) not in pos_set:
                    pos_set.add((x, y))
        neg_set = set((x, y) for x in range(1,24+1) for y in range(1,30+1))
        
        if self.mode:
            data_df = pd.DataFrame(columns=['img_path', 'label', 'writer_id'])
            FLAG = False
        else:
            # read CSV file (for full mode)
            FLAG = os.path.exists(os.path.join(data_root, "pairs.csv"))
            data_df_pair = pd.DataFrame(columns=['item_0','item_1', 'label', 'writer_id'])
            if FLAG:
                data_df_pair = pd.read_csv(os.path.join(data_root, "pairs.csv"))

        # sort dataset list
        self.img_dict = {}
        lst = os.listdir(data_root)
        lst.sort()
        lst_final = lst[test_idx:test_idx+1] if test_idx is not None else lst#[:10] # subset
        
        # json file: test set
        # json_file = open('test_GPDS_1.json')
        # test_idx_set = set(json.loads(json_file.read()))
        # json_file.close()
        
        for dir in tqdm(lst_final): # 800
        # for idx, dir in tqdm(enumerate(lst_final)): # 800
            dir_path = os.path.join(data_root, dir)
            
            if os.path.isdir(dir_path) and (int(dir)>800 if train else int(dir)<=800):
            # if os.path.isdir(dir_path) and idx+1 in test_idx_set:
                # print(idx+1)
                single_dict = {}
                for img in os.listdir(dir_path):
                    if img.endswith('.png') or img.endswith('.jpg'):
                        img_path = os.path.join(dir_path, img)
                        img_split = img.split('-')
                        label = None
                        if img_split[0] == 'c': # G
                            label = 1
                        elif img_split[0] == 'cf': # F
                            label = 0
                        assert label is not None
                        index = int(img_split[-1][:-4])
                        if self.mode:
                            # sample mode
                            data_df = data_df.append({'img_path': img_path, 'label': label, 'writer_id': int(dir)}, ignore_index=True)
                        else:
                            # full mode
                            single_dict[(label, index)] = img_path
                        if mode == 'normalized':
                            sig_image, _ = imread_tool(img_path, binarization=True)
                        elif mode == 'cropped':
                            _, sig_image = imread_tool(img_path, binarization=True)
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

                        sig_image = self.basic_transforms(sig_image)
                        self.img_dict[img_path] = sig_image
                        
                # full mode
                if not FLAG and not self.mode:
                    for item in list(pos_set):
                        first, second = item
                        path_first = single_dict[(1, first)]
                        path_second = single_dict[(1, second)]
                        assert path_first is not None
                        assert path_second is not None
                        data_df_pair = data_df_pair.append({'item_0': path_first, 'item_1': path_second, 'label': 1, 'writer_id': int(dir)}, ignore_index=True)
                    for item in list(neg_set):
                        first, second = item
                        path_first = single_dict[(1, first)]
                        path_second = single_dict[(0, second)]
                        assert path_first is not None
                        assert path_second is not None
                        data_df_pair = data_df_pair.append({'item_0': path_first, 'item_1': path_second, 'label': 0, 'writer_id': int(dir)}, ignore_index=True)

        if self.mode: # sample mode
            print(f'total {len(data_df)} images !!')
            self.total_data_df = data_df
            pos_df = data_df.groupby('label').get_group(1)
            self.train_df = pos_df[pos_df['writer_id'] > 800]
            self.test_df = pos_df[pos_df['writer_id'] <= 800]
            #self.train_df = pos_df
            #self.test_df = pos_df
        else: # full mode
            print(f'total {len(data_df_pair)} pairs !!')
            self.total_data_df = data_df_pair
            self.test_df = data_df_pair[data_df_pair['writer_id'] <= 10]
            if not FLAG:
                data_df_pair.to_csv(os.path.join(data_root, "pairs.csv"))
   
        #self.train_df = pos_df[pos_df['writer_id'] > 50]    
        # load testing part
        #self.test_df = pos_df[pos_df['writer_id'] <= 50]
        
        self.train = train
        self.shift = opt.shift
        self.save = save

    def __len__(self):
        if self.train:
            return len(self.train_df)
        else:
            return 2*len(self.test_df) if self.mode else len(self.test_df)
    
    def __getitem__(self, index):
        if self.mode:
            if self.train:
                self.group = self.total_data_df.groupby('writer_id')
                self.data_df = self.train_df
                index_anchor = index
            else:
                self.group = self.total_data_df.groupby('writer_id')
                self.data_df = self.test_df
                index_anchor = index//2
            # Anchor
            img_path = self.data_df.iloc[index_anchor]['img_path']
            sig_image = self.img_dict[img_path]
            #if self.train:
            #    sig_image = self.augment_transforms(sig_image)
            writer_id = self.data_df.iloc[index_anchor]['writer_id']
            label = self.data_df.iloc[index_anchor]['label']
            
            ###
            group_writer_id = self.data_df.iloc[index_anchor]['writer_id']
            in_class_df = self.group.get_group(group_writer_id)

            positive_path, negative_path = None, None

            # positive
            if self.train or index%2 == 0:
                while True:
                    #print(img_path, len(in_class_df[in_class_df['label'] == label]))
                    if len(in_class_df[in_class_df['label'] == label]) == 1:
                        positive_path = img_path
                        break
                    else:
                        sample_df = in_class_df.sample()
                        if sample_df['label'].item() == label and sample_df['img_path'].item() != img_path:
                            positive_path = sample_df['img_path'].item()
                            break
                positive_sig_image = self.img_dict[positive_path]
                #if self.train:
                #    positive_sig_image = self.augment_transforms(positive_sig_image)

            # negative
            if self.train or index%2 != 0:
                while True:
                    sample_df = in_class_df.sample()
                    if sample_df['label'].item() != label:
                        negative_path = sample_df['img_path'].item()
                        break
                negative_sig_image = self.img_dict[negative_path]
                #if self.train:
                #    negative_sig_image = self.augment_transforms(negative_sig_image)
            
            if self.train:
                if self.shift:
                    sig_image = self.shift_transform(sig_image)
                    positive_sig_image = self.shift_transform(positive_sig_image)
                    negative_sig_image = self.shift_transform(negative_sig_image)
                
                #print(sig_image.shape, positive_sig_image.shape, negative_sig_image.shape)
                image_pair_0 = torch.cat((sig_image, positive_sig_image), dim=0)
                image_pair_0 = torch.squeeze(\
                    self.augment_transforms(torch.unsqueeze(image_pair_0, dim=1)))
                image_pair_1 = torch.cat((sig_image, negative_sig_image), dim=0)
                image_pair_1 = torch.squeeze(\
                    self.augment_transforms(torch.unsqueeze(image_pair_1, dim=1)))
                image_pair_2 = torch.cat((positive_sig_image, sig_image), dim=0)
                image_pair_2 = torch.squeeze(\
                    self.augment_transforms(torch.unsqueeze(image_pair_2, dim=1)))
                image_pair_3 = torch.cat((negative_sig_image, sig_image), dim=0)
                image_pair_3 = torch.squeeze(\
                    self.augment_transforms(torch.unsqueeze(image_pair_3, dim=1)))
                
                image_pairs = torch.cat((torch.unsqueeze(image_pair_0, 0),
                                        torch.unsqueeze(image_pair_1, 0),
                                        torch.unsqueeze(image_pair_2, 0),
                                        torch.unsqueeze(image_pair_3, 0)
                                        ), dim=0)
                    
                return image_pairs, torch.tensor([[1],[0],[1],[0]])
            else:
                if index%2 == 0:
                    image_pair_final = torch.cat((sig_image, positive_sig_image), dim=0)
                    label_final = torch.tensor([[1]])
                else:
                    image_pair_final = torch.cat((sig_image, negative_sig_image), dim=0)
                    label_final = torch.tensor([[0]])
                
                if self.save:
                    #return image_pair_final, label_final, {'Anchor':img_path, 'positive':positive_path, 'negative':negative_path}
                    return image_pair_final, label_final, {'Anchor':img_path, 'ref':positive_path if index%2 == 0 else negative_path}
                else:
                    return image_pair_final, label_final
        else:
            if self.train:
                self.data_df = self.train_df
            else:
                self.data_df = self.test_df
            # full mode
            # Anchor
            img_path_0 = self.data_df.iloc[index]['item_0']
            sig_image_0 = self.img_dict[img_path_0]
            img_path_1 = self.data_df.iloc[index]['item_1']
            sig_image_1 = self.img_dict[img_path_1]
            label = self.data_df.iloc[index]['label']
            writer_id = self.data_df.iloc[index]['writer_id']
            image_pair = torch.cat((sig_image_0, sig_image_1), dim=0)
            '''
            if self.train:
                image_pair = torch.squeeze(self.augment_transforms(torch.unsqueeze(image_pair, dim=1)))
            '''
            #return image_pair, torch.tensor([[label]])
            return image_pair, torch.tensor([[label]]), writer_id

if __name__ == '__main__':
    img_path = './../BHSig260/Bengali/012/B-S-12-G-17.tif'
    image = Image.open(img_path)
    #print(image.mode)
    #print(list(image.getdata()))
    image = image.convert('L')
    image = np.asarray(image)
    inputSize = image.shape
    normalized_image, cropped_image = normalize_image(image.astype(np.uint8), inputSize) # RETURN: normalized, cropped
    print(normalized_image.tolist())