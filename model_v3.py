import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
import numpy as np
import os

from models.emd import Sinkhorn, SinkhornDistance_new
from models.gw import GW_distance_uniform
from models.gw_v2 import GW_distance_uniform_v2

from einops import rearrange

from module.loss import ContrastiveLoss

def mar_different(input_pair):
    batch_size, _, h, w = input_pair.shape # [batch_size, 2, h, w]
    eps = 1e-3
    x1 = input_pair[:, 0, :, :] + eps
    x2 = input_pair[:, 1, :, :] + eps

    PATCH_SIZE = 16
    x1_patches = x1.unfold(1, PATCH_SIZE, PATCH_SIZE).unfold(2, PATCH_SIZE, PATCH_SIZE)
    x2_patches = x2.unfold(1, PATCH_SIZE, PATCH_SIZE).unfold(2, PATCH_SIZE, PATCH_SIZE)

    x1_patches = rearrange(x1_patches, 'b x y px py -> b (x y) (px py)').contiguous().sum(dim=-1)
    x2_patches = rearrange(x2_patches, 'b x y px py -> b (x y) (px py)').contiguous().sum(dim=-1)
    u, v = (x1_patches / x1.sum()), (x2_patches / x2.sum())
    return u, v    

def mar_uniform_v2(input_pair):
    u, v = mar_different(input_pair)
    
    def normalize(input):
        min_mask = input.min() + 1e-5
        input = input.ge(min_mask).float()
        binary = torch.where(input > 0, input, torch.finfo(torch.float).min)
        output = F.softmax(binary, dim=-1)
        return output
    
    return normalize(u), normalize(v)

def EMD_batched(input_1, input_2, dis_type = 'cos', mar_type = 'uniform', input_pair = None, eps=5e-2):
    B, S1, C = input_1.shape
    B, S2, C = input_2.shape
    
    GW = False
    if GW:
        dist_gw, T_gw, C_gw = GW_distance_uniform(input_1.transpose(2,1), input_2.transpose(2,1))
    else:
        dist_gw=0
    
    # EMD
    if mar_type == 'uniform':
        u = torch.zeros(B, S2).fill_(1. / S2).cuda()
        v = torch.zeros(B, S1).fill_(1. / S1).cuda()
    elif mar_type == 'different': # calulate marginal distribution by pixel value
        u, v = mar_different(input_pair)
    elif mar_type == 'uniform_v2':
        u, v = mar_uniform_v2(input_pair)

    if dis_type == 'cos':
        sim = F.cosine_similarity(input_1[..., None, :, :], input_2[..., :, None, :], dim=-1)
        #sigma = 1e3
        #sim = torch.exp(sigma * (sim - 1))
        dis = 1.0 - sim
    elif dis_type == 'l2':
        dis = torch.cdist(input_2, input_1, p=2)
        #dis = torch.norm(input_1[..., None, :, :] - input_2[..., :, None, :], p=2, dim=3, keepdim=False)
        sim = 1.0 / (1.0 + dis)

    sinkhorn = SinkhornDistance_new(eps=eps, max_iter=100) # 5e-2, 5e-3, 5e-7
    dist, P, C = sinkhorn(u, v, dis)
    
    total_dist = dist + dist_gw
    return torch.unsqueeze(total_dist, 1), P, C

class ViT_for_OSV_DSNet(nn.Module):
    def __init__(self, opt):
        super(ViT_for_OSV_DSNet, self).__init__()
        from models.dsnet import dsnet
        self.model = dsnet()
        #checkpoint = torch.load("pretrain weight")
        #self.model.load_state_dict(checkpoint, strict=False)
        ###
        self.model.patch_embed.proj1 = nn.Conv2d(1, 48, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        
        self.model.head = nn.Identity()
        
        self.head_str = nn.Linear(384, 96) if opt.emd else nn.Identity()
        
        self.pdist = nn.PairwiseDistance(p=2, keepdim = True)
        
        self.dmloss = ContrastiveLoss()
        self.bceloss = nn.BCELoss()
        
        self.opt = opt
        
    def forward_one(self, x, masks):
        # stage 1
        x = self.model.patch_embed(x)
        B, H, W, C = x.shape
        x = x + self.model._get_pos_embed(self.model.pos_embed1, self.model.num_patches1, H, W) 
        x, hx, lx = self.model.blocks1(x)
        
        # stage 2
        x = x.permute(0, 3, 1, 2)       
        x = self.model.patch_embed2(x)
        B, H, W, C = x.shape
        x = x + self.model._get_pos_embed(self.model.pos_embed2, self.model.num_patches2, H, W) 
        x, hx, lx = self.model.blocks2(x)
        
        # stage 3
        x = x.permute(0, 3, 1, 2)  
        x = self.model.patch_embed3(x)
        B, H, W, C = x.shape
        x = x + self.model._get_pos_embed(self.model.pos_embed3, self.model.num_patches3, H, W) 
        x, hx, lx = self.model.blocks3(x)
        
        # stage 4
        x = x.permute(0, 3, 1, 2)  
        x = self.model.patch_embed4(x)
        B, H, W, C = x.shape
        x = x + self.model._get_pos_embed(self.model.pos_embed4, self.model.num_patches4, H, W) 
        x, hx, lx = self.model.blocks4(x)
        x = x.flatten(1,2)
        
        min_mask = masks.min() + 1e-5
        masks = masks.ge(min_mask)
        x_patch_new_ = nn.utils.rnn.pad_sequence([r[m] for r, m in zip(x, masks)], batch_first=True)
        
        x_patch_new_ = self.head_str(x_patch_new_)
        x = self.model.norm(x)
        
        x_reg = x_patch_new_
        return x, x_patch_new_, masks, x_reg # training times
    
    def forward(self, x, Y):
        batch_size, _, h, w = x.shape # [batch_size, 2, h, w]
        x1 = x[:, 0, :, :].unsqueeze(1)
        x2 = x[:, 1, :, :].unsqueeze(1)
        
        ###
        x_mask_tmp = F.interpolate(x, size=[7,7], mode='area')
        #x_mask_tmp = F.interpolate(x, size=[14,14], mode='area')
        x_mask_tmp = rearrange(x_mask_tmp, 'b c h w -> b c (h w)')
        mask_1, mask_2 = x_mask_tmp[:, 0,], x_mask_tmp[:, 1,]
        
        #out1, masks1, attn_weight1 = self.forward_one(x1, mask_1)
        #out2, masks2, attn_weight2 = self.forward_one(x2, mask_2)
        
        out1, out1_, masks1, x_reg1 = self.forward_one(x1, mask_1)
        out2, out2_, masks2, x_reg2 = self.forward_one(x2, mask_2)
        #class_token_1 = out1.mean(1)
        #class_token_2 = out2.mean(1)
        
        # add (linear) head
        class_token_1 = self.model.head(out1.mean(1))
        class_token_2 = self.model.head(out2.mean(1))
        
        #class_token_1 = out1_.mean(1)
        #class_token_2 = out2_.mean(1)

        #class_token_1, out1_ = out1[:, 0], out1[:, 1:]
        #class_token_2, out2_ = out2[:, 0], out2[:, 1:]
        
        B, S1, C = out1_.shape
        B, S2, C = out2_.shape
        preds_str = 0
        eps = 5e-2 if self.training else 5e-8 # 5e-2, 5e-4, 5e-6
        #eps = 5e-8
        if self.opt.emd and S1 != 0 and S2 != 0: # train with emd, if no only token selection
            preds_str, transport, cost = EMD_batched(out1_, out2_, dis_type=self.opt.dis_type, mar_type = self.opt.mar_type, input_pair=x, eps=eps) # mar_type = 'uniform_v2'
        
        if self.opt.gol_dis == 'l2':
            preds_gol = self.pdist(class_token_1, class_token_2)
        elif self.opt.gol_dis == 'cos':
            preds_gol = 1 - F.cosine_similarity(class_token_1[:,None,:], class_token_2[:,None,:], dim=-1)
        else:
            return NameError
        
        if self.opt.emd:
            preds_str *= self.opt.temp
            out = preds_gol + preds_str
            #out = preds_str
        else:
            out = preds_gol
        
        final_loss = self.dmloss(out, Y.float()) # original
        return out, final_loss
        #return out, final_loss, class_token_1-class_token_2
        #return (preds_gol, preds_str), final_loss
    
    def save_to_file(self, file_path: str) -> None:
        torch.save(self.state_dict(), file_path)
