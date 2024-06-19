import torch
import torch.nn as nn
import torchvision
import numpy as np
from timm.models.vision_transformer import Block
from timm.models.layers import to_2tuple
from .tcformer_module.tcformer_layers import TCBlock, OverlapPatchEmbed, CTM, TokenConv

class PatchEmbed_v0(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            norm_layer=None,
            flatten=True,
            bias=True,
    ):
        super().__init__()
        #
        stride_size = (img_size - patch_size) // 13 # calculate stride size
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        #
        stride_size = to_2tuple(stride_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (((img_size[0]-patch_size[0]) // stride_size[0]) +1, ((img_size[1]-patch_size[1]) // stride_size[1]) +1)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

class T2T_module(nn.Module):
    """
    Tokens-to-Token encoding module
    """
    def __init__(self, img_size=224, tokens_type='transformer', in_chans=1, embed_dim=768, token_dim=64, patch_size=None):
        super().__init__()

        if tokens_type == 'transformer':
            print('adopt transformer encoder for tokens-to-token')
            self.soft_split0 = nn.Unfold(kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))
            self.soft_split1 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.soft_split2 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

            self.attention1 = Block(dim=token_dim, num_heads=1, mlp_ratio=1.0) #in_dim=token_dim
            self.attention2 = Block(dim=token_dim, num_heads=1, mlp_ratio=1.0)
            self.project = nn.Linear(token_dim * 3 * 3, embed_dim)

            self.proj_before_attention1 = nn.Linear(in_chans * 7 * 7, token_dim)
            self.proj_before_attention2 = nn.Linear(token_dim * 3 * 3, token_dim)
        
        elif tokens_type == 'tcformer':
            print('adopt tcformer encoder for tokens-to-token')
            self.soft_split0 = nn.Unfold(kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))
            self.soft_split1 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.soft_split2 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            
            self.ctm1 = CTM(0.25, token_dim, token_dim, 5)
            self.ctm2 = CTM(0.25, token_dim, token_dim, 5)

            self.attention1 = TCBlock(dim=token_dim, num_heads=1, mlp_ratio=1.0)
            self.attention2 = TCBlock(dim=token_dim, num_heads=1, mlp_ratio=1.0)
            self.attention3 = TCBlock(dim=token_dim, num_heads=1, mlp_ratio=1.0)
            self.project = nn.Linear(token_dim, embed_dim)

            self.proj_before_attention1 = nn.Linear(in_chans * 7 * 7, token_dim)
            self.proj_before_attention2 = nn.Linear(token_dim * 3 * 3, token_dim)

        elif tokens_type == 'convolution':  # just for comparison with conolution, not our model
            # for this tokens type, you need change forward as three convolution operation
            print('adopt convolution layers for tokens-to-token')
            self.soft_split0 = nn.Conv2d(3, token_dim, kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))  # the 1st convolution
            self.soft_split1 = nn.Conv2d(token_dim, token_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) # the 2nd convolution
            self.project = nn.Conv2d(token_dim, embed_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) # the 3rd convolution

        self.num_patches = (img_size // (4 * 2 * 2)) * (img_size // (4 * 2 * 2))  # there are 3 sfot split, stride are 4,2,2 seperately

    def forward(self, x):
        # step0: soft split
        x = self.soft_split0(x).transpose(1, 2)
        x = self.proj_before_attention1(x)
        # iteration1: re-structurization/reconstruction
        x = self.attention1(x)
        B, new_HW, C = x.shape
        x = x.transpose(1,2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        # iteration1: soft split
        x = self.soft_split1(x).transpose(1, 2)
        x = self.proj_before_attention2(x)
        # iteration2: re-structurization/reconstruction
        x = self.attention2(x)
        B, new_HW, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        # iteration2: soft split
        x = self.soft_split2(x).transpose(1, 2)

        # final tokens
        x = self.project(x)

        return x

    def forward_tc(self, x):
        x = self.soft_split0(x).transpose(1, 2)
        x = self.proj_before_attention1(x)
        H, W = 56, 56
        B, N, _ = x.shape
        device = x.device
        idx_token = torch.arange(N)[None, :].repeat(B, 1).to(device)
        agg_weight = x.new_ones(B, N, 1)
        token_dict = {'x': x,
                      'token_num': N,
                      'map_size': [H, W],
                      'init_grid_size': [H, W],
                      'idx_token': idx_token,
                      'agg_weight': agg_weight}
        
        token_dict = self.attention1(token_dict)
        token_dict = self.ctm1(token_dict)
        token_dict = self.attention2(token_dict)
        token_dict = self.ctm2(token_dict)
        token_dict = self.attention3(token_dict)

        # final tokens
        token_dict['x'] = self.project(token_dict['x'])

        return token_dict['x']

class T2T_tcformer_module(nn.Module):
    def __init__(self, img_size=224, in_chans=1, embed_dim=768, token_dim=64, patch_size=None):
        super().__init__()
        self.proj = nn.Conv2d(1, token_dim, kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))
        self.ctm1 = CTM(0.25, token_dim, token_dim, 5)
        self.ctm2 = CTM(0.25, token_dim, token_dim, 5)

        self.attention1 = TCBlock(dim=token_dim, num_heads=1, mlp_ratio=1.0)
        self.attention2 = TCBlock(dim=token_dim, num_heads=1, mlp_ratio=1.0)
        self.attention3 = TCBlock(dim=token_dim, num_heads=1, mlp_ratio=1.0)
        self.head = nn.Linear(token_dim, embed_dim)
    
    def forward(self, x):
        # conventional soft split
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        H, W = 56, 56
        B, N, _ = x.shape
        device = x.device
        idx_token = torch.arange(N)[None, :].repeat(B, 1).to(device)
        agg_weight = x.new_ones(B, N, 1)
        token_dict = {'x': x,
                      'token_num': N,
                      'map_size': [H, W],
                      'init_grid_size': [H, W],
                      'idx_token': idx_token,
                      'agg_weight': agg_weight}
        
        token_dict = self.attention1(token_dict)
        # merge: 3136 -> 784
        token_dict = self.ctm1(token_dict)
        token_dict = self.attention2(token_dict)
        # merge: 784 -> 196
        token_dict = self.ctm2(token_dict)
        token_dict = self.attention3(token_dict)

        # final tokens
        token_dict['x'] = self.head(token_dict['x'])

        return token_dict['x']

class T2T_tcformer_module_simple(nn.Module):
    def __init__(self, img_size=224, in_chans=1, embed_dim=768, token_dim=64, patch_size=None):
        super().__init__()
        self.proj = nn.Conv2d(1, token_dim, kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))
        #self.proj2 = nn.Conv2d(token_dim, token_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        #self.proj = nn.Unfold(kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))
        #self.ctm1 = CTM(0.25, in_chans * 7 * 7, token_dim, 5)
        self.ctm1 = CTM(0.25, token_dim, token_dim, 5)
        self.ctm2 = CTM(0.25, token_dim, token_dim, 5)

        self.attention = TCBlock(dim=token_dim, num_heads=1, mlp_ratio=1.0)
        self.head = nn.Linear(token_dim, embed_dim)
        
        #self.proj_before_attention1 = nn.Linear(in_chans * 7 * 7, token_dim)
    
    def forward(self, x):
        # conventional soft split
        #x = self.proj(x).transpose(1, 2)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        H, W = 56, 56
        B, N, _ = x.shape
        device = x.device
        idx_token = torch.arange(N)[None, :].repeat(B, 1).to(device)
        agg_weight = x.new_ones(B, N, 1)
        token_dict = {'x': x,
                      'token_num': N,
                      'map_size': [H, W],
                      'init_grid_size': [H, W],
                      'idx_token': idx_token,
                      'agg_weight': agg_weight}
        
        # merge: 3136 -> 784 -> 196
        #token_dict, _ = self.ctm0(token_dict)
        token_dict, _ = self.ctm1(token_dict)
        token_dict, _ = self.ctm2(token_dict)

        # final tokens
        token_dict['x'] = self.head(token_dict['x'])

        return token_dict['x']

class T2T_tcformer_module_simple_v2(nn.Module):
    def __init__(self, img_size=224, in_chans=1, embed_dim=768, token_dim=64, patch_size=None):
        super().__init__()
        self.proj = nn.Conv2d(1, token_dim, kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))
        self.ctm0 = CTM(0.25, token_dim, token_dim, 5)
        self.ctm1 = CTM(0.25, token_dim, token_dim, 5)
        self.ctm2 = CTM(0.25, token_dim, token_dim, 5)

        self.attention = TCBlock(dim=token_dim, num_heads=1, mlp_ratio=1.0)
        self.head = nn.Linear(token_dim, embed_dim)
        
        #self.proj_before_attention1 = nn.Linear(in_chans * 7 * 7, token_dim)
    
    def forward(self, x):
        # conventional soft split
        #x = self.proj(x).transpose(1, 2)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        H, W = 56, 56
        B, N, _ = x.shape
        device = x.device
        idx_token = torch.arange(N)[None, :].repeat(B, 1).to(device)
        agg_weight = x.new_ones(B, N, 1)
        token_dict = {'x': x,
                      'token_num': N,
                      'map_size': [H, W],
                      'init_grid_size': [H, W],
                      'idx_token': idx_token,
                      'agg_weight': agg_weight}
        
        # merge: 3136 -> 784 -> 196 -> 49
        token_dict, _ = self.ctm0(token_dict)
        token_dict, _ = self.ctm1(token_dict)
        token_dict, _ = self.ctm2(token_dict)

        # final tokens
        token_dict['x'] = self.head(token_dict['x'])

        return token_dict['x']

class DeformableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(DeformableConv2d, self).__init__()
        
        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        
        self.offset_conv = nn.Conv2d(in_channels, 
                                     2 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size, 
                                     stride=stride,
                                     padding=self.padding, 
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        
        self.modulator_conv = nn.Conv2d(in_channels, 
                                     1 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size, 
                                     stride=stride,
                                     padding=self.padding, 
                                     bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)
        
        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      bias=bias)

    def forward(self, x):
        #h, w = x.shape[2:]
        #max_offset = max(h, w)/4.

        offset = self.offset_conv(x)#.clamp(-max_offset, max_offset)
        #print(offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        
        x = torchvision.ops.deform_conv2d(input=x, offset=offset, weight=self.regular_conv.weight, bias=self.regular_conv.bias, \
                                          padding=self.padding, mask=modulator, stride=self.stride,)
        return x

class PatchEmbed_wdeform(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, patch_count=14, in_chans=1, embed_dim=768, with_norm=False):
        super().__init__()  
        patch_stride = img_size // patch_count
        patch_pad = (patch_stride * (patch_count - 1) + patch_size - img_size) // 2
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = patch_count * patch_count
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj_deform = DeformableConv2d(in_chans, embed_dim // 2, kernel_size=patch_size, stride=patch_stride, padding=patch_pad)
        self.proj = nn.Conv2d(in_chans, embed_dim // 2, kernel_size=patch_size, stride=patch_stride, padding=patch_pad)
        if with_norm:
            self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x_reg = self.proj(x)
        x_deform = self.proj_deform(x)
        x = torch.cat((x_reg, x_deform), 1)
        x = x.flatten(2).transpose(1, 2)
        if hasattr(self, "norm"):
            x = self.norm(x)
        assert x.shape[1] == self.num_patches
        return x

class Part_Attention(nn.Module):
    def __init__(self):
        super(Part_Attention, self).__init__()

    def forward(self, x):
        length = len(x)
        last_map = x[0]
        for i in range(1, length):
            last_map = torch.matmul(x[i], last_map)
        last_map = last_map[:,:,0,1:]

        _, max_inx = last_map.max(2)
        return _, max_inx

if __name__ == '__main__':
    from tcformer_module.tcformer_layers import TCBlock, OverlapPatchEmbed, CTM
    device = torch.device("cuda")

    x = torch.randn([2, 1, 224, 224])
    x = x.to(device)

    #model = T2T_module(tokens_type='tcformer') # tokens_type='tcformer'
    #model = PatchEmbed_wdeform()
    model = T2T_tcformer_module_simple()
    model.cuda()
    out = model(x)
    #out = model.forward_tc(x)

    print(out.shape)