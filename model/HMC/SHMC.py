import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
import torch.utils.checkpoint as checkpoint
from functools import partial
import pywt
from typing import Callable, Optional   
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
torch.manual_seed(123)






class ECAAttention(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        """
        
        """
        super(ECAAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  

        
        t = int(abs((math.log2(channels) + b) / gamma))
        k_size = t if t % 2 else t + 1  

        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (B, C, H, W)
        y = self.avg_pool(x)  # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(-1, -2)  # (B, 1, C)
        y = self.conv(y)  # (B, 1, C)
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)  # (B, C, 1, 1)
        return y  

    




class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16, 
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner, 
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
       
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError


        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    
    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y
   
    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out



class FIM(nn.Module):
    
    def __init__(self, dim): 
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()
        self.pos_enc1 = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.pos_enc2 = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.conv_spatial1_1 = nn.Conv2d(dim, dim, 1, padding=0, bias=True)
        self.conv_spatial1_2 = nn.Conv2d(dim, dim, 1, padding=0, bias=True)
        self.conv_spatial2_1 = nn.Conv2d(dim, dim, 1, padding=0, bias=True)
        self.conv_spatial2_2 = nn.Conv2d(dim, dim, 1, padding=0, bias=True)

    def forward(self, x):
        x1 = self.pos_enc1(x) + x
        x1 = self.conv_spatial1_1(x1)
        x1 = self.sigmoid(self.avg_pool(x1)) * x1
        x1 = self.conv_spatial1_2(x1)

        x2 = self.conv_spatial2_1(x)
        x2 = self.sigmoid(self.avg_pool(x2)) * x2
        x2 = self.conv_spatial2_2(x2)
        x2 = self.pos_enc2(x2) + x2

        out =  x1 + x2
        return out
    


    
class AdaptiveChannelGate(nn.Module): 
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, h, w = x.size()
        
        y = self.avg_pool(x).view(b, c)
        
        gate_weights = self.gate(y).view(b, c, 1, 1)
        return x * gate_weights   
    
    


class MambaB(nn.Module):
    def __init__(
            self,
            dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            mlp_ratio: float = 2.,
            reduction: int = 16, 
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(dim)
        self.self_attention = SS2D(d_model=dim, d_state=d_state,expand=mlp_ratio,dropout=attn_drop_rate, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.skip_scale= nn.Parameter(torch.ones(dim)) 
        self.ACG = AdaptiveChannelGate(dim , reduction=reduction) 
        self.FAMblk = FAM(dim)
        self.ln_2 = nn.LayerNorm(dim)
        self.skip_scale2 = nn.Parameter(torch.ones(dim))


    def forward(self, input, x_size):
        B, L, C = input.shape
        input = input.view(B, *x_size, C).contiguous()  # [B,H,W,C]
        x = self.ln_1(input)
        x = input*self.skip_scale + self.drop_path(self.self_attention(x))
        y = self.FAMblk(self.ACG(self.ln_2(x).permute(0, 3, 1, 2).contiguous())).permute(0, 2, 3, 1).contiguous() 
        x = x*self.skip_scale2 + y
        x = x.view(B, -1, C).contiguous()
        return x
    


class ConvFFN(nn.Module):
    """
    Convolutional Feed Forward Network
    Structure: Conv -> GELU -> Conv
    """
    def __init__(self, in_channels, hidden_channels=None, out_channels=None, 
                 kernel_size=1, stride=1, padding=0, dropout=0.0):
        super(ConvFFN, self).__init__()
        
       
        if hidden_channels is None:
            hidden_channels = in_channels * 4
            
        
        if out_channels is None:
            out_channels = in_channels
        
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 
                              kernel_size=kernel_size, stride=stride, padding=padding)
        

        self.activation = nn.GELU()
        
     
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, 
                              kernel_size=kernel_size, stride=stride, padding=padding)
    
    def forward(self, x):
        """
       
        """
        # Conv -> GELU -> Dropout -> Conv
        x = self.conv1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return x


class HMCB(nn.Module):      
    def __init__(self,
                 dim: int = 0, 
                 drop_path: float = 0, 
                 norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
                 attn_drop_rate: float = 0, 
                 d_state: int = 16, 
                 mlp_ratio: float = 2., 
                 reduction: int = 16, 
                 drop: float = 0,
                 **kwargs):
        super().__init__()
        self.dim = dim
        self.drop_path = drop_path
        self.norm_layer = norm_layer
        self.attn_drop_rate = attn_drop_rate                
        self.d_state = d_state
        self.mlp_ratio = mlp_ratio
        self.reduction = reduction
        self.drop = drop
        
        self.drop_path = DropPath(drop_path)
        self.skip_scale1 = nn.Parameter(torch.ones(dim)) 
        self.skip_scale2 = nn.Parameter(torch.ones(dim)) 
        self.ln_1 = norm_layer(dim) 
        self.ln_2 = norm_layer(dim)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)
        
       
        self.convffn = ConvFFN(in_channels=dim,  out_channels=dim, kernel_size=1, stride=1, padding=0, dropout=drop)
     
        self.mamba_b = MambaB(dim=dim, d_state=d_state,mlp_ratio=mlp_ratio, attn_drop_rate=attn_drop_rate, **kwargs)
       
        self.dwconv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        
       
        self.channel_interaction = ECAAttention(dim)
        
        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(dim, dim // 16, kernel_size=1),
            nn.BatchNorm2d(dim // 16),
            nn.GELU(),
            nn.Conv2d(dim // 16, 1, kernel_size=1)
        )
        
    def forward(self, x, x_size): 
        B, L, C = x.shape
        assert L == x_size[0] * x_size[1], "flatten img_tokens has wrong size"
        input = x.view(B, *x_size, C).contiguous()  # [B,H,W,C]
        
        x = x.view(B, *x_size, C).contiguous()  # [B,H,W,C]
        x = self.ln_1(x)
        #深度卷积,得到 conv_x
        
        conv_x = self.dwconv(x.permute(0, 3, 1, 2).contiguous()) #B C H W
        
        
        mamba_x = self.mamba_b(x.view(B , -1 , C), x_size) #B L C
        
       
        C_Imap = self.channel_interaction(conv_x).permute(0, 2, 3, 1).contiguous().view(B, 1, C) # B 1 C
        S_Imap = self.spatial_interaction(mamba_x.transpose(-2,-1).contiguous().view(B, C, *x_size)) #B 1 H W
        
        mamba_x = mamba_x * torch.sigmoid(C_Imap) # B L C
        conv_x = torch.sigmoid(S_Imap) * conv_x  
        conv_x = conv_x.view(B, L, C) # B L C
        
      
        x = mamba_x + conv_x
        x = self.proj_drop(self.proj(x)) # B L C
        x = input * self.skip_scale1 + x.view(B ,*x_size , C)
        
        
        input = x.view(B, *x_size, C).contiguous() # B H W C
        x = self.ln_2(x) # B H W C 
        x = self.convffn(x.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous() # B H W C
        x = input * self.skip_scale2 + x # B H W C
        x = x.view(B, -1, C).contiguous()  # [B,L,C]
        return x
        


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]] 
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1] 

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)#（b, l, c）
        
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size): 
        x = x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1])
        return x


    
   

 

class HMCBS(nn.Module):
    def __init__(self,
                 dim: int = 0, 
                 drop_path: float = 0, 
                 norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
                 attn_drop_rate: float = 0, 
                 d_state: int = 16, 
                 mlp_ratio: float = 2., 
                 reduction: int = 16, 
                 drop: float = 0, 
                 depth : int = 2, 
                 **kwargs):
        super().__init__()
        self.dim = dim
        self.drop_path = drop_path
        self.norm_layer = norm_layer
        self.attn_drop_rate = attn_drop_rate
        self.d_state = d_state
        self.mlp_ratio = mlp_ratio
        self.reduction = reduction
        self.drop = drop    
        self.depth = depth 
        self.norm = norm_layer(self.dim)
        self.patch_embed = PatchEmbed()
        self.patch_unembed = PatchUnEmbed(embed_dim=self.dim)
        self.HMCB = HMCB(dim=dim, drop_path=drop_path, norm_layer=norm_layer)
        
    def forward(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        for _ in range(self.depth): 
            x = self.HMCB(x , x_size) 
        x = self.norm(x)
        x = self.patch_unembed(x, x_size)
        return x   
     
     
## ------------------------------------------------SymletVERSION -------------------------------------------------##

def get_symlet_filters(wavelet='sym4'):
    
    wavelet_obj = pywt.Wavelet(wavelet)
    
    
    dec_lo = np.array(wavelet_obj.dec_lo)  
    dec_hi = np.array(wavelet_obj.dec_hi)  
    
    
    rec_lo = np.array(wavelet_obj.rec_lo)  
    rec_hi = np.array(wavelet_obj.rec_hi)  
    
    return dec_lo, dec_hi, rec_lo, rec_hi

def get_wav_symlet(in_channels, pool=True, wavelet='sym4'):
    
    
    dec_lo, dec_hi, rec_lo, rec_hi = get_symlet_filters(wavelet)
    
    if pool:
        
        filter_L = dec_lo
        filter_H = dec_hi
    else:
        
        filter_L = rec_lo
        filter_H = rec_hi
    
    
    filter_L = filter_L / np.sqrt(2)
    filter_H = filter_H / np.sqrt(2)
    
    
    filter_L_2d = filter_L.reshape(1, -1)
    filter_H_2d = filter_H.reshape(1, -1)
    
    
    filter_LL = np.transpose(filter_L_2d) * filter_L_2d
    filter_LH = np.transpose(filter_L_2d) * filter_H_2d
    filter_HL = np.transpose(filter_H_2d) * filter_L_2d
    filter_HH = np.transpose(filter_H_2d) * filter_H_2d
    
    
    filter_LL = torch.from_numpy(filter_LL).unsqueeze(0)
    filter_LH = torch.from_numpy(filter_LH).unsqueeze(0)
    filter_HL = torch.from_numpy(filter_HL).unsqueeze(0)
    filter_HH = torch.from_numpy(filter_HH).unsqueeze(0)
    
    
    kernel_size = filter_LL.shape[-1]
    
    if pool:
        net = nn.Conv2d
    else:
        net = nn.ConvTranspose2d
        
    LL = net(in_channels, in_channels*2,
             kernel_size=kernel_size, stride=2, padding=kernel_size//2, bias=False,
             groups=in_channels)
    LH = net(in_channels, in_channels*2,
             kernel_size=kernel_size, stride=2, padding=kernel_size//2, bias=False,
             groups=in_channels)
    HL = net(in_channels, in_channels*2,
             kernel_size=kernel_size, stride=2, padding=kernel_size//2, bias=False,
             groups=in_channels)
    HH = net(in_channels, in_channels*2,
             kernel_size=kernel_size, stride=2, padding=kernel_size//2, bias=False,
             groups=in_channels)

    
    LL.weight.requires_grad = False
    LH.weight.requires_grad = False
    HL.weight.requires_grad = False
    HH.weight.requires_grad = False

    
    LL.weight.data = filter_LL.float().unsqueeze(0).expand(in_channels*2, -1, -1, -1)
    LH.weight.data = filter_LH.float().unsqueeze(0).expand(in_channels*2, -1, -1, -1)
    HL.weight.data = filter_HL.float().unsqueeze(0).expand(in_channels*2, -1, -1, -1)
    HH.weight.data = filter_HH.float().unsqueeze(0).expand(in_channels*2, -1, -1, -1)

    return LL, LH, HL, HH


class WavePoolSymlet(nn.Module):
    def __init__(self, in_channels, wavelet='sym4'):
        super(WavePoolSymlet, self).__init__()
        self.LL, self.LH, self.HL, self.HH = get_wav_symlet(in_channels, wavelet=wavelet)

    def forward(self, x):
        return self.LL(x), self.LH(x), self.HL(x), self.HH(x)


def get_wav_two_symlet(in_channels, pool=True, wavelet='sym4'):
    
    
    dec_lo, dec_hi, rec_lo, rec_hi = get_symlet_filters(wavelet)
    
    if pool:
        
        filter_L = dec_lo
        filter_H = dec_hi
    else:
        
        filter_L = rec_lo
        filter_H = rec_hi
    
    
    filter_L = filter_L / np.sqrt(2)
    filter_H = filter_H / np.sqrt(2)
    
    
    filter_L_2d = filter_L.reshape(1, -1)
    filter_H_2d = filter_H.reshape(1, -1)
    
    
    filter_LL = np.transpose(filter_L_2d) * filter_L_2d
    filter_LH = np.transpose(filter_L_2d) * filter_H_2d
    filter_HL = np.transpose(filter_H_2d) * filter_L_2d
    filter_HH = np.transpose(filter_H_2d) * filter_H_2d
    
    
    filter_LL = torch.from_numpy(filter_LL).unsqueeze(0)
    filter_LH = torch.from_numpy(filter_LH).unsqueeze(0)
    filter_HL = torch.from_numpy(filter_HL).unsqueeze(0)
    filter_HH = torch.from_numpy(filter_HH).unsqueeze(0)
    
    
    kernel_size = filter_LL.shape[-1]

    if pool:
        net = nn.Conv2d
    else:
        net = nn.ConvTranspose2d
        
    LL = net(in_channels, in_channels,
             kernel_size=kernel_size, stride=2, padding=kernel_size//2, bias=False,
             groups=in_channels)
    LH = net(in_channels, in_channels,
             kernel_size=kernel_size, stride=2, padding=kernel_size//2, bias=False,
             groups=in_channels)
    HL = net(in_channels, in_channels,
             kernel_size=kernel_size, stride=2, padding=kernel_size//2, bias=False,
             groups=in_channels)
    HH = net(in_channels, in_channels,
             kernel_size=kernel_size, stride=2, padding=kernel_size//2, bias=False,
             groups=in_channels)

    
    LL.weight.requires_grad = False
    LH.weight.requires_grad = False
    HL.weight.requires_grad = False
    HH.weight.requires_grad = False

    
    LL.weight.data = filter_LL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    LH.weight.data = filter_LH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    HL.weight.data = filter_HL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    HH.weight.data = filter_HH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)

    return LL, LH, HL, HH


class WaveUnpoolSymlet(nn.Module):
    def __init__(self, in_channels, wavelet='sym4'):
        super(WaveUnpoolSymlet, self).__init__()
        self.in_channels = in_channels
        self.LL, self.LH, self.HL, self.HH = get_wav_two_symlet(self.in_channels, pool=False, wavelet=wavelet)

    def forward(self, LL, LH, HL, HH):
        return self.LL(LL) + self.LH(LH) + self.HL(HL) + self.HH(HH)   

###---------------------------------------------------HaarVERSION-------------------------------------------------###
def get_wav(in_channels, pool=True): 
    """ wavelet decomposition using conv2d """
    harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2)) 
    harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2)) 
    harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]

    harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L
    harr_wav_LH = np.transpose(harr_wav_L) * harr_wav_H
    harr_wav_HL = np.transpose(harr_wav_H) * harr_wav_L
    harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H

    filter_LL = torch.from_numpy(harr_wav_LL).unsqueeze(0)
    filter_LH = torch.from_numpy(harr_wav_LH).unsqueeze(0)
    filter_HL = torch.from_numpy(harr_wav_HL).unsqueeze(0)
    filter_HH = torch.from_numpy(harr_wav_HH).unsqueeze(0)

    
    if pool:
        net = nn.Conv2d
    else:
        net = nn.ConvTranspose2d
    LL = net(in_channels, in_channels*2,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    LH = net(in_channels, in_channels*2,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HL = net(in_channels, in_channels*2,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HH = net(in_channels, in_channels*2,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)

    LL.weight.requires_grad = False 
    LH.weight.requires_grad = False
    HL.weight.requires_grad = False
    HH.weight.requires_grad = False

    LL.weight.data = filter_LL.float().unsqueeze(0).expand(in_channels*2, -1, -1, -1) 
    LH.weight.data = filter_LH.float().unsqueeze(0).expand(in_channels*2, -1, -1, -1)
    HL.weight.data = filter_HL.float().unsqueeze(0).expand(in_channels*2, -1, -1, -1)
    HH.weight.data = filter_HH.float().unsqueeze(0).expand(in_channels*2, -1, -1, -1)

    return LL, LH, HL, HH


class WavePool(nn.Module):
    def __init__(self, in_channels):
        super(WavePool, self).__init__()
        self.LL, self.LH, self.HL, self.HH = get_wav(in_channels)

    def forward(self, x):
        return self.LL(x), self.LH(x), self.HL(x), self.HH(x)





def get_wav_two(in_channels, pool=True): 
    """ wavelet decomposition using conv2d """
    harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]

    harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L
    harr_wav_LH = np.transpose(harr_wav_L) * harr_wav_H
    harr_wav_HL = np.transpose(harr_wav_H) * harr_wav_L
    harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H

    filter_LL = torch.from_numpy(harr_wav_LL).unsqueeze(0)
    filter_LH = torch.from_numpy(harr_wav_LH).unsqueeze(0)
    filter_HL = torch.from_numpy(harr_wav_HL).unsqueeze(0)
    filter_HH = torch.from_numpy(harr_wav_HH).unsqueeze(0)

    if pool:
        net = nn.Conv2d
    else:
        net = nn.ConvTranspose2d
    LL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    LH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)

    LL.weight.requires_grad = False
    LH.weight.requires_grad = False
    HL.weight.requires_grad = False
    HH.weight.requires_grad = False

    LL.weight.data = filter_LL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    LH.weight.data = filter_LH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    HL.weight.data = filter_HL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    HH.weight.data = filter_HH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)

    return LL, LH, HL, HH


class WaveUnpool(nn.Module):
    def __init__(self, in_channels):
        super(WaveUnpool, self).__init__()
        self.in_channels = in_channels
        self.LL, self.LH, self.HL, self.HH = get_wav_two(self.in_channels, pool=False)

    def forward(self, LL, LH, HL, HH):
        return self.LL(LL) + self.LH(LH) + self.HL(HL) + self.HH(HH)



class FDBNet(nn.Module):
    """ main architecture """
    def __init__(self, img_range=1,FDB_embed_dim=32, mlp_ratio=4, drop_rate=0.,
                 token_projection='linear'):
        super().__init__()
        
        self.mlp_ratio = mlp_ratio
        self.dropout = drop_rate
        self.token_projection = token_projection
        
        rgb_mean = (0.4488, 0.4371, 0.4040)
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        self.img_range = img_range
        # Input/Output
        self.shallow_conv = nn.Conv2d(3, FDB_embed_dim, 3, 1, 1)
        self.shallow_act = nn.GELU()
        self.out = nn.Conv2d(64,3,3,1,1)
        self.out_act = nn.GELU()

        
        self.encoderlayer_0 = HMCBS(dim=32,depth=2)
        self.encoderlayer_1 = HMCBS(dim=64,depth=2)
        self.encoderlayer_2 = HMCBS(dim=128,depth=2)
        self.encoderlayer_3 = HMCBS(dim=256,depth=2)
        self.encoderlayer_bottom = HMCBS(dim=256,depth=2)
        self.decoderlayer_0 = HMCBS(dim=512,depth=2)
        self.decoderlayer_1 = HMCBS(dim=256,depth=2)
        self.decoderlayer_2 = HMCBS(dim=128,depth=2)
        self.decoderlayer_3 = HMCBS(dim=64,depth=2)

        
        self.pool0 = nn.Conv2d(32,64,4,2,1)
        self.pool1 = nn.Conv2d(64,128,4,2,1)
        self.pool2 = nn.Conv2d(128,256,4,2,1)
        
        self.up0 =  nn.Conv2d(512, 256, 1, 1)
        self.up1 =  nn.Conv2d(256, 128, 1, 1)
        self.up2 =  nn.Conv2d(128, 64, 1, 1)

        self.sigmoid = nn.Sigmoid()
        
        self.Upsample0 = nn.ConvTranspose2d(256, 256, 2, 2)
        self.Upsample1 = nn.ConvTranspose2d(128, 128, 2, 2)
        self.Upsample2 = nn.ConvTranspose2d(64, 64, 2, 2)
       
        self.high2_LL = nn.Conv2d(256, 256, 3, 1, 1)
        self.high2_LH = nn.Conv2d(256, 256, 3, 1, 1)
        self.high2_HL = nn.Conv2d(256, 256, 3, 1, 1)

        self.high1_LL = nn.Conv2d(128, 128, 3, 1, 1)
        self.high1_LH = nn.Conv2d(128, 128, 3, 1, 1)
        self.high1_HL = nn.Conv2d(128, 128, 3, 1, 1)

        self.high0_LL = nn.Conv2d(64, 64, 3, 1, 1)
        self.high0_LH = nn.Conv2d(64, 64, 3, 1, 1)
        self.high0_HL = nn.Conv2d(64, 64, 3, 1, 1)

        ## wave
        self.wave_pool0 = WavePool(32)
        self.wave_pool1 = WavePool(64)
        self.wave_pool2 = WavePool(128)
        
        self.recon_block0 = WaveUnpool(256)
        self.recon_block1 = WaveUnpool(128)
        self.recon_block2 = WaveUnpool(64)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B, _, _, _ = x.shape
        self.mean = self.mean.type_as(x) 
        x = (x - self.mean) * self.img_range
        y = self.shallow_conv(x)
        y = self.shallow_act(y)

        trans0 = self.encoderlayer_0(y)
        
        LL0, LH0, HL0, _ = self.wave_pool0(trans0)
        pool0 = self.pool0(trans0)

        trans1 = self.encoderlayer_1(pool0)
        

        LL1, LH1, HL1, _ = self.wave_pool1(trans1)
        pool1 = self.pool1(trans1)

        trans2 = self.encoderlayer_2(pool1)
        
        LL2, LH2, HL2, _ = self.wave_pool2(trans2)
        pool2 = self.pool2(trans2)

        trans3 = self.encoderlayer_3(pool2)
        

        trans_bottom = self.encoderlayer_bottom(trans3)

        decoder_input0 = torch.cat([trans_bottom, trans3], 1)
        
        
        trans_decoder_0 = self.decoderlayer_0(decoder_input0)
        trans_decoder_0_l, trans_decoder_0_r =  trans_decoder_0.chunk(2, dim=1)
        trans_decoder_0_l = self.recon_block0(self.high2_LL(LL2), self.high2_LH(LH2), self.high2_HL(HL2),
                                              trans_decoder_0_l)
        
        trans_decoder_0_r = self.Upsample0(trans_decoder_0_r)
        trans_decoder_0 = torch.cat([trans_decoder_0_l, trans_decoder_0_r], 1)
        up0 = self.up0(trans_decoder_0)

        trans_decoder_1 = self.decoderlayer_1(up0)
        trans_decoder_1_l, trans_decoder_1_r =  trans_decoder_1.chunk(2, dim=1)
        trans_decoder_1_l = self.recon_block1(self.high1_LL(LL1), self.high1_LH(LH1), self.high1_HL(HL1),
                                              trans_decoder_1_l)
        trans_decoder_1_r = self.Upsample1(trans_decoder_1_r)
        trans_decoder_1 = torch.cat([trans_decoder_1_l, trans_decoder_1_r], 1)
        up1 = self.up1(trans_decoder_1)


        trans_decoder_2 = self.decoderlayer_2(up1)
        trans_decoder_2_l, trans_decoder_2_r =  trans_decoder_2.chunk(2, dim=1)
        
        trans_decoder_2_l = self.recon_block2(self.high0_LL(LL0), self.high0_LH(LH0), self.high0_HL(HL0),
                                              trans_decoder_2_l)
        trans_decoder_2_r = self.Upsample2(trans_decoder_2_r)

        trans_decoder_2 = torch.cat([trans_decoder_2_l, trans_decoder_2_r], 1)
        up2 = self.up2(trans_decoder_2)

        trans_decoder_3 = self.decoderlayer_3(up2)

        final_output = self.out(trans_decoder_3)
        final_output = self.out_act(final_output)

        return (final_output + x) / self.img_range + self.mean





class two_stageHMC(nn.Module):
    def __init__(self, img_size=128 , img_range=1,embed_dim=32, mlp_ratio=4, drop_rate=0.,
                 token_projection='linear'):                 
        super(two_stageHMC, self).__init__()
        self.img_size = img_size
       
        self.FDBNet = FDBNet(FDB_embed_dim=embed_dim, mlp_ratio=mlp_ratio, drop_rate=drop_rate,
                             token_projection=token_projection)
        
    #网络实现
    def forward(self, x):
        x = self.FDBNet(x)
        return x  
    
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn((1, 3, 128,128)).to(device)
    net = two_stageHMC().to(device)
    y= net(x)
    print(y.shape)