import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from tqdm import tqdm

from data import bandpass
from data import generate_synthetic_das
from masks import xmask, get_offsets

@torch.no_grad()
def xreconstruct(model, data, nx, nt, nx_width, nx_stride=None, nt_stride=None, batch_size=128, dx=4, fs=50, ende=False, verbose=True):
    try:
        device = next(model.parameters()).device
    except:
        device = torch.device('cpu')
    
    off1, off2, toff1, toff2 = get_offsets(nx_width, dx=dx, fs=fs)
    nt_width = toff1 + toff2 + 1 # int(np.round((nx_width-1)/m_l))
    if nx_stride is None:
        nx_stride = nx_width // 2
    if nt_stride is None:
        nt_stride = nt_width // 2
    NX, NT = data.shape
    
    lower_res = max(nx//2 - off1, 0) 
    upper_res = nx-1 - min(nx//2 + off2, nx-1)
    left_res = max(nt//2 - toff1, 0)
    right_res = nt-1 - min(nt//2 + toff2, nt-1)
    if ende:
        left_res = 0
        right_res = 0
        nt_width = nt
    
    # pad wrt. the window size and stride
    NX_pad = (NX // nx_stride) * nx_stride + nx_width - NX
    NT_pad = (NT // nt_stride) * nt_stride + nt_width - NT
    data_pad = F.pad(data, (left_res, right_res + NT_pad, lower_res, upper_res + NX_pad), mode='constant')
    
    result = torch.zeros_like(data_pad).to(device)
    freq = torch.zeros_like(data_pad).to(device)
    
    input_mask, center_mask = xmask(torch.randn(1,1,nx,nt).to(device), torch.tensor([[nx//2,nt//2]]), nx_width=nx_width, dx=dx, fs=fs)
    if ende:
        center_mask = torch.ones_like(center_mask) * (torch.sum(center_mask, dim=-1, keepdim=True) > 0)
        input_mask = 1 - center_mask
    
    def run_batch(batch):
        x = torch.stack(batch['samples']).unsqueeze(1).float().to(device)
        
        x_in = (x*input_mask).float()

        out = model(x_in) * center_mask
        for k, (p1, p2) in enumerate(batch['patch_slices']):
            result[p1, p2] += out[k,0]
            freq[p1, p2] += center_mask[0,0]

        return {'samples': [], 'patch_slices': []}
    
    batch = {'samples': [], 'patch_slices': []}
    
    num_patches_x = (NX + NX_pad - nx_width) // nx_stride + 1
    num_patches_t = (NT + NT_pad - nt_width) // nt_stride + 1
    
    for i in tqdm(range(num_patches_x), disable=verbose):
        for j in range(num_patches_t):
            
            p1 = slice(i*nx_stride, i*nx_stride+nx)
            p2 = slice(j*nt_stride, j*nt_stride+nt)
            
            batch['samples'].append(data_pad[p1, p2])
            batch['patch_slices'].append((p1, p2))
            if len(batch['samples']) == batch_size:
                batch = run_batch(batch)


    if len(batch['samples']) > 0:  
        run_batch(batch)
        
    # remove padding
    result = result[lower_res:lower_res+NX, left_res:left_res+NT]  
    freq = freq[lower_res:lower_res+NX, left_res:left_res+NT]
    return result / freq

@torch.no_grad()
def channelwise_reconstruct(model, data, nx, nt, batch_size=256):
    datas = data.split(batch_size, dim=0)
    recs = []
    for data in datas:
        recs.append(channelwise_reconstruct_part(model, data, nx, nt))
    return torch.cat(recs, dim=0)

@torch.no_grad()
def channelwise_reconstruct_part(model, data, nx, nt):

    try:
        device = next(model.parameters()).device
    except:
        device = torch.device('cpu')

    NX, NT = data.shape
    stride = 2048

    NT_pad = (NT // stride) * stride + stride - NT
    num_patches_t = (NT - 2048) // stride + 1
    rec = torch.zeros((NX, NT))
    freq = torch.zeros((NX, NT))
    
    lower_res = int(np.floor(nx/2))
    upper_res = int(np.ceil(nx/2))
    data_pad = F.pad(data, (0,0,lower_res, upper_res), mode='constant')
    
    masks = torch.ones((NX, 1, nx, nt)).to(device)
    masks[:,:,nx//2] = 0
    for i in range(num_patches_t):    
        noisy_samples = torch.zeros((NX, 1, nx, nt)).to(device)
        for j in range(NX):
            noisy_samples[j] = data_pad[j:j+nx, i*stride:i*stride + nt]
        x = (noisy_samples * masks).float().to(device)
        out = (model(x) * (1 - masks)).detach().cpu()
        rec[:, i*stride:i*stride + nt] += torch.sum(out, axis=(1,2))
        freq[:, i*stride:i*stride + nt] += torch.sum(1 - masks.detach().cpu(), axis=(1,2))

    if NT % stride != 0:
        noisy_samples = torch.zeros((NX, 1, nx, nt)).to(device)
        for j in range(NX):
            noisy_samples[j] = data_pad[j:j+nx, -nt:]
        x = (noisy_samples * masks).float().to(device)
        out = (model(x) * (1 - masks)).detach().cpu()
        rec[:, -nt:] += torch.sum(out, axis=(1,2))
        freq[:, -nt:] += torch.sum(1 - masks.detach().cpu(), axis=(1,2))
        
    rec /= freq    
    return rec