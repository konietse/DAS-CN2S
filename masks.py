import torch
import numpy as np

def get_slopes(dx, fs):
    v_max = 120 * 10 / 36
    v_min = 60 * 10 / 36
    m_u = v_max / fs / dx
    m_l = v_min / fs / dx
    return m_u, m_l

def get_offsets(nx_width, dx=4, fs=50):
    m_u, m_l = get_slopes(dx, fs)
    
    off1 = int(np.floor((nx_width - 1)/2))
    off2 = int(np.ceil((nx_width - 1)/2))
    
    toff1 = int(np.floor((off1 + off2)/(2*m_l)))
    toff2 = int(np.ceil((off1 + off2)/(2*m_l)))
    
    return off1, off2, toff1, toff2

def xmask(x, center=None, nx_width=1, dx=4, fs=50, hourglass=False):
    
    batch_size, _, nx, nt = x.shape
    NX, NT = torch.meshgrid(torch.arange(nx), torch.arange(nt), indexing='ij')
    NX, NT = NX[None, None, :], NT[None, None, :]

    if center is None:
        center = torch.cat([torch.randint(0, nx, (batch_size,1)),
                            torch.randint(0, nt, (batch_size,1))], dim=1)
    c1 = center[:, 1, None, None, None]
    c2 = center[:, 0, None, None, None]
    
    m_u, m_l = get_slopes(dx, fs)
    if hourglass:
        m_l = 0
    
    off1, off2, toff1, toff2 = get_offsets(nx_width, dx, fs)
    ft = lambda m: m * (NT - c1) + c2
    
    upper_pos = torch.logical_or(NX > ft(m_u) + off1, NX <= ft(m_l) - off2)
    lower_pos = torch.logical_or(NX > ft(m_l) + off1, NX <= ft(m_u) - off2)
    pos_mask = torch.logical_and(upper_pos, lower_pos)

    upper_neg = torch.logical_or(NX > ft(-m_u) + off1, NX <= ft(-m_l) - off2)
    lower_neg = torch.logical_or(NX > ft(-m_l) + off1, NX <= ft(-m_u) - off2)
    neg_mask = torch.logical_and(upper_neg, lower_neg)

    center_mask = torch.logical_and(
        torch.logical_and(NX < c2 + off2 + 1, NX >= c2 - off1),
        torch.logical_and(NT < c1 + toff2 + 1, NT >= c1 - toff1)
    ).float()
    
    mask = torch.logical_and(pos_mask, neg_mask).float()
    mask = 1 - torch.logical_or(1 - mask, center_mask).float()
    
    return mask.to(x.device), center_mask.to(x.device)
    

def channelwise_mask(x, width=1, indices=None):

    batch_size, _, nx, nt = x.shape
    mask = torch.ones_like(x)
    u = int(np.floor(width/2))
    l = int(np.ceil(width/2))
    if indices is None:
        indices = torch.randint(u, nx - l, (batch_size,))
    for i in range(batch_size):
        mask[i, :, indices[i]-u:indices[i]+l] = 0
    
    return mask