import numpy as np

import os
import h5py

from scipy import signal
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, random_split

from datetime import timedelta
from obspy import UTCDateTime, Catalog, read
from obspy.clients.fdsn import Client
from obspy.taup import TauPyModel


def get_test_samples():
    
    test_path = '../../data/earthquakes/sissle/eq_data_50Hz/test/'
    test_paths = sorted([test_path + f for f in os.listdir(test_path)])

    indices = [6250,6750,6250,5250,7500,5500]
    test_data = []
    for i, (p, idx) in enumerate(zip(test_paths, indices)):
        with h5py.File(p, 'r') as hf:
            test_data.append(hf['DAS'][81:,idx-3000:idx+3000])
    test_data = np.stack(test_data)[[2,3,5,0,1,4]]

    gutter = 1000
    test_data = np.pad(test_data, ((0,0),(0,0),(gutter,gutter)), mode='constant', constant_values=0)
    test_data = bandpass(test_data, low=1.0, high=10.0, fs=50, gutter=gutter)
    test_scale = test_data.std(axis=-1, keepdims=True)


    test_data = torch.from_numpy(test_data.copy())
    test_scale = torch.from_numpy(test_scale.copy())
    
    return test_data, test_scale

def bandpass(x, low, high, fs, gutter, alpha=0.1):
    """
    alpha: taper length
    """
    
    passband = [2 * low/fs, 2 * high/fs]
    b, a = signal.butter(2, passband, btype="bandpass")
    window = signal.windows.tukey(x.shape[-1], alpha=alpha)
    x = signal.filtfilt(b, a, x * window, axis=-1)

    return x[..., gutter:-gutter]

def generate_synthetic_das(strain_rate, dx, fs, slowness, nx=512):

    shift = dx * fs * slowness

    sample = torch.zeros((nx, len(strain_rate)))
    for i in range(nx):
        sample[i] = torch.roll(strain_rate, int(i*shift + np.random.randn(1)))
    
    return sample

def shift_traffic_rates(traffic_rates, dx, fs, slowness):

    shift = dx * fs * slowness
    
    # traffic_rates shape (512,3000)
    traffic_rates = torch.tile(traffic_rates, (1,3))
    sample = torch.zeros_like(traffic_rates)
    for i in range(len(traffic_rates)):
        sample[i] = torch.roll(traffic_rates[i], int(i*shift + np.random.randn(1)))
    sample = torch.roll(sample, np.random.randint(0,traffic_rates.shape[1]//3))

    return sample

class SyntheticTrafficDAS(Dataset):
    def __init__(self, eq_strain_rates, traffic_inc, traffic_dec, 
                 nx=64, nt=256, eq_slowness=(1e-4, 5e-3), log_SNR=(-2,4), traffic_slowness=(3e-2, 6e-2),
                 dx=4, fs=50.0, size=1000):
        self.eq_strain_rates = eq_strain_rates / eq_strain_rates.std(dim=-1, keepdim=True)
        self.traffic_inc = traffic_inc / traffic_inc.std(dim=-1, keepdim=True)
        self.traffic_dec = traffic_dec / traffic_dec.std(dim=-1, keepdim=True)
        self.nx = nx
        self.nt = nt
        self.eq_slowness = eq_slowness
        self.log_SNR = log_SNR
        self.traffic_slowness = traffic_slowness
        self.dx = dx
        self.fs = fs
        self.size = size
        
        if self.dx == 20:
            self.traffic_inc = self.traffic_inc[:,::5]
            self.traffic_dec = self.traffic_dec[:,::5]
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        
        sample = torch.zeros((self.nx, self.nt))
        eq_strain_rate = self.eq_strain_rates[np.random.randint(0,len(self.eq_strain_rates))].clone()
        if np.random.random() < 0.5:
            eq_strain_rate = torch.flip(eq_strain_rate, dims=(0,))
        if np.random.random() < 0.5:
            eq_strain_rate *= -1
            
        slowness = np.random.uniform(*self.eq_slowness)
        if np.random.random() < 0.5:
            slowness *= -1
        eq_das = generate_synthetic_das(eq_strain_rate, self.dx, self.fs, slowness, nx=self.nx)
        idx = np.random.randint(0, 9000-self.nt+1)
        eq_das = eq_das[:,idx:idx+self.nt]
        
        snr = 10 ** np.random.uniform(*self.log_SNR)  # log10-uniform distribution
        
        eq_das /= eq_das.std(dim=-1, keepdim=True)
        amp = 2 * np.sqrt(snr) / torch.abs(eq_das + 1e-10).max()
        eq_das *= amp
        
        if np.random.random() < 0.5:
            idx = np.random.randint(0, len(self.traffic_inc))
            start = np.random.randint(0, self.traffic_inc.shape[1]-self.nx+1)
            traffic_rates = self.traffic_inc[idx, start:start+self.nx].clone()
            direction = 1
        else:
            idx = np.random.randint(0, len(self.traffic_dec))
            start = np.random.randint(0, self.traffic_dec.shape[1]-self.nx+1)
            traffic_rates = self.traffic_dec[idx, start:start+self.nx].clone()
            direction = -1
        
        slowness = np.random.uniform(*self.traffic_slowness)
        traffic_das = shift_traffic_rates(traffic_rates, self.dx, self.fs, direction*slowness)
        
        if np.random.random() < 0.5:
            if direction == 1:
                idx = np.random.randint(0, len(self.traffic_dec))
                start = np.random.randint(0, self.traffic_dec.shape[1]-self.nx+1)
                traffic_rates = self.traffic_dec[idx, start:start+self.nx].clone()
            else:
                idx = np.random.randint(0, len(self.traffic_inc))
                start = np.random.randint(0, self.traffic_inc.shape[1]-self.nx+1)
                traffic_rates = self.traffic_inc[idx, start:start+self.nx].clone()
                
            slowness = np.random.uniform(*self.traffic_slowness)
            traffic_das += (0.3*torch.randn(1).item() + 1) * shift_traffic_rates(traffic_rates, self.dx, self.fs, -1*direction*slowness)
        
        
        gutter = 100
        idx = np.random.randint(gutter, 9000-self.nt-gutter+1)
        traffic_das = traffic_das[:,idx-gutter:idx+self.nt+gutter]
        traffic_das = torch.from_numpy(bandpass(traffic_das, 1.0, 10.0, self.fs, gutter).copy())
        traffic_das /= traffic_das.std(dim=-1, keepdim=True)
        
        sample = eq_das + traffic_das
        scale = sample.std(dim=-1, keepdim=True)
        sample /= scale        
        return sample.unsqueeze(0), (eq_das / amp).unsqueeze(0), traffic_das.unsqueeze(0), scale.unsqueeze(0), amp
    


class SyntheticNoiseDAS(Dataset):
    def __init__(self, eq_strain_rates, 
                 nx=11, nt=2048, eq_slowness=(1e-4, 5e-3), log_SNR=(-2,4),
                 dx=4, fs=50.0, size=1000):
        self.eq_strain_rates = eq_strain_rates / eq_strain_rates.std(dim=-1, keepdim=True)
        self.nx = nx
        self.nt = nt
        self.eq_slowness = eq_slowness
        self.log_SNR = log_SNR
        self.dx = dx
        self.fs = fs
        self.size = size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        
        eq_strain_rate = self.eq_strain_rates[np.random.randint(0,len(self.eq_strain_rates))].clone()
        if np.random.random() < 0.5:
            eq_strain_rate = torch.flip(eq_strain_rate, dims=(0,))
        if np.random.random() < 0.5:
            eq_strain_rate *= -1
            
        slowness = np.random.uniform(*self.eq_slowness)
        if np.random.random() < 0.5:
            slowness *= -1
        eq_das = generate_synthetic_das(eq_strain_rate, self.dx, self.fs, slowness, nx=self.nx)
        j = np.random.randint(0, eq_strain_rate.shape[-1]-self.nt+1)
        eq_das = eq_das[:,j:j+self.nt]

        snr = 10 ** np.random.uniform(*self.log_SNR)
        amp = 2 * np.sqrt(snr) / torch.abs(eq_das + 1e-10).max()
        eq_das *= amp

        gutter = 100
        noise = np.random.randn(self.nx, self.nt + 2*gutter)
        noise = torch.from_numpy(bandpass(noise, 1.0, 10.0, self.fs, gutter).copy())

        sample = eq_das + noise
        scale = sample.std(dim=-1, keepdim=True)
        sample /= scale
                
        return sample.unsqueeze(0), eq_das.unsqueeze(0), noise.unsqueeze(0), scale.unsqueeze(0), amp


class RealDAS(Dataset):
    def __init__(self, data, nx=128, nt=512, size=1000):
        
        self.data = torch.from_numpy(data.copy())
        self.nx, self.nt = nx, nt
        self.size = size
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        
        n, nx_total, nt_total = self.data.shape
        nx = np.random.randint(0, nx_total - self.nx)
        nt = np.random.randint(0, nt_total - self.nt)
        
        patch = self.data[idx % n, nx:nx+self.nx, nt:nt+self.nt].clone()
        
        if np.random.random() < 0.5:
            patch = torch.flip(patch, dims=(0,))
        if np.random.random() < 0.5:
            patch = torch.flip(patch, dims=(1,))
        if np.random.random() < 0.5:
            patch *= -1 

        return patch.unsqueeze(0)