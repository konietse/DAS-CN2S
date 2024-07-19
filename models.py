import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    
    def __init__(self, in_ch, out_ch, h_ch=None, dropout_p=0):
        super().__init__()
        if h_ch is None:
            h_ch = out_ch
        self.layers = nn.Sequential(
            nn.Conv2d(in_ch, h_ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.BatchNorm2d(h_ch),
            nn.Conv2d(h_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.BatchNorm2d(out_ch),
        )
        
    def forward(self, x):
        return self.layers(x)

class MaxBlurPool(nn.Module):
    
    # https://arxiv.org/pdf/1904.11486
    
    def __init__(self, ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2, stride=1)
        a = torch.tensor([1.,1.])
        a = a[:,None] * a[None,:]
        a = a/a.sum()
        self.blur_kernel = nn.Parameter(a[None,None,:,:].repeat(ch, 1, 1, 1), requires_grad=True)
        
    def forward(self, x):
        x = self.pool(x)
        x = F.conv2d(x, weight=self.blur_kernel, stride=(2,2), padding=(1,1), groups=x.shape[1])
        return x
        

class Down(nn.Module):
    
    def __init__(self, in_ch, out_ch, h_ch=None, dropout_p=0):
        super().__init__()
        self.convs = ConvBlock(in_ch, out_ch, h_ch, dropout_p)
        self.downsample = MaxBlurPool(out_ch)
        
    def forward(self, x):
        h = self.convs(x)
        x = self.downsample(h)
        return x, h
    
class Up(nn.Module):
    
    def __init__(self, in_ch, out_ch, dropout_p=0):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)    
        self.convs = ConvBlock(in_ch, out_ch, dropout_p=dropout_p)
        
    def forward(self, x, h):
        x = self.upsample(x)
        x = torch.cat([x, h], dim=1)
        x = self.convs(x)
        return x
        
class Mid(nn.Module):
    
    def __init__(self, in_ch, out_ch, dropout_p=0):
        super().__init__()
        self.convs = ConvBlock(in_ch, out_ch, dropout_p=dropout_p)
        
    def forward(self, x):
        x = self.convs(x)
        return x

def orthogonal_init(module):
    if isinstance(module, (nn.Conv2d,)):
        nn.init.orthogonal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    
class CN2SUNet(nn.Module):
        
    
    def __init__(self, in_ch=3, out_ch=3, hidden_ch=64, n_layers=4, dropout_p=0):
        super().__init__()
        
        down_dims = [in_ch] + [hidden_ch*2**i for i in range(n_layers)]
        up_dims = [hidden_ch*2**i for i in range(n_layers,-1,-1)]
        self.downs = nn.ModuleList([Down(down_dims[i], down_dims[i+1], dropout_p=dropout_p) for i in range(len(down_dims) - 1)])
        
        self.mid = Mid(down_dims[-1], down_dims[-1]*2, dropout_p=dropout_p)
        self.ups = nn.ModuleList([Up(up_dims[i], up_dims[i+1], dropout_p=dropout_p) for i in range(len(up_dims) - 1)])
        
        self.outconv = nn.Conv2d(hidden_ch, out_ch, 1)
        
        self.apply(orthogonal_init)
        
    def forward(self, x):
        
        hist = []
        for down in self.downs:
            x, h = down(x)
            hist.append(h)
        
        x = self.mid(x)
        
        for up in self.ups:
            h = hist.pop()
            x = up(x, h)
        
        x = self.outconv(x)
        return x



# This function is adapted on the implementation from https://doi.org/10.6084/m9.figshare.14152277.v1 by Martijn van den Ende, Itzhak Lior, Jean-Paul Ampuero, Anthony Sladen, André Ferrari, Cédric Richard
# Used under the CC BY 4.0 License
# https://creativecommons.org/licenses/by/4.0/

class ConvBlock2(nn.Module):
    
    def __init__(self, in_channels, out_channels, hidden_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=(3, 5), padding='same'),#(1, 2) #(3, 5)
            nn.SiLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=(3, 5), padding='same'),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.layers(x)

class Down2(nn.Module):
        
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.pool = nn.MaxPool2d((1, 4), stride=(1, 1))
            w = torch.tensor([1., 3., 3., 1.]) / 8.
            self.w = nn.Parameter(w[None, None, None, :].repeat((in_channels, 1, 1, 1)), requires_grad=False)
            self.conv = ConvBlock2(in_channels, out_channels, out_channels)
    
        def forward(self, x):
            x = self.pool(x)
            x = F.conv2d(x, weight=self.w, stride=(1, 4), padding=(0, 2), groups=x.shape[1])
            x = self.conv(x)
            return x


class Up2(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=(1, 4), mode='bilinear')
        self.conv = ConvBlock2(in_channels+out_channels, out_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)

class N2SUNet(nn.Module):

    def __init__(self, in_channels, out_channels, hidden_dim=4):
        super().__init__()

        self.inconv = ConvBlock2(in_channels, hidden_dim, hidden_dim)

        self.down1 = Down2(hidden_dim, 2*hidden_dim)
        self.down2 = Down2(2*hidden_dim, 4*hidden_dim)
        self.down3 = Down2(4*hidden_dim, 8*hidden_dim)
        self.down4 = Down2(8*hidden_dim, 16*hidden_dim)

        self.up1 = Up2(16*hidden_dim, 8*hidden_dim)
        self.up2 = Up2(8*hidden_dim, 4*hidden_dim)
        self.up3 = Up2(4*hidden_dim, 2*hidden_dim)
        self.up4 = Up2(2*hidden_dim, hidden_dim)

        self.outconv = nn.Conv2d(hidden_dim, out_channels, kernel_size=(3, 5), padding='same') #(1, 2) #(3, 5)

        self.apply(orthogonal_init)

    def forward(self, x):    

        x0 = self.inconv(x)

        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)

        x = self.outconv(x)

        return x