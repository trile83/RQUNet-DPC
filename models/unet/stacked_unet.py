import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np
import math as m
import cmath as cm   

pi = torch.tensor(m.pi)
eps = torch.finfo(float).eps
j = torch.tensor(cm.sqrt(-1))

# Func - Isotropic polyharmonic Bspline torch:
def IPBspline(omega1, omega2, gamma):   
    z1 = torch.exp(j*omega1)
    z2 = torch.exp(j*omega2)
    
    z1_1 = z1**(-1)     # z1_1 = 1/(z1.astype(np.complex128))
    z2_1 = z2**(-1)     # z2_1 = 1/(z2.astype(np.complex128))
    
    # Isotropic Bspline:
    Lp_z = 4 - z1 - z1_1 - z2 - z2_1    
    Lm_z = 0.5 * ( 4 - z1*z2 - z1_1*z2 - z1*z2_1 - z1_1*z2_1 )
    
    V2_z = 2/3*Lp_z + 1/3*Lm_z
    beta_omega = (eps + V2_z)**(gamma/2) / (eps + omega1**2 + omega2**2)**(gamma/2)
    
    return beta_omega


# Func - Scaling autocorrelation func:
def AutocorrelationFunc(omega1, omega2, gamma):    
    Height, Width = omega1.shape
    A = torch.zeros((Height, Width))    
    for m1 in range(-5, 6, 1):  # -5, 6, 1
        for m2 in range(-5, 6, 1):
            A = A + IPBspline(2*pi*m1 + omega1, 2*pi*m2 + omega2, 2*gamma)
        
    return A


# Func - Scaling autocorrelation func at scale D:
def AutocorrelationFunc_scaleD(omega1, omega2, gamma):   
    A_D = 0.5 * ( torch.abs(Lowpass(omega1, omega2, gamma))**2 * AutocorrelationFunc(omega1, omega2, gamma) + torch.abs(Lowpass(omega1+pi, omega2+pi, gamma))**2 * AutocorrelationFunc(omega1+pi, omega2+pi, gamma));
    return A_D

# Func - Primal lowpass filter:
def ScalingFunc_dual(omega1, omega2, gamma):   
    beta_D =IPBspline(omega1, omega2, gamma) / AutocorrelationFunc(omega1, omega2, gamma)
    return beta_D

# Func - Primal lowpass filter:
def Lowpass(omega1, omega2, gamma):   
    H = torch.tensor(m.sqrt(2)) * IPBspline(omega1 + omega2, omega1 - omega2, gamma) / IPBspline(omega1, omega2, gamma)
    return H

# Func - Primal Highpass filter:
def Highpass_primal(omega1, omega2, gamma):   
    G = - torch.exp(-j*omega1) * Lowpass( -(omega1+pi), -(omega2+pi), gamma) * AutocorrelationFunc(omega1+pi, omega2+pi, gamma)
    return G

# Func - Primal lowpass filter:
def Highpass_dual(omega1, omega2, gamma):   
    G_D = - torch.exp(-j*omega1) * Lowpass( -(omega1+pi), -(omega2+pi), gamma) / AutocorrelationFunc_scaleD(omega1, omega2, gamma);
    return G_D

def BsplineQuincunxScalingWaveletFuncs(Height, Width, Scales, gamma):
    # Scales = 3   # 0, 1, 2, 3  definition: 
    # Height = 256 # 128
    # Width = 256  # 128
    # gamma = 5 # 1.2
    #   
    psi_i = torch.zeros((Scales+1, Height, Width), dtype=torch.complex64)    
    psi_D_i = torch.zeros((Scales+1, Height, Width), dtype=torch.complex64)    
    
    #
    #for i in range(0, Scales+1):
    for i in range(0, Scales):
        # print(i)
        # i = 1
    
        # Fourier coordinate:
        omega2, omega1 = torch.meshgrid(torch.linspace(-pi, pi, Width), torch.linspace(-pi, pi, Height))        
    
        if np.mod(i,2) == 0:
            omega1 = 2**(i/2)*omega1;
            omega2 = 2**(i/2)*omega2; 
        else:
            omega1_temp = 2**((i-1)/2)*(omega1 + omega2)
            omega2_temp = 2**((i-1)/2)*(omega1 - omega2)
            
            omega1 = omega1_temp
            omega2 = omega2_temp 
            
        # Primal/Dual wavelet funcs:
        psi_i[i,:,:] = torch.tensor(1/m.sqrt(2)) * Highpass_primal( 0.5*(omega1+omega2), 0.5*(omega1-omega2), gamma) * IPBspline( 0.5*(omega1+omega2), 0.5*(omega1-omega2), gamma)
        psi_D_i[i,:,:] = torch.tensor(1/m.sqrt(2)) * Highpass_dual( 0.5*(omega1+omega2), 0.5*(omega1-omega2), gamma) * ScalingFunc_dual( 0.5*(omega1+omega2), 0.5*(omega1-omega2), gamma)
    
    # Fourier coordinate:
    omega2, omega1 = torch.meshgrid(torch.linspace(-pi, pi, Width), torch.linspace(-pi, pi, Height))        
    
    i = Scales
    
    if np.mod(i,2) == 0:
        omega1 = 2**(i/2)*omega1;
        omega2 = 2**(i/2)*omega2; 
    else:
        omega1_temp = 2**((i-1)/2)*(omega1 + omega2);
        omega2_temp = 2**((i-1)/2)*(omega1 - omega2);
            
    omega1 = omega1_temp;
    omega2 = omega2_temp;    
        
    # Primal scaling funcs:
    beta_I = IPBspline(omega1, omega2, gamma)    
    beta_D_I = ScalingFunc_dual(omega1, omega2, gamma);
    
    # Correction of the last wavelet subband:
    Matrix_one = torch.ones((Height, Width), dtype=torch.complex64)    
    
    scalingFunc = torch.conj(beta_D_I) * beta_I
    # check Identity:
    waveletFunc = torch.zeros((Height, Width), dtype=torch.complex64)
    for i in range(1, Scales+1):
        waveletFunc = waveletFunc + torch.conj(psi_D_i[i,:,:])*psi_i[i,:,:]
        
    psi_i[0,:,:] = ( Matrix_one - scalingFunc - waveletFunc ) / ( torch.conj(psi_D_i[0,:,:])  ) 

    return beta_I, beta_D_I, psi_i, psi_D_i

def ActivationFuncs(method, x, t):

    if method=="HardShrink":
        y = x*(np.abs(x) > t)
    elif method=="SoftShrink":
        #y = x/(np.abs(x)+0.000000000000001)*np.maximum(np.abs(x)-t, 0)
        y = x/(torch.abs(x)+0.000000000000001)*torch.maximum(torch.abs(x)-t, torch.tensor(0))
    elif method=="Identity":
        y = x      
    elif method=="BinaryStep":
        y = 0*(x<0) + 1*(x>=0)
    elif method=="Sigmoid":
        y = 1/(1 + np.exp(-x))            
    elif method=="Tanh":
        y = ( np.exp(x) - np.exp(-x) )/( np.exp(x) + np.exp(-x) )            
    elif method=="ReLU":
        y = np.maximum(x,0)    
    elif method=="ReLU_min":
        y = np.minimum(x,0)          
    elif method=="LeakyReLU":
        alpha = 0.01
        y = alpha*x*(x<0) + x*(x>=0)                  
    elif method=="Softplus":
        y = np.log(1 + np.exp(x))            
    elif method=="SELU":
        lamda = 1  # 1.0507
        alpha = 1.67326
        y = lamda*( alpha*(np.exp(x) - 1)*(x<0) + x*(x>=0) )      
    elif method=="SiLU":
        y = x/(1 + np.exp(-x))              
    elif method=="Mish":
        y_1 = np.log(1 + np.exp(x))
        y_2 = ( np.exp(y_1) - np.exp(-y_1) )/( np.exp(y_1) + np.exp(-y_1) )
        y = x*y_2            
    elif method=="Gaussian":
        y = np.exp(-x**2)            
    elif method=="GCU":
        y = x*np.cos(x)    
    elif method=="DoublePareto":
        a = 0.1
        gamma = 0.1
        y = 0.5*x/np.abs(x)*( np.abs(x) - a + np.sqrt( (a - np.abs(x))**2 + 4*np.maximum(a*np.abs(x) - gamma, 0) ) )

    # else:
    #     print("Invalid choice of activation functions.")
    #     sys.exit()    
    
    return y

# RIESZ QUINCUNX: ========================
 
# Riesz Quincunx Wavelet Funcs:  
def RieszQuincunxWaveletFuncs(N, psi_i, psi_D_i):
    # N = 3
    # beta_I, beta_D_I, psi_i, psi_D_i = BsplineQuincunxScalingWaveletFuncs(Height, Width, Scales, gamma)
    
    Scales1, Height, Width = psi_i.shape
    Scales = Scales1 - 1
    
    #
    omega2, omega1 = torch.meshgrid(torch.linspace(-pi, pi, Width), torch.linspace(-pi, pi, Height)) 
    
    # Spectrum of Riesz opt:
    Rn_omega = torch.zeros((N+1, Height, Width), dtype=torch.complex64)
    
    for n in range(0, N+1):
        coeff = (-j)**N * m.sqrt( m.factorial(N) / m.factorial(n) / m.factorial(N - n) )
        #coeff = torch.from_numpy(coeff)
        Rn_omega[n,:,:] = coeff * omega1**n * omega2**(N-n) / ( omega1**2 + omega2**2 )**(N/2)
    
    #convert to cuda
    Rn_omega = Rn_omega.cuda()
    # N-th order Riesz Quincunx wavelet:
    psi_in = torch.zeros((Scales+1, N+1, Height, Width), dtype=torch.complex64).cuda()
    psi_D_in = torch.zeros((Scales+1, N+1, Height, Width), dtype=torch.complex64).cuda()
    
    #
    for i in range(0, Scales+1):
        for n in range(0, N+1):
            psi_D_in[i,n,:,:] = Rn_omega[n,:,:] * psi_D_i[i,:,:]
            psi_in[i,n,:,:] = Rn_omega[n,:,:] * psi_i[i,:,:]    
    
    return psi_in, psi_D_in

# 4.
def RieszQuincunxWaveletTransform_Forward(f, beta_D_I, psi_D_in):
    # alpha = 0.1   # 0.5 
    # activation_method = "SoftShrink"
    
    from torch.fft import fft2, ifft2, fftshift, ifftshift
    
    Scales1, N1, Height, Width = psi_D_in.shape
    Scales = Scales1 - 1
    N = N1 - 1

    #print(psi_D_in.shape)
    #print("beta_D_I shape: ",beta_D_I.shape)

    F = fftshift(fft2(f))

    # Scaling coefficients:
    c_I = torch.real( ifft2( ifftshift( F * torch.conj(beta_D_I) ) ) )
    #print("c_I shape: ", c_I.shape)
        
    # Wavelet coefficients:
    d_in = torch.zeros((Scales+1, N+1, Height, Width))
    #print("shape d_in: ", d_in.shape)

    for i in range(0, Scales+1):
        for n in range(0, N+1):
            d_in[i,n,:,:] = torch.real( ifft2( ifftshift( F * torch.conj(psi_D_in[i,n,:,:]) ) ) )
    
    return c_I, d_in

# 5.
def RieszQuincunxWaveletTransform_Inverse(c_I, d_in, beta_I, psi_in):
    
    from torch.fft import fft2, ifft2, fftshift, ifftshift
    
    Scales1, N1, Height, Width = psi_in.shape
    Scales = Scales1 - 1
    N = N1 - 1
    
    F_re_scaling = fftshift(fft2(c_I)) * beta_I    

    F_re_wavelet = torch.zeros((Height, Width), dtype=torch.complex64).cuda()

    for i in range(0, Scales+1):
        for n in range(0, N+1):
            F_re_wavelet = F_re_wavelet + fftshift(fft2(d_in[i,n,:,:])) * psi_in[i,n,:,:]

    # using both scaling coefficient and wavelet coefficient
    F_re = F_re_scaling + F_re_wavelet
    #
    f_re = torch.real( ifft2( ifftshift( F_re ) ) )
    
    return f_re

# 6.
def RieszWaveletTruncation(d_in, alpha, activation_method):
    # alpha = 0.1   # 0.5 
    # activation_method = "SoftShrink"

    Scales1, N1, Height, Width = d_in.shape
    Scales = Scales1 - 1
    N = N1 - 1    

    for i in range(0, Scales+1):
        for n in range(0, N+1):
            thres = alpha*torch.max(d_in[i,n,:,:])
            d_in[i,n,:,:] = ActivationFuncs(activation_method, d_in[i,n,:,:], thres)

    return d_in


# Create Unet VAE 
# 3x3 convolution module for each block
def conv3x3(in_channels, out_channels, stride=1, 
            padding=1, bias=True, groups=1):    
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)

# decoding (up) convolution module for decoder block
def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2)
    else:
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv1x1(in_channels, out_channels))

def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)

def conv_out(in_channels, out_channels, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)

# Encoder Block for UNet
class DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, segment = True, pooling=True, batchnorm=True, dropout=False):
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.segment = segment
        self.pooling = pooling
        self.dropout = dropout
        self.batchnorm = batchnorm
        
        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)
        self.batchnormalize = nn.BatchNorm2d(out_channels)

        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        if self.dropout:
            self.drop = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        if self.segment and self.batchnorm:
            x = self.batchnormalize(x) # better for segmentation
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        if self.dropout:
            x = self.drop(x)

        return x, before_pool

## Decoder Block for Unet
class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, segment=True,
                 merge_mode='concat', up_mode='bilinear'):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.segment = segment
        self.merge_mode = merge_mode
        self.up_mode = up_mode
        self.batchnorm = nn.BatchNorm2d(out_channels)

        self.upconv = upconv2x2(self.in_channels, self.out_channels,
            mode=self.up_mode)

        # skip connection from decoder to encoder
        if self.merge_mode == 'concat':
            self.conv1 = conv3x3(
                2*self.out_channels, self.out_channels)
        else:
            # num of input channels to conv2 is same
            self.conv1 = conv3x3(self.out_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)


    def forward(self, from_down, from_up):
        """ Forward pass
        Arguments:
            from_down: tensor from the encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        from_up = self.upconv(from_up)
        if self.segment:
            from_up = self.batchnorm(from_up) # better for segmentation
        if self.merge_mode == 'concat':
            x = torch.cat((from_up, from_down), 1)
        else:
            x = from_up + from_down
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


class UNet_VAE_Block(nn.Module):
    def __init__(self, num_classes, segment, alpha, in_channels=3, depth=5, 
                 start_filts=64, up_mode='upsample', 
                 merge_mode='concat', enc_out_dim=1024, latent_dim=100):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
            depth: int, number of MaxPools in the U-Net.
            start_filts: int, number of convolutional filters for the 
                first conv.
            up_mode: string, type of upconvolution. Choices: 'transpose'
                for transpose convolution or 'upsample' for nearest neighbour
                upsampling.
        """
        super(UNet_VAE_Block, self).__init__()

        if up_mode in ('transpose', 'upsample'):
            self.up_mode = up_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "upsampling. Only \"transpose\" and "
                             "\"upsample\" are allowed.".format(up_mode))
    
        if merge_mode in ('concat', 'add'):
            self.merge_mode = merge_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for"
                             "merging up and down paths. "
                             "Only \"concat\" and "
                             "\"add\" are allowed.".format(up_mode))

        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        if self.up_mode == 'upsample' and self.merge_mode == 'add':
            raise ValueError("up_mode \"upsample\" is incompatible "
                             "with merge_mode \"add\" at the moment "
                             "because it doesn't make sense to use "
                             "nearest neighbour to reduce "
                             "depth channels (by half).")

        self.num_classes = num_classes
        self.segment = segment
        self.alpha = alpha
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth

        ##### parameters for RQ
        self.scale = 5
        self.gamma = 1.2

        self.down_convs = []
        self.up_convs = []

        # create the encoder pathway and add to a list
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts*(2**i)
            pooling = True if i < depth-1 else False
            batchnorm = True if i < depth-1 else False
            if self.segment and i > (depth-3):
                dropout = False
            else:
                dropout = False

            down_conv = DownConv(ins, outs, segment=self.segment, pooling=pooling, batchnorm=batchnorm, dropout=dropout)
            self.down_convs.append(down_conv)

        #Flatten
        self.flatten = nn.Flatten()

        # create the decoder pathway and add to a list
        # - careful! decoding only requires depth-1 blocks

        for i in range(depth-1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, segment=self.segment, up_mode=up_mode,
                merge_mode=merge_mode)
            self.up_convs.append(up_conv)

        self.conv_final = conv1x1(outs, self.num_classes)

        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

        ##############################

        # RQ shrinkage calculation

        size_lst = [256,128,64,32,16]
        # Tri: Create dictionary where first key is the filter_size of tensors, second key is the quincunx scale
        # Step 0 - Riesz-Quincunx filter banks:
        beta_I_dict = {}
        beta_D_I_dict = {} 
        psi_in_dict = {}
        psi_D_in_dict = {}

        # add keys into 4 dictionary with tensor size, so that is easier to call out these values for shrinkage operation

        #for index in range(len(size_lst)):
        for index in range(self.scale):
            tensor_size = size_lst[index]
            height = size_lst[index]       
            width = size_lst[index]
            
            #for i in range(Scales_quincunx):
            #beta_I, beta_D_I, psi_i, psi_D_i = BsplineQuincunxScalingWaveletFuncs(int(height/2**i), int(width/2**i), self.scale, self.gamma)
            beta_I, beta_D_I, psi_i, psi_D_i = BsplineQuincunxScalingWaveletFuncs(height, width, self.scale, self.gamma)
            beta_I, beta_D_I, psi_i, psi_D_i = beta_I.cuda(), beta_D_I.cuda(), psi_i.cuda(), psi_D_i.cuda()

            # Riesz Quincunx wavelet:
            N = 3
            psi_in, psi_D_in = RieszQuincunxWaveletFuncs(N, psi_i, psi_D_i)
            psi_in, psi_D_in = psi_in.cuda(), psi_D_in.cuda()
        
            # save values into dictionary
            beta_I_dict[index] = beta_I
            beta_D_I_dict[index] = beta_D_I 
            psi_in_dict[index] = psi_in  
            psi_D_in_dict[index] = psi_D_in


        self.beta_I_dict = beta_I_dict
        self.beta_D_I_dict = beta_D_I_dict
        self.psi_in_dict = psi_in_dict
        self.psi_D_in_dict = psi_D_in_dict

        #############################################

        # the dimension before flatten is 1024 x 16 x 16 = 262144
        self.fc1 = nn.Linear(enc_out_dim * 16 * 16, latent_dim)
        self.fc2 = nn.Linear(enc_out_dim * 16 * 16, latent_dim)
        self.fc3 = nn.Linear(latent_dim, enc_out_dim * 16 * 16)
        self.act = nn.ReLU()

        #self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)
        
    def reparameterize(self, mu, logvar): # similar to sampling class in Keras code
        std = logvar.mul(0.5).exp_()
        std = std.cuda()
        eps = torch.normal(mu, std)
        eps = eps.cuda()
        z = mu + std * eps
        return z
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
        
    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x):
         
        # encoder pathway, save outputs for merging
        x_ori = x        
        s_i_ori = {}  
        for i, module in enumerate(self.down_convs):
            x_ori, s_ori = module(x_ori)
            s_i_ori[i] = s_ori

        # 2 - variational term (latent variable) for downsampled signals:
        x_encoded_ori = self.flatten(x_ori)

        # calculate z, mu, and logvar
        z_ori, mu_ori, logvar_ori = self.bottleneck(x_encoded_ori)
        z_ori = self.act(self.fc3(z_ori))
        z_ori = torch.reshape(z_ori, x_ori.shape)      

        # 3 - Riesz-Quincunx truncation for skip-connecting signals (alpha):
        c_I_ori = {}
        for i in range(0, len(s_i_ori)):
            c_I_ori[i] = torch.zeros(s_i_ori[i].shape)


        ## smoothing operations for original calculation
        for i in range(len(s_i_ori)):

            #print("level: ", i)
            f = s_i_ori[i]
            tensor_size = f.size(2)
            #print("tensor size: ", tensor_size)

            # Extract Riesz-Quincunx bases:
            beta_I = self.beta_I_dict[i]
            beta_D_I = self.beta_D_I_dict[i]
            psi_in = self.psi_in_dict[i]
            psi_D_in = self.psi_D_in_dict[i]

            # Forward Riesz-Quincunx wavelet:
            for k in range(f.size(0)):
                for l in range(f.size(1)):
                    c_I, d_in = RieszQuincunxWaveletTransform_Forward(f[k,l,:,:], beta_D_I, psi_D_in)
                    c_I, d_in = c_I.cuda(), d_in.cuda()
                    c_I_ori[i][k,l,:,:] = c_I   ## get c_I values and save in dictionaries          
           
       
        # ---------------------------------------------------------------------
        u = x
        # Step-II. Iteration for u:
        # Step 1 - Encoder:
        s_i = {}
        for i, module in enumerate(self.down_convs):
            u, s = module(u)
            s_i[i] = s

        # Step 2 - variational term (latent variable) for downsampled signals:
        u_encoded = self.flatten(u)

        # calculate z, mu, and logvar
        z, mu, logvar = self.bottleneck(u_encoded)
        z = self.act(self.fc3(z))
        z = torch.reshape(z, u.shape)
           
        # Step 3 - Riesz-Quincunx truncation for skip-connecting signals (alpha):
        # Duy2: intial for s_i_shrink to be zeros here?
        s_i_shrink = {} # new list for shrinkage tensors

        ## smoothing operations
        for i in range(len(s_i)):

            f = s_i[i]
            # tensor_size = f.size(2)
            if i not in self.beta_D_I_dict.keys():
                s_i_shrink[i] = s_i[i]
            else:
                # Extract Riesz-Quincunx bases:
                beta_I = self.beta_I_dict[i]
                beta_D_I = self.beta_D_I_dict[i]
                psi_in = self.psi_in_dict[i]
                psi_D_in = self.psi_D_in_dict[i]

                # transfer to gpu
                beta_I, beta_D_I, psi_in, psi_D_in = beta_I.cuda(), beta_D_I.cuda(), psi_in.cuda(), psi_D_in.cuda()

                f_re = torch.zeros(f.shape)

                #print("f_re shape: ", f_re.shape)
                # Forward Riesz-Quincunx wavelet:
                for k in range(f_re.size(0)):
                    for l in range(f_re.size(1)):
                        c_I, d_in = RieszQuincunxWaveletTransform_Forward(f[k,l,:,:], beta_D_I, psi_D_in)
                        c_I, d_in = c_I.cuda(), d_in.cuda()
                           
                        # get c_I value
                        c_I = c_I_ori[i][k,l,:,:]
                        c_I = c_I.cuda()
                           
                        # # Shrinkage:
                        # case 1 - alpha-quantile:
                        # alpha = self.alpha
                        # alpha = 0.1
                        activation_method = "SoftShrink"
                        w_ikl = RieszWaveletTruncation(d_in, self.alpha, activation_method)

                        # # case 2 - fixed threshold:
                        # thres = 1/beta
                        # activation_method = "SoftShrink"
                        # w_ikl = RieszWaveletTruncation_FixThres(d_in, thres, activation_method)

                        # Inverse wavelet:
                        f_re[k,l,:,:] = RieszQuincunxWaveletTransform_Inverse(c_I, w_ikl, beta_I, psi_in)        
                               

                f_re = f_re.cuda() 
           
                s_i_shrink[i] = f_re

        for i, module in enumerate(self.up_convs):

            s = s_i_shrink[self.depth-2-i]
            #print("s shape: ", s.shape)
            #print("z shape: ", z_ori.shape)
            if i == 0: ## if i==0 then concatenate the variation layer (z) instead of original downconv tensor (x), then after the loop, x becomes the upconv tensor
                u = module(s, z_ori)
            else:
                u = module(s, u)

        x = u
        x = self.conv_final(x)
        x = F.relu(x)
        # x_recon = F.relu(x)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return x


class UNet_VAE_Stacked(nn.Module):
    def __init__(self, num_classes, segment, alpha, device, model_weight, in_channels=3, unet_num_block=20):
        """
        Arguments:
        """
        super(UNet_VAE_Stacked, self).__init__()

        self.num_classes = num_classes
        self.segment = segment
        self.in_channels = in_channels
        self.unet_num_block = unet_num_block

        self.unet_blocks = []

        for i in range(self.unet_num_block):
            block = UNet_VAE_Block(3, segment, alpha)
            if model_weight is not None:
                block.load_state_dict(torch.load(model_weight))
            self.unet_blocks.append(block)

        self.unet_blocks = nn.ModuleList(self.unet_blocks)

    def forward(self, x):

        #print(self.unet_blocks)
        # go through all blocks
        for i, module in enumerate(self.unet_blocks):
            x = module(x)

        x_recon = x

        #x_recon = F.relu(x)

        return x, x_recon


            
