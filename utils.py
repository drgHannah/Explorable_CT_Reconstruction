import numpy as np
import torch
import torch.nn as nn

class TV(nn.Module):
    ''' Total Variation.'''

    def __init__(self, orig_image=None):
        ''' Initialization.
    
        args:
            orig_image: Used for calculating the weighting of tv based on image values. 
            If orig_image=None, no weighting is applied.
        '''

        super(TV,self).__init__()
        if orig_image is not None:
            self.img = orig_image.unsqueeze(0)
        else:
            self.img = None

    def forward(self,x):
        ''' Returns TV of the 4D (batch size, channels, height, width) input image x.'''

        # TV
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()

        # Weight
        loss_weight = 1
        if self.img is not None:
            gamma = 5
            g_img = torch.mean(self.img, dim=1)
            h_tv_img = torch.pow((g_img[:,1:,:]-g_img[:,:-1,:]),2).sum()
            w_tv_img = torch.pow((g_img[:,:,1:]-g_img[:,:,:-1]),2).sum()
            deriv_img = (torch.abs(h_tv_img/count_h) + torch.abs(w_tv_img/count_w))/batch_size
            loss_weight = torch.exp(-gamma * deriv_img)/2

        return loss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]



def makeGaussian(size, fwhm = 3, center=None):
    """ Create a square gaussian kernel.
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)
