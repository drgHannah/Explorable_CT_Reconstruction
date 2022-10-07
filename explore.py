import torch
from tqdm import tqdm
import utils
from radon_transformation import radon 
import dataset as ds

import torch
import torchvision
import numpy as np
import utils
import warnings
import itertools
from network import BasicResnet

import warnings
warnings.filterwarnings("ignore")


class Optim:

    def __init__(self, angles, name= "model.pt", device = "cuda") -> None:
        '''
        '''

        self.device = device
        self.angles = angles
        self.name = name
        self.nosz = 16

        # load model
        self.model = self.load_network()
        self.model.to(device)

        # get data
        self.dataset = ds.IDLC_Dataset(angles = angles, mode = "valid")

        # classification loss and model predition
        mse_loss = torch.nn.HuberLoss(delta=0.01)
        self.en_class = lambda x, model, pos, malig: mse_loss(torch.nn.functional.softmax(model(ds.crop_center(x, pos, self.nosz)),dim=1)[:,1],malig)
        self.en_class_nocrop = lambda x, model, malig: mse_loss(torch.nn.functional.softmax(model(x),dim=1)[:,1],malig)
        self.model_pred = lambda  x, model, pos: torch.nn.functional.softmax(model(ds.crop_center(x, pos,self.nosz)),dim=1)[:,1]
        
        # total variation
        tova = utils.TV()
        self.c_tova = lambda xt2, loc, nosz: tova(ds.crop_center(xt2, loc,nosz))
        self.frame=[]

    def load_network(self):
        ''' Load and return the pretrained classification network.
        '''

        # get path
        net_path = self.name
        # load network
        model = BasicResnet()
        model.load_state_dict(torch.load(net_path))
        model.eval()
        print("Loaded", net_path)
        return model

    def prepared_data(self, idx):
        ''' Load and prepare an data item. Define the radon loss E_1.

        args:
            idx: index of dataitem

        return:
            mean and std of cropped low dose, the true reconstruction, the sinogram, 
            the low dose, the location of the nodule of interest, 
            the malignancy of the nodule and the number of angles
        '''

        # get data
        slice, _, _, loc, malig, angles = self.dataset[idx]
        malig = malig[None].to(self.device)
        loc = loc[None]
        slice = slice[None]

        # load and apply radon transform
        self.radon_t, self.iradon_t = radon.get_operators(n_angles=angles, image_size=slice.shape[-1], circle=1, device='cpu')
        sino = self.radon_t(slice).float()
        low_dose = self.iradon_t(sino).float()

        # energy E_1
        radon_t, _ = radon.get_operators(n_angles=angles, image_size=slice.shape[-1], circle=1, device='cpu')
        radon_t.rotated = radon_t.rotated.to(self.device)
        self.en_radon = lambda x, sino: torch.mean((radon_t(x) - sino)**2)

        # mean and std of low dose data
        tmean = ds.crop_center(low_dose, loc, self.nosz).mean([1,2,3])[:,None,None,None]
        tstd = ds.crop_center(low_dose, loc, self.nosz).std([1,2,3])[:,None,None,None]
        return tmean.to(self.device), tstd.to(self.device), slice.to(self.device), \
                sino.to(self.device), low_dose.to(self.device), loc, malig, angles
        
    def create_gradient_mask(self, loc, bzs, shape_val=512+200, steps = 4):
        ''' Creates and returns the gaussian mask G.

        args:
            loc: the location of the nodule
            bzs: the batchsize of the data
            shape_val: size of one axis of the reconstruction
            steps: std of the gaussian mask G

        return:
            gaussian mask
        '''

        steps = steps * (self.nosz // 16)
        gaussian_mask=[]
        for i in range(bzs):
            gaussian_mask.append(torch.tensor(utils.makeGaussian(shape_val, fwhm=steps, center=np.array([loc[i][1],loc[i][0]])))[None])
        gaussian_mask = torch.stack(gaussian_mask,dim=0)
        return (gaussian_mask).to(self.device)

    def measure_error(self, loc, xt, sino):
        ''' Measures interior and exterior error (e_i and e_o).

        args:
            loc: location of the nodule
            xt: reconstruction
            sino: corresponding sinogram
        
        return:
            interior error, exterior error
        '''

        xt = xt.detach().cpu()
        sino = sino.detach().cpu()
        def get_mask(loc, shape_val=512):
            mask = torch.zeros([loc.shape[0],1,shape_val,shape_val])
            for i in range(loc.shape[0]):
                mask[i,:,int(loc[i,0]-self.nosz):int(loc[i,0]+self.nosz),int(loc[i,1]-self.nosz):int(loc[i,1]+self.nosz)] = 1
                mask_sino = self.radon_t(mask) > 0
            return mask, mask_sino
    
        _, mask_sino = get_mask(loc,xt.shape[-1])
        sinogram_calc = self.radon_t(xt)
    
        # apply mask on nodule
        masked_nodule_xt = sinogram_calc * mask_sino
        masked_nodule = sino * mask_sino
        norm_nodule = torch.sum((masked_nodule_xt-masked_nodule)**2,dim=[2,3])/torch.sum(mask_sino,dim=[2,3])

        # apply mask on surrounding
        masked_sur_xt = sinogram_calc * ~mask_sino
        masked_sur = sino * ~mask_sino
        norm_sur = torch.sum((masked_sur_xt-masked_sur)**2,dim=[2,3])/torch.sum(~mask_sino,dim=[2,3])
        return norm_nodule, norm_sur


    def add_perms(self, x, pos, tmean, tvar, perms=None):
        ''' Add permutations to the input.

        args:
            x: input reconstruction
            pos: position of the nodule
            tmean: mean value of the nodule (ideally after filtered backprojection)
            tvar: mean value of the nodule (ideally after filtered backprojection)
            perms: permutations to apply

        return:
            permutated input, tmean and tvar 
        '''
        
        if perms is not None:
            s1 = perms['shift']
            reszs = perms['zoom'](self.nosz * 2)
            rot_as = perms['rotate']
        else:
            s1 = [0]
            red = self.nosz * 2
            reszs = [red]
            rot_as = list(range(0, 360, 360//7))

        shifts = list(itertools.product(s1, s1))
        shifts = list(itertools.product(rot_as, shifts))
        perms = []
        tmeans = []
        tvars = []
        for rot_a, shift in shifts:
            xc = ds.crop_center(x, pos+torch.tensor(shift), self.nosz*2)
            xc = torchvision.transforms.functional.rotate(xc, angle=rot_a)
            xc = ds.crop_center(xc, torch.tensor([[(xc.shape[2]//2),(xc.shape[3]//2)]]).repeat([xc.shape[0],1]).float(), self.nosz)
            if shift == (0,0) and reszs!=[]:
                for resz in reszs:
                    padv = (self.nosz*2 - resz)//2
                    xci = torch.nn.functional.interpolate(xc, size=resz,mode='bilinear',align_corners=False)
                    xci = torch.nn.functional.pad(xci, (padv,padv,padv,padv), mode='replicate')
                    perms.append(xci)
                    tmeans.append(tmean)
                    tvars.append(tvar)
            else:
                perms.append(xc)
                tmeans.append(tmean)
                tvars.append(tvar)
        return torch.cat(perms,0), torch.cat(tmeans,0), torch.cat(tvars,0)

    def pretrain_lowdose(self, low_dose, sino):
        ''' Pre-calculation of the reconstruction.

        args:
            low_dose: initial reconstruction, ideally the low dose
            sino: corresponding sinogram

        return:
            optimized reconstruction
        '''

        print("Pre-calulate the input...")
        low_dose[low_dose>1]=1
        low_dose[low_dose<0]=0
        xt = low_dose.detach().clone().to(self.device).float()
        xt.requires_grad=True
        pbar = tqdm(range(600), disable=False)
        optim=torch.optim.SGD([xt], lr=0.0005, momentum=0.9)
        start_en = self.en_radon(xt, sino)
        for iter in pbar:
            optim.zero_grad()
            loss = self.en_radon(xt, sino)
            loss.backward()
            optim.step()
            with torch.no_grad():
                xt[xt>1]=1
                xt[xt<0]=0
        return xt.detach().clone().to(self.device).float()

    def optimize(self, dataid, epochs = 2000, mask_w=11, w_r=1.0, w_c=1.0, w_tv=0.01, lr=1.0, opt_to = None, perms = None):
        ''' Data consistant optimization of the data towards a pre-defined malignancy.

        args:
            dataid: data index
            epochs: optimization iteration
            mask_w: width of the gaussian mask
            w_r: weighting of the radon loss E_1
            w_c: weighting of the classification loss (E_2 (part 1))
            w_tv: weighting of the tv loss (E_2 (part 2))
            lr: learning rate
            opt_to: optimization towrds this malignancy. If none, the malignancy is chosen to be opposite the true malignancy value.
            perms: permutation

        return:
            the resulting reconstruction, a list with the interior and exterior errors, 
            the loss, the radon loss, the classificatio loss E2, the predictions of the network, the stopping iteration,
            a tupel containing: (mean value of the low dose, std of the low dose, original rec., sinogram, low_dose, 
                                location of the nodule, malignancy, number of angles, final sinogram, low dose sinogram)

        '''
        # saves
        tmeans = []
        tstds = []
        mean_loss = []
        mean_loss_prev = 1e9

        with torch.no_grad():
            tmean, tstd, slice, sino, low_dose, loc, malig, angles = self.prepared_data(dataid)
            ld_sino = np.abs(self.radon_t(low_dose.cpu()) - sino.cpu())

        # pretrain on E_1
        pretrained_lowdose = self.pretrain_lowdose(low_dose, sino)
        low_dose=pretrained_lowdose
        low_dose[low_dose>1]=1
        low_dose[low_dose<0]=0
        xt = low_dose.detach().clone().to(self.device).float()
        xt.requires_grad=True

        # optimizer
        optim=torch.optim.SGD([xt], lr=lr, momentum=0.0)

        # optimize to
        with torch.no_grad():
            if opt_to is None:
                opt_to = 1.0-malig
            else:
                opt_to = torch.tensor(opt_to)[None].to(self.device)

        # gradient mask
        gradient_mask=self.create_gradient_mask(loc, low_dose.shape[0], shape_val=low_dose.shape[-1], steps = mask_w)
        # saves
        losses = []
        losses_r = []
        losses_c = []
        preds=[]
        error_list = []
        self.frame=[]
        # save iteration 0
        with torch.no_grad():
            self.frame.append((ds.crop_center(low_dose, loc, size=2*self.nosz)).cpu().detach().numpy())
            preds.append(self.model_pred((low_dose-tmean)/tstd, self.model, loc).cpu().detach().numpy())
        pbar = tqdm(range(epochs), disable=False)
        stopiter = epochs
        for iter in pbar:
            optim.zero_grad()
            # add gradient mask
            xt2 = gradient_mask * xt + (1 - gradient_mask) * xt.detach()
            # total variation
            tv = self.c_tova(xt2, loc, self.nosz)
            # add permutation
            xt2,tmeans,tstds = self.add_perms(xt2, loc, tmean, tstd, perms)
            # calculate loss
            en_c = self.en_class_nocrop((xt2-tmeans)/tstds, self.model, opt_to)
            en_r = self.en_radon(xt, sino)
            loss = w_r * en_r + w_c * en_c + w_tv * tv
            loss.backward()
            optim.step()
            with torch.no_grad():
                xt[xt>1]=1
                xt[xt<0]=0
            with torch.no_grad():
                mean_loss.append(loss.item())
                # check every 500 iteration for updating the learning rate or early stopping
                if len(mean_loss) > 500:
                    meanlosscalc = sum(mean_loss) / len(mean_loss)
                    condition = (meanlosscalc >= mean_loss_prev) 
                    clr = optim.param_groups[0]['lr']
                    if condition and (en_r<0.01 or clr<1e-5):
                        print(f"Stopped at iteration {iter} with loss: {meanlosscalc}, E1: {en_r}, lr<1e-5: {clr<1e-5}")
                        stopiter = iter
                        break
                    elif condition:
                        optim.param_groups[0]['lr'] = clr * 0.8
                        print(f"Learning rate: {optim.param_groups[0]['lr']}")
                        mean_loss = []
                    else:
                        mean_loss_prev = meanlosscalc
                        mean_loss = []
                # save loss and prediction
                if (iter+1) % 10 == 0:
                    self.frame.append((ds.crop_center(xt, loc, size=2*self.nosz)).cpu().detach().numpy())
                    preds.append(self.model_pred((xt-tmean)/tstd, self.model, loc).cpu().detach().numpy())
                    losses.append(loss.item()) 
                    losses_r.append(en_r.item()) # (radon loss E_1)
                    losses_c.append(en_c.item()) # (class loss E_2 (w/o tv))
                    pbar.set_description("loss: {:.04f}, radon: {:.04f}, class: {:.04f}, pred: {:.04f}".format(losses[-1], losses_r[-1], losses_c[-1], preds[-1][0]))
                # save interior and exterior error
                if (iter+1) % 1000 == 0:
                    errors = self.measure_error(loc, xt, sino)
                    error_list.append((iter, errors))
        # save error and resulting sinogram
        with torch.no_grad():
            error_list.append((iter, self.measure_error(loc, xt, sino)))
            end_sino= np.abs(self.radon_t(xt.detach().cpu()) - sino.cpu())
        del self.en_radon
        torch.cuda.empty_cache()
        return xt.detach().cpu(), error_list, losses, losses_r, losses_c, preds, stopiter, \
                (tmean , tstd , slice.cpu(), sino.cpu(), low_dose.cpu(), loc.cpu(), malig, angles.item(),end_sino,ld_sino)