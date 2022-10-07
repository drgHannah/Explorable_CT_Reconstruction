import torch
import os
from radon_transformation import radon 
import numpy as np
from dicom_helper import Scanotation
import glob
from torchvision import transforms

class IDLC_Dataset(torch.utils.data.Dataset):
    def __init__(self, angles = 50, mode = "train"):
        ''' Dataset for optimizing towards a given malignancy. 
            Loads its data from the file ./data/[mode]/angle-[angles]/...

        args:
            angles: number of prjection angles
            mode: either "train" or "valid"
        '''

        base = "./data/{}/angle-{}/*/*".format(mode, angles)
        all_files = glob.glob(base+'*')
        all_files.sort()
        self.data = []
        for file in all_files:
            self.data.append(np.load(file))
        print("Loaded {} dataitems...".format(len(self.data)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ''' Returns: full dose, low dose, sinogram, nodule pos, angles, malignancy
        '''
        data_item = self.data[idx]
        slice = torch.tensor(data_item['slice'])[None]
        sino = torch.tensor(data_item['sino'])[None]
        low_dose = torch.tensor(data_item['low_dose'])[None]
        loc = torch.tensor(data_item['loc'])[:2]
        malig = torch.tensor(data_item['malig'])
        angles = torch.tensor(data_item['angles'])
        return slice, sino, low_dose, loc, malig, angles 


class IDLC_Dataset_Crop(torch.utils.data.Dataset):
    def __init__(self, angles = 50, mode = "train", nodule_size = 16,  device = "cpu"):
        ''' Dataset for training a classificatio network on cropped nodules. 
            Loads its data from the file ./data/[mode]/angle-[angles]/...

        args:
            angles: number of prjection angles
            mode: either "train" or "valid"
            nodule_size: size of crop around the nodules
            device: cpu or cuda
        '''

        base = "./data/{}/angle-{}/*/*".format(mode, angles)
        all_files = glob.glob(base+'*')
        self.data = []
        self.files = all_files
        self.mode=mode
        self.percentage_adv_sp = None
        self.percentage_adv_g = None
        assert nodule_size > 10

        self.transform = torch.nn.Sequential(
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=180),
            transforms.RandomCrop(size=(nodule_size*2*2-5, nodule_size*2*2-5))
        )
        self.device = device
        self.nodule_size = nodule_size
        self.data_len=len(all_files)
            
    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        '''
        Returns: full dose, low dose, sinogram, nodule pos, angles, malignancy
        '''
    
        data_item = np.load(self.files[idx])
        low_dose = torch.tensor(data_item['low_dose'])[None]
        loc = torch.tensor(data_item['loc'])[:2]
        malig = torch.tensor(data_item['malig'])
        if self.mode == "train":
            cropped = crop_center(low_dose[None], loc[None], size=self.nodule_size*2)[0]
            cropped = self.transform(cropped)
            c,h,w = cropped.shape
            loc_new = torch.tensor([[h/2,w/2]])
            cropped = crop_center(cropped[None], loc_new, size=self.nodule_size)[0]
            
        else:
            cropped = crop_center(low_dose[None], loc[None], size=self.nodule_size)[0]
        return cropped.float(), malig.long()


def crop_center(input, center, size):
    ''' Crops at center of the input. 

    args:
        input: input image (size: bsz x c x w x h)
        center: location to crop
        size: size of crop
    '''

    cropped = torch.zeros([center.shape[0],1,size*2,size*2]).to(input.device)
    for i in range(center.shape[0]):
        if np.floor(center[i,1])==center[i,1]:
            center[i,1]+=1e-4
        if np.floor(center[i,0])==center[i,0]:
            center[i,0]+=1e-4
        left_crop = -int(np.floor(0 + center[i,1]-size))
        right_crop = -int(np.floor(input.shape[2] - (center[i,1]+size)))-1
        upper_crop = -int(np.floor(0 + center[i,0]-size))
        lower_crop = -int(np.floor(input.shape[3] - (center[i,0]+size)))-1
        cropped[i]=torch.nn.functional.pad(input[i],(left_crop,right_crop,upper_crop,lower_crop))
    return cropped


def prepare_data(angles, patients=range(1,1000), device='cpu'):
    ''' Preparation of data. Needs to be applied before training and optimization.

    args:
        angles: number of projection angles
        patients: patient ids the be prpared
        device: cpu or cuda
    '''

    radon_t, iradon_t = radon.get_operators(n_angles=angles, image_size=512, circle = True, device=device)
    for patient in patients:
        classif = 'valid'
        base = "./data/{}/angle-{}/".format(classif,angles)#data
        cancer_paths = [os.path.join(base,"noncancer"),os.path.join(base,"cancer")]
        try:
            scan=Scanotation(patient)
            for i in range((scan.no_nodules)):
                malig = scan.malignancy(i)
                if classif=='valid' and len(glob.glob(os.path.join(cancer_paths[malig],'*'))) >= 50:
                    classif = 'train'
                    base = "./data/{}/angle-{}/".format(classif,angles)#data
                    cancer_paths = [os.path.join(base,"noncancer"),os.path.join(base,"cancer")]
                os.makedirs(cancer_paths[malig],exist_ok=True)
                name = 'LIDC-IDRI-{}-NODULE-{}'.format(str(patient).zfill(4),str(i).zfill(3))
                full_path = os.path.join(cancer_paths[malig],name)
                if os.path.exists(full_path):
                    continue
                if malig > -1:
                    slice = torch.tensor(scan.nodule_slice(i))[None,None]
                    loc = torch.tensor(scan.center(i))
                    slice = (slice-slice.min())/(slice.max()-slice.min())
                    sinogram = radon_t(slice)
                    low_dose = iradon_t(sinogram)
                    np.savez_compressed(full_path, 
                        slice=slice[0,0],
                        sino=sinogram[0,0],
                        low_dose=low_dose[0,0],
                        loc=loc,
                        malig=malig,
                        angles=angles)
        except:
            print("Ups, somthing went wrong with patient",patient)

if __name__ == '__main__':
    # prepare data for 100 angles 
    prepare_data(100,range(1,1200),True)
