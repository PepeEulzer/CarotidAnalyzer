import os
from collections import OrderedDict
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from monai.networks.nets import UNet
from skimage import morphology
from skimage.exposure import rescale_intensity

class CarotidDataset(Dataset):
    """
    Subclass of torch dataset that contains a carotid volume.
    """
    def __init__(self, img_data, filename, h=120, w=144, d=248, wl=415, ww=470):
        """
            Args:
            img_data (numpy array): image volume data
            h (int): height
            w (int): width
            d (int): depth
            wl (int): window level for preprocessing
            ww (int): window width for preprocessing
        """
        self.img_data = np.copy(img_data)
        self.label = torch.zeros(img_data.shape)
        self.filename = filename

        # crop if necessary
        h0, w0, d0 = self.img_data.shape
        if h0 != h or w0 != w or d0 != d:
            assert h0 >= h and w0 >= w and d0 >= d, 'cannot crop'
            self.img_data = self.img_data[:h, :w, :d]

        # windowing
        upper_threshold = wl + ww//2
        lower_threshold = wl - ww//2
        self.img_data[self.img_data < lower_threshold] = lower_threshold
        self.img_data[self.img_data > upper_threshold] = upper_threshold

        # normalization
        self.img_data = rescale_intensity(self.img_data, out_range=(0, 1))

        # cast to tensor
        self.img_data = torch.tensor(self.img_data)


    def __getitem__(self, idx):
        return self.img_data, self.label, self.filename
    
    def __len__(self):
        return 1
    


class CarotidSegmentationPredictor():
    """
    Wrapper object to call segmentation predictions based on trained UNet.
    """
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.weights = 'C:\\Git\\carotid-segmentation\\models\\best_model2022-03-12.pth' # TODO adapt path
        self.model = UNet(spatial_dims=3,
                          in_channels=1,
                          out_channels=3,
                          channels=(16, 32, 64, 128),  
                          strides=(2, 2, 2),
                          num_res_units=3,
                          norm='INSTANCE',
                          ).to(self.device)
        self.model.load_state_dict(torch.load(self.weights, map_location=self.device))
        self.model.eval()
        self.dataset = None
        self.dataloader = None


    def setData(self, img_data, filename):
        self.dataset = CarotidDataset(img_data, filename)
        self.dataloader = DataLoader(self.dataset, batch_size=1)

    
    def run_inference(self):
        pred = None
        for item in self.dataloader:
            # infer label prediction
            img = item[0].type(torch.FloatTensor).unsqueeze(1).to(self.device)
            output = self.model(img).squeeze(0)
            pred = torch.argmax(output, dim=0).cpu().numpy().astype(np.uint8)

            # TODO postprocess
            pred = pred.astype(np.uint8)
            pred = np.rot90(pred, 0, axes=(1, 2))
        return pred

    def discard(self):
        self.dataset = None
        self.dataloader = None



