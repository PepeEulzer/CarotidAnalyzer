import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from monai.networks.nets import UNet
from skimage import morphology
from skimage.exposure import rescale_intensity

from defaults import *

class CarotidDataset(Dataset):
    """
    Subclass of torch dataset that contains a carotid volume.
    """
    def __init__(self, img_data, h=120, w=144, d=248, wl=415, ww=470):
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
        return self.img_data, self.label
    
    def __len__(self):
        return 1
    


class CarotidSegmentationPredictor():
    """
    Wrapper object to call segmentation predictions based on trained UNet.
    """
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.weights = 'seg_model_weights.pth'
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


    def setData(self, img_data):
        self.dataset = CarotidDataset(img_data)
        self.dataloader = DataLoader(self.dataset, batch_size=1)

    
    def run_inference(self):
        pred = None
        for item in self.dataloader:
            # infer label prediction
            img = item[0].type(torch.FloatTensor).unsqueeze(1).to(self.device)
            output = self.model(img).squeeze(0)
            pred = torch.argmax(output, dim=0).cpu().numpy().astype(np.uint8)

            # postprocess
            pred = pred.astype(np.uint8)
            pred = np.rot90(pred, 0, axes=(1, 2))
            pred = morphology.closing(pred) # close small gaps
            region_label_img = morphology.label(pred, connectivity=2)
            region_label_img[pred==1] = 0 # ignore plaque (TODO better method? remove only far away plaque?)
            region_label_hist, _ = np.histogram(region_label_img, bins=np.max(region_label_img)+1)
            for i in range(1, len(region_label_hist)):
                cluster_size = region_label_hist[i]
                if 0 < cluster_size < MIN_CLUSTER_SIZE:
                    pred[region_label_img==i] = 0 # remove small clusters
            pred = morphology.opening(pred) # remove spikes
        return pred

    def discard(self):
        self.dataset = None
        self.dataloader = None