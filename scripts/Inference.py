import os
from collections import OrderedDict
import numpy as np
import nrrd
from glob import glob

import torch
from torch.utils.data import Dataset, DataLoader
from monai.networks.nets import UNet
from skimage import morphology
from skimage.exposure import rescale_intensity


def init_model(device, weights=None):
    model = UNet(
    dimensions=3,
    in_channels=1,
    out_channels=3,
    channels=(16, 32, 64, 128),  
    strides=(2, 2, 2),
    num_res_units=3,
    norm='INSTANCE',
    ).to(device)

    if weights:
        model.load_state_dict(torch.load(weights, map_location=device))

    return model


class CarotidDataset(Dataset):
    def __init__(self, datapaths, labelpaths=None, transform=False, wl=415, ww=470):
        '''
        Args:
            datapaths (list): list of volumetric data paths
            labelpaths (list): list of segmentation file paths
            transform (bool): whether the data should be augmented
            wl (int): window level for preprocessing
            ww (int): window width for preprocessing
        '''
        if labelpaths:  # Test data requires no labels
            self.test = False
            assert len(datapaths) == len(labelpaths), 'Data and labels are not matched.'
        else:
            self.test = True
        self.datapaths = datapaths
        self.labelpaths = labelpaths
        self.transform = transform
        self.wl = wl
        self.ww = ww

    def _load_nrrd(self, filepath, requires_header=False):
        if requires_header:
            data, header = nrrd.read(filepath)
            return data, header
        else:
            data, _ = nrrd.read(filepath)
        return data


    def __getitem__(self, idx):
        if not self.test:
            img = self._load_nrrd(self.datapaths[idx]) 
            label, header = self._load_nrrd(self.labelpaths[idx], True)
            img, label = self._preprocess(img, label, header)
        else:
            img = self._load_nrrd(self.datapaths[idx])
            img = self._preprocess(img)

        fname = os.path.basename(self.datapaths[idx]).split('.')[0]

        # Trasnform for both images and labels:
        # if self.transform:
        #     # Random horizontal flipping
        #     if random.random() > 0.5:
        #         img= TF.hflip(img)
        #         label = TF.hflip(label)

        #     # Random vertical flipping
        #     if random.random() > 0.5:
        #         img = TF.vflip(img)
        #         label = TF.vflip(label)

        if self.test:
            label = torch.zeros(img.shape)
        return img, label, fname

    def __len__(self):
        return len(self.datapaths)

    def _preprocess(self, img, label=None, header=None):

        if not self.test:
            if label.shape != img.shape:
                assert header is not None, 'header is missing'
                seg_ext0, seg_ext1 = header['Segment0_Extent'], header['Segment1_Extent']
                offset = header['Segmentation_ReferenceImageExtentOffset']
                label = self._expand_label(img, label, seg_ext0, seg_ext1, offset)  # Expand label to img size
                label = self._combine_label_channels(label)  # Combine label channels

            # Cropping
            label = self._crop(label)
        img = self._crop(img)
        
        # Windowing:
        upper_threshold = self.wl + self.ww//2
        lower_threshold = self.wl - self.ww//2
        img[img<lower_threshold] = lower_threshold
        img[img>upper_threshold] = upper_threshold

        # Normalization
        # img = (img / 800 - 0.5) * 2
        img = rescale_intensity(img, out_range=(0, 1))

        if self.test:
            return torch.tensor(img)
        else:
            return torch.tensor(img), torch.tensor(label)

    def _get_label_extend(self, seg_ext0, seg_ext1, offset):
        seg_ext0 = list(int(i) for i in seg_ext0.split(' '))
        seg_ext1 = list(int(i) for i in seg_ext1.split(' '))
        x0_min, x0_max, y0_min, y0_max, z0_min, z0_max = seg_ext0
        x1_min, x1_max, y1_min, y1_max, z1_min, z1_max = seg_ext1
        offset = list(int(i) for i in offset.split(' '))
        xo, yo, zo = offset
        return xo+min(x0_min, x1_min), xo+max(x0_max, x1_max)+1, \
               yo+min(y0_min, y1_min), yo+max(y0_max, y1_max)+1, \
               zo+min(z0_min, z1_min), zo+max(z0_max, z1_max)+1

    def _expand_label(self, img, label, seg_ext0, seg_ext1, offset):
        h0, w0, d0 = img.shape
        h, w, d = label.shape
        xa, xb, ya, yb, za, zb = self._get_label_extend(seg_ext0, seg_ext1, offset)

        # Crop if label is larger
        if h>h0: label = label[:h0, :, :]
        if w>w0: label = label[:, :w0, :]
        if d>d0: label = label[:, :, :d0]
        h, w, d = label.shape  # label new shape

        # Expand
        if h<h0 or w<w0 or d<d0:
            label_expanded = np.zeros((h0, w0, d0))
            # Ensure data in bound
            if xb > h0: xb = h0
            if yb > w0: yb = h0
            if zb > d0: zb = d0  #label = label[:, :, :-(zb-d0)]
            label_expanded[xa:xb, ya:yb, za:zb] = label
            return label_expanded
        else:
            return label


    def _combine_label_channels(self, label):
        if len(label.shape) == 4:  # (ch, h, w, d): plaque, lumen
            label3d = np.zeros(label.shape[1:])
            for i in range(label.shape[0]):
                label3d[label[i, ...] == 1] = i+1
        else:
            label3d = label
        return label3d

    def _crop(self, data, h=120, w=144, z=248):
        h0, w0, z0 = data.shape
        assert h0 >= h and w0 >= w and z0 > z, 'cannot crop'
        return data[:h, :w, :z]


def post_processing(pred):
    ''' Remove small objects using morphology processing
    Args:
        pred (numpy array): predicted volume
    Return:
        pred1 (numpy array)
    '''
    binary = pred.copy()
    binary[binary>0] = 1  # Combine foreground
    labels = morphology.label(binary)  # Identify possible foreground clusters
    labels_num = [len(labels[labels==each]) for each in np.unique(labels)]  # Count number of voxels for each cluster
    rank = np.argsort(np.argsort(labels_num))  # Argsort1: cluster index sorted by number of voxels, argsort2:
    index = list(rank).index(len(rank)-2)
    pred1 = pred.copy()
    pred1[labels!=index] = 0
    pred1 = morphology.opening(pred1)
    return pred1


def save_prediction(file_name, pred, header):
    print("Writing", file_name)
    nrrd.write(file_name, pred, header)


def inference(datapaths, weights, device, wl=415, ww=470):
    dataset = CarotidDataset(datapaths, labelpaths=None, wl=wl, ww=ww)
    dataloader = DataLoader(dataset, batch_size=1)
    model = init_model(device, weights).eval()

    i = 0
    for item in dataloader:
        # infer label prediction
        img = item[0].type(torch.FloatTensor).unsqueeze(1).to(device)
        output = model(img).squeeze(0)
        pred = torch.argmax(output, dim=0).cpu().numpy().astype(np.uint8)

        # postprocess
        pred = post_processing(pred)
        pred = pred.astype(np.uint8)
        pred = np.rot90(pred, 0, axes=(1, 2))

        # create header for prediction nrrd
        header_img = nrrd.read_header(datapaths[i])
        header = OrderedDict()
        header['type'] = 'unsigned char'
        header['dimension'] = 3
        header['space'] = 'left-posterior-superior'
        header['sizes'] = '120 144 248' # fixed model size
        header['space directions'] = header_img['space directions']
        header['kinds'] = ['domain', 'domain', 'domain']
        header['endian'] = 'little'
        header['encoding'] = 'gzip'
        header['space origin'] = header_img['space origin']
        header['Segment0_ID'] = 'Segment_1'
        header['Segment0_Name'] = 'plaque'
        header['Segment0_Color'] = str(241/255) + ' ' + str(214/255) + ' ' + str(145/255)
        header['Segment0_LabelValue'] = 1
        header['Segment0_Layer'] = 0
        header['Segment0_Extent'] = '0 119 0 143 0 247'
        header['Segment1_ID'] = 'Segment_2'
        header['Segment1_Name'] = 'lumen'
        header['Segment1_Color'] = str(216/255) + ' ' + str(101/255) + ' ' + str(79/255)
        header['Segment1_LabelValue'] = 2
        header['Segment1_Layer'] = 0
        header['Segment1_Extent'] = '0 119 0 143 0 247'

        save_path = datapaths[i]
        save_path = save_path[:-5] + "_pred.seg" + save_path[-5:]
        save_prediction(save_path, pred, header)
        i += 1

def run_inference():
    #datapaths = ['C:\\Users\\Pepe Eulzer\\Desktop\\pipeline_test\\patient_w_000\\patient_w_000_left.nrrd']
    datapaths = (glob('C:\\Users\Pepe Eulzer\\Nextcloud\\daten_wei_chan\\patient_?_*\\patient_?_*_left.nrrd') +
                 glob('C:\\Users\Pepe Eulzer\\Nextcloud\\daten_wei_chan\\patient_?_*\\patient_?_*_right.nrrd'))
    weights = 'C:\\Git\\carotid-segmentation\\models\\best_model2022-03-12.pth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Running inference! (" + device + ")")
    inference(datapaths, weights, device)

run_inference()
print("Done!")