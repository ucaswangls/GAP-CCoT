from torch.utils.data import Dataset
import os
import os.path as osp
import scipy.io as scio
import numpy as np 
from image_utils import *
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")
class TrainData(Dataset):
    def __init__(self, train_data_path,mask):
        self.data_path = train_data_path
        self.mask = mask
        self.scene_list = []
        for i in range(50):
            for scene in os.listdir(train_data_path):
                self.scene_list.append(scene)

    def __getitem__(self, index):
        scene_path = osp.join(self.data_path,self.scene_list[index])
        img_dict = scio.loadmat(scene_path)
        if "img_expand" in img_dict:
            img = img_dict['img_expand']/65536.
        elif "img" in img_dict:
            img = img_dict['img']/65536.
        img = img.astype(np.float32)
        img = random_scale(img)
        nc,crop_h = self.mask.shape[:2]
        crop_img = shuffle_crop(img,crop_h)
        crop_img = random_h_flip(crop_img)
        meas = gen_meas(crop_img,self.mask)
        nc = self.mask.shape[0]
        gt = crop_img[0:nc] 
        return meas,gt


    def __len__(self):
        return len(self.scene_list)

class TestData(Dataset):
    def __init__(self,data_path,mask):
        self.data_path = data_path
        self.mask = mask
        self.scene_list = []
        for scene in os.listdir(data_path):
            self.scene_list.append(scene)
        self.scene_list.sort()

    def __getitem__(self,index):
        scene_path = osp.join(self.data_path,self.scene_list[index])
        img_dict = scio.loadmat(scene_path)
        img = img_dict['img']
        img = img.astype(np.float32).transpose((2,0,1))
        meas = gen_meas(img,self.mask)

        nc = self.mask.shape[0]
        gt = img[0:nc] 
        return meas,gt
    def __len__(self):
        return len(self.scene_list)