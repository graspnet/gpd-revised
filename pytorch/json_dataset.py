import json
import numpy as np
import torch
import tqdm
import cv2
import torch.utils.data as torchdata


class JsonDataset(torchdata.Dataset):
    def __init__(self, file_path, start=0, end=None):
        super(JsonDataset, self).__init__()
        with open(file_path, 'r') as data_file:
            data_dict = json.load(data_file)
        img_path_list = data_dict['images'][start : end]
        img_labels = data_dict['labels'][start : end]
        all_img = []
        print("Open Image Files ...")
        for img_path in tqdm.tqdm(img_path_list):
            all_img.append(cv2.imread(img_path))
        print("Successfully open", len(all_img), "images.")
        self.images = torch.from_numpy(np.array(all_img))
        self.labels = torch.from_numpy(np.array(img_labels)).to(torch.int32)


    def __getitem__(self, index):
        img = self.images[index]
        img_label = self.labels[index]
        img = img[:, :].to(torch.float32) * 1/256.0
        # Pytorch uses NCHW format
        img = img.permute(2, 0, 1)
        return (img, img_label)

    def __len__(self):
        return len(self.labels)
