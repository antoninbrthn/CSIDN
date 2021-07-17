'''
Code for paper "Confidence Scores Make Instance-dependent Label-noise Learning Possible"
Antonin Berthon, 2021
-----------
Script description:
Utility functions for the Clothing1M dataset.
'''

import os
import tqdm
import numpy as np

from PIL import Image

import torch.utils.data
import torchvision.transforms as transforms


class Clothing1M(torch.utils.data.Dataset):
    def __init__(self, path, transform=None, mode='clean_train', dataset_type='drive'):
        assert mode in ['clean_test', 'noisy_train', 'custom_noisy_train', 'clean_train',
                        'clean_val'], "invalid mode"
        assert dataset_type in ['drive', 'custom'], "invalid dataset_type"
        self.dataset_type = dataset_type
        self.mode = mode
        self.path = path
        self.transform = transform
        if self.dataset_type == 'drive':
            self.imlist = self.get_imlist_drive()
        else:
            self.imlist = self.get_imlist_custom()

    def __getitem__(self, i):
        img_name, label = self.imlist[i]
        if self.dataset_type == 'drive':
            img_path = os.path.join(self.path, img_name)
        else:
            img_path = os.path.join(self.path, self.mode, label, img_name)

        label = int(label)
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label, img_path.replace(self.path, "")

    def get_imlist_drive(self):
        if self.mode[:5] == 'clean':
            label_file = os.path.join(self.path, "clean_label_kv.txt")
            key_list = os.path.join(self.path, self.mode + "_key_list.txt")
        elif self.mode[:5] == 'noisy':
            label_file = os.path.join(self.path, "noisy_label_kv.txt")
            key_list = os.path.join(self.path, self.mode + "_key_list.txt")
        elif self.mode == 'custom_noisy_train':
            label_file = os.path.join(self.path, "noisy_export.txt")
            key_list = os.path.join(self.path, "noisy_train_key_list.txt")
        else:
            raise Exception("invalid label file name")

        self.label_dict = {}
        # Build label lookup table
        print("Building label lookup table")
        with open(label_file, "r") as f:
            i = 0
            for line in f.readlines():
                i += 1
                line = line.replace("\n", "")
                row = line.split(" ")
                img_path = row[0]
                label = row[1]
                # try opening image
                full_img_path = os.path.join(self.path, img_path)
                if not os.path.exists(full_img_path):
                    continue
                else:
                    self.label_dict[img_path] = label

        # Build file list
        l = []
        with open(key_list, "r") as f:
            i = 0
            not_found = 0
            for line in f.readlines():
                name = line.replace("\n", "")
                try:
                    i += 1
                    label = self.label_dict[name]
                except KeyError:
                    not_found += 1
                    continue
                l.append((name, label))
        print(f"read {i} imgs")
        print(f"did not find {not_found} imgs")
        # print(f"found {i - not_found}/{i} imgs")
        return l

    def get_imlist_custom(self):
        l = []
        class_dir = os.path.join(self.path, self.mode)
        for label in tqdm.tqdm(os.listdir(class_dir)):
            file_dir = os.path.join(class_dir, label)
            for name in os.listdir(file_dir):
                l.append((name, label))
        return l

    def __len__(self):
        return len(self.imlist)


class Clothing1M_confidence(torch.utils.data.Dataset):
    def __init__(self, path, fn, transform=None, dataset_type='drive', og_labels=False):
        self.path = path
        self.fn = fn
        self.transform = transform
        self.og_labels = og_labels
        self.imlist = self.get_imlist()

    def __getitem__(self, i):
        img_name, label, conf, _ = self.imlist[i]
        img_path = os.path.join(self.path, img_name)

        label = int(label)
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label, conf

    def get_imlist(self):
        label_file = os.path.join(self.path, self.fn)
        print(f"Clothing1M_confidence: Loading {label_file} with {'original labels' if self.og_labels else 'new labels'}")

        self.label_dict = {}
        l = []
        # Build label lookup table
        with open(label_file, "r") as f:
            i = 0
            for line in f.readlines():
                line = line.replace("\n", "")
                row = line.split(" ")
                assert len(row) == 5
                img_path = row[0]
                label = int(row[1])
                conf = float(row[2])
                old_label = int(row[3])
                old_conf = float(row[4])

                if self.og_labels:  # use old label and corresponding confidence
                    conf = old_conf
                    label = old_label
                # try opening image
                full_img_path = os.path.join(self.path, img_path)
                if not os.path.exists(full_img_path):
                    continue
                else:
                    i += 1
                    self.label_dict[img_path] = (label, conf)
                    l.append((img_path, label, conf, old_label))
            print(f"read {i} lines")
        return l

    def get_labels(self):
        y_noisy, y_true, r = [], [], []
        for (img, lab, conf, old_lab) in self.imlist:
            y_noisy.append(lab)
            y_true.append(old_lab)
            r.append(conf)
        return np.array(y_noisy), np.array(y_true), np.array(r)

    def __len__(self):
        return len(self.imlist)


train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
