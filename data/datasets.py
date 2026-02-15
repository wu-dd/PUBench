import sys
import os
import torch
from torchvision import transforms
import torchvision.datasets as dsets
from torch.utils.data import Dataset
import numpy as np
from .randaugment import RandAugmentMC
from .imagenette_randaugment import RandAugment
from .transform import RandomResizedCropAndInterpolation
from scipy.io import loadmat
from PIL import Image
import pandas as pd
import math
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)
usps_mean = (0.1307,)
usps_std = (0.3081,)
DATASETS = [
    
]

IMAGE_DATASETS = [
    "CIFAR10",
    "IMAGENETTE"
]

TABULAR_DATASETS = [
    "Letter",
    "USPS",
    "Creditcard"
]           


def CIFAR10(args):
    class CIFAR10SSL(dsets.CIFAR10):
        def __init__(self, root, indexs, train=True,
                     transform=None, target_transform=None,
                     download=False):
            super().__init__(root, train=train,
                             transform=transform,
                             target_transform=target_transform,
                             download=download)
            if indexs is not None:
                self.data = self.data[indexs]
                self.targets = np.array(self.targets)[indexs]
        def __getitem__(self, index):
            img, target = self.data[index], self.targets[index]
            img = Image.fromarray(img)
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return index, img, target
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])  
    base_dataset = dsets.CIFAR10(args.data_dir, train=True, download=True)
    if args.u_type == 'one_sample':
        train_positive_idxs, train_unlabeled_idxs = p_u_split_one_sample(args, base_dataset.targets)
    elif args.u_type == 'two_sample':
        train_positive_idxs, train_unlabeled_idxs = p_u_split_two_sample(args, base_dataset.targets)
    p_dataset = CIFAR10SSL(
        args.data_dir, train_positive_idxs, train=True,
        transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std),
        target_transform=TransformPTarget(positive_label_list=args.positive_label_list))
    u_dataset = CIFAR10SSL(
        args.data_dir, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std),
        target_transform=TransformUTarget(positive_label_list=args.positive_label_list)
        )
    test_dataset = dsets.CIFAR10(
        args.data_dir, train=False, transform=transform_val, download=False,
        target_transform=TransformTestTarget(positive_label_list=args.positive_label_list))
    return p_dataset, u_dataset, test_dataset

def IMAGENETTE(args):
    class ImageNetteSSL(dsets.ImageFolder):
        def __init__(self, root, indexs=None, transform=None, target_transform=None):
            super().__init__(root, transform=transform, target_transform=target_transform)
            if indexs is not None:
                self.data = [self.samples[i] for i in indexs]
                self.targets = np.array([s[1] for s in self.data])
            else:
                self.targets = np.array([s[1] for s in self.samples])

        def __getitem__(self, index):
            path, target = self.data[index]
            img = Image.open(path).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return index, img, target

        def __len__(self):
            return len(self.data)

    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)

    crop_ratio=0.875
    img_size=64 # consistent with USB
    transform_val = transforms.Compose([
        transforms.Resize(math.floor(int(img_size / crop_ratio))),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    # base dataset for splitting indices
    base_dataset = dsets.ImageFolder(os.path.join(args.data_dir, "train"))

    if args.u_type == 'one_sample':
        train_positive_idxs, train_unlabeled_idxs = p_u_split_one_sample(args, base_dataset.targets)
    elif args.u_type == 'two_sample':
        train_positive_idxs, train_unlabeled_idxs = p_u_split_two_sample(args, base_dataset.targets)

    p_dataset = ImageNetteSSL(
        os.path.join(args.data_dir, "train"), indexs=train_positive_idxs,
        transform=IMAGENETTE_Transform(mean=imagenet_mean, std=imagenet_std),
        target_transform=TransformPTarget(positive_label_list=args.positive_label_list))
    

    u_dataset = ImageNetteSSL(
        os.path.join(args.data_dir, "train"), indexs=train_unlabeled_idxs,
        transform=IMAGENETTE_Transform(mean=imagenet_mean, std=imagenet_std),
        target_transform=TransformUTarget(positive_label_list=args.positive_label_list))


    test_dataset = dsets.ImageFolder(
        os.path.join(args.data_dir, "val"),
        transform=transform_val,
        target_transform=TransformTestTarget(positive_label_list=args.positive_label_list)
    )

    return p_dataset, u_dataset, test_dataset

def USPS(args):
    class USPSSSL(dsets.USPS):
        def __init__(self, root, indexs, train=True,
                     transform=None, target_transform=None,
                     download=False):
            super().__init__(root, train=train,
                             transform=transform,
                             target_transform=target_transform,
                             download=download)
            if indexs is not None:
                self.data = self.data[indexs]
                self.targets = np.array(self.targets)[indexs]
        def __getitem__(self, index):
            img, target = self.data[index], self.targets[index]
            img = Image.fromarray(img)
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return index, img, target

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=usps_mean, std=usps_std)
    ])
    base_dataset = dsets.USPS(args.data_dir, train=True, download=True)
    if args.u_type == 'one_sample':
        train_positive_idxs, train_unlabeled_idxs = p_u_split_one_sample(args, base_dataset.targets)
    elif args.u_type == 'two_sample':
        train_positive_idxs, train_unlabeled_idxs = p_u_split_two_sample(args, base_dataset.targets)

    p_dataset = USPSSSL(
        args.data_dir, train_positive_idxs, train=True,
        transform=USPSFixMatch(mean=usps_mean, std=usps_std),
        target_transform=TransformPTarget(positive_label_list=args.positive_label_list))
    u_dataset = USPSSSL(
        args.data_dir, train_unlabeled_idxs, train=True,
        transform=USPSFixMatch(mean=usps_mean, std=usps_std),
        target_transform=TransformUTarget(positive_label_list=args.positive_label_list)
    )
    test_dataset = dsets.USPS(
        args.data_dir, train=False, transform=transform_val, download=True,
        target_transform=TransformTestTarget(positive_label_list=args.positive_label_list))
    return p_dataset, u_dataset, test_dataset


def Creditcard(args):
    class CreditcardSSL():
        def __init__(self, root, indexs,transform=None, target_transform=None, train=True):
            base_dataset=pd.read_csv(root)
            data, targets=np.array(base_dataset.iloc[:, :-1].values),np.array(base_dataset['Class'].values)
            sc = StandardScaler()
            self.data, self.targets = sc.fit_transform(data), 1-targets
            self.transform = transform
            self.target_transform = target_transform
            self.train=train
            if indexs is not None:
                self.data = self.data[indexs]
                self.targets = np.array(self.targets)[indexs]
        def __getitem__(self, index):
            img, target = self.data[index], self.targets[index]
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)

            if self.train==True:
                return index, img, target
            else:
                return img[0],target
        def __len__(self):
            return len(self.data)

    file_path = os.path.join(args.data_dir,'creditcard.csv')
    base_dataset = pd.read_csv(file_path)
    data, targets=np.array(base_dataset.iloc[:, :-1].values),np.array(base_dataset['Class'].values)
    sc = StandardScaler()
    data = sc.fit_transform(data)
    targets = 1 - targets  # positive: fraud(1), negative: normal(0)


    total_size=data.shape[0]
    train_size=int(total_size*(1-args.tabular_test_fraction))
    keys = list(range(total_size))
    np.random.RandomState(args.seed).shuffle(keys)
    train_idx = keys[:train_size]
    test_idx = keys[train_size:]
    if args.u_type == 'one_sample':
        train_positive_idxs, train_unlabeled_idxs = p_u_split_one_sample(args, targets[train_idx])
        train_positive_idxs, train_unlabeled_idxs = np.array(train_idx)[train_positive_idxs], np.array(train_idx)[train_unlabeled_idxs]
    elif args.u_type == 'two_sample':
        train_positive_idxs, train_unlabeled_idxs = p_u_split_two_sample(args, targets[train_idx])
        train_positive_idxs, train_unlabeled_idxs = np.array(train_idx)[train_positive_idxs], np.array(train_idx)[train_unlabeled_idxs]
    
    p_dataset = CreditcardSSL(
        file_path, train_positive_idxs,
        transform=TransformFixMatch_Tabular(),
        target_transform=TransformPTarget(positive_label_list=args.positive_label_list),train=True
    )
    u_dataset = CreditcardSSL(
        file_path, train_unlabeled_idxs,
        transform=TransformFixMatch_Tabular(),
        target_transform=TransformUTarget(positive_label_list=args.positive_label_list),train=True
    )
    test_dataset = CreditcardSSL(
        file_path, test_idx,
        transform=TransformFixMatch_Tabular(),
        target_transform=TransformTestTarget(positive_label_list=args.positive_label_list),train=False
    )
    return p_dataset, u_dataset, test_dataset

def Letter(args):
    class LetterSSL():
        def __init__(self, root, indexs,transform=None, target_transform=None, train=True):
            base_dataset=pd.read_csv(root)
            self.data, self.targets=np.array(base_dataset.iloc[:, 1:]),np.array(base_dataset['letter'].tolist())
            self.transform = transform
            self.target_transform = target_transform
            self.train=train
            if indexs is not None:
                self.data = self.data[indexs]
                self.targets = np.array(self.targets)[indexs]
        def __getitem__(self, index):
            img, target = self.data[index], self.targets[index]
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)

            if self.train==True:
                return index, img, target
            else:
                return img[0],target
        def __len__(self):
            return len(self.data)

    file_path = os.path.join(args.data_dir,'letter.csv')
    base_dataset = pd.read_csv(file_path)
    data, targets=np.array(base_dataset.iloc[:, 1:]),np.array(base_dataset['letter'].tolist())
    total_size=data.shape[0]
    train_size=int(total_size*(1-args.tabular_test_fraction))
    keys = list(range(total_size))
    np.random.RandomState(args.seed).shuffle(keys)
    train_idx = keys[:train_size]
    test_idx = keys[train_size:]
    if args.u_type == 'one_sample':
        train_positive_idxs, train_unlabeled_idxs = p_u_split_one_sample(args, targets[train_idx])
        train_positive_idxs, train_unlabeled_idxs = np.array(train_idx)[train_positive_idxs], np.array(train_idx)[train_unlabeled_idxs]
    elif args.u_type == 'two_sample':
        train_positive_idxs, train_unlabeled_idxs = p_u_split_two_sample(args, targets[train_idx])
        train_positive_idxs, train_unlabeled_idxs = np.array(train_idx)[train_positive_idxs], np.array(train_idx)[train_unlabeled_idxs]
    p_dataset = LetterSSL(
        file_path, train_positive_idxs,
        transform=TransformFixMatch_Tabular(),
        target_transform=TransformPTarget(positive_label_list=args.positive_label_list),train=True
    )
    u_dataset = LetterSSL(
        file_path, train_unlabeled_idxs,
        transform=TransformFixMatch_Tabular(),
        target_transform=TransformUTarget(positive_label_list=args.positive_label_list),train=True
    )
    test_dataset = LetterSSL(
        file_path, test_idx,
        transform=TransformFixMatch_Tabular(),
        target_transform=TransformTestTarget(positive_label_list=args.positive_label_list),train=False
    )
    return p_dataset, u_dataset, test_dataset

# one-sample setting
'''
def p_u_split_one_sample(args, labels):
    #print('positive classes:', args.positive_label_list)
    label_per_class = args.num_p // len(args.positive_label_list)
    labels = np.array(labels)
    positive_idx = []
    # unlabeled data:
    unlabeled_idx = np.array(range(len(labels))) 
    for i in args.positive_label_list:
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        positive_idx.extend(idx)
    positive_idx = np.array(positive_idx)
    unlabeled_idx = np.setdiff1d(unlabeled_idx, positive_idx)
    assert len(positive_idx) == args.num_p
    np.random.shuffle(positive_idx)
    np.random.shuffle(unlabeled_idx)
    return positive_idx, unlabeled_idx
'''
def p_u_split_one_sample(args, labels):
    #print('positive classes:', args.positive_label_list)
    label_per_class = args.num_p // len(args.positive_label_list)
    total_per_class = args.num_p_and_u // args.num_classes_original
    if label_per_class > total_per_class:
        raise NotImplementedError("not enough unlabeled data")
    labels = np.array(labels)
    positive_idx_total = np.array([]).astype(int)
    unlabeled_idx_total = np.array([]).astype(int)
    # unlabeled data:
    #unlabeled_idx = np.array(range(len(labels)))
    # print(label_per_class)
    for i in range(args.num_classes_original):
        if args.dataset=='Letter':
            i=letter = chr(i + ord('A'))
        unlabeled_idx_curr = np.where(labels == i)[0]
        unlabeled_idx_curr = np.random.choice(unlabeled_idx_curr, total_per_class, False)
        if i in args.positive_label_list:
            pos_idx_curr = np.random.choice(unlabeled_idx_curr, label_per_class, False)
            positive_idx_total = np.append(positive_idx_total, pos_idx_curr)
            unlabeled_idx_curr = np.setdiff1d(unlabeled_idx_curr, pos_idx_curr)
        unlabeled_idx_total = np.append(unlabeled_idx_total, unlabeled_idx_curr)
    
    assert len(positive_idx_total) == args.num_p
    assert len(unlabeled_idx_total) == (args.num_p_and_u - args.num_p)
    np.random.shuffle(positive_idx_total)
    np.random.shuffle(unlabeled_idx_total)
    # print(len(positive_idx_total), len(unlabeled_idx_total))
    return positive_idx_total, unlabeled_idx_total

def p_u_split_two_sample(args, labels):
    #print('positive classes:', args.positive_label_list)
    label_per_class = args.num_p // len(args.positive_label_list)
    unlabel_per_class = args.num_u // args.num_classes_original
    labels = np.array(labels)
    positive_idx_total = np.array([]).astype(int)
    unlabeled_idx_total = np.array([]).astype(int)
    if label_per_class + unlabel_per_class > len(labels) // args.num_classes_original:
        raise NotImplementedError("not enough data")
    # unlabeled data:
    #unlabeled_idx = np.array(range(len(labels)))
    for i in range(args.num_classes_original):
        if args.dataset=='Letter':
            i=letter = chr(i + ord('A'))
        all_idx_curr = np.where(labels == i)[0]
        unlabeled_idx_curr = np.random.choice(all_idx_curr, unlabel_per_class, False)  
        if i in args.positive_label_list:
            left_idx_curr = np.setdiff1d(all_idx_curr, unlabeled_idx_curr)
            if label_per_class > len(left_idx_curr):
                pos_idx_curr = np.array(left_idx_curr)  # Take all
            else:
                pos_idx_curr = np.random.choice(left_idx_curr, label_per_class, replace=False)
            positive_idx_total = np.append(positive_idx_total, pos_idx_curr)
        unlabeled_idx_total = np.append(unlabeled_idx_total, unlabeled_idx_curr)
    # assert len(positive_idx_total) == args.num_p
    # assert len(unlabeled_idx_total) == args.num_u
    np.random.shuffle(positive_idx_total)
    np.random.shuffle(unlabeled_idx_total)
    return positive_idx_total, unlabeled_idx_total

class USPSFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=usps_mean, std=usps_std)
    ])
        self.strong = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=usps_mean, std=usps_std)
    ])

    def __call__(self, x):
        weak1 = self.weak(x)
        strong = self.strong(x)
        return weak1, strong
class TransformFixMatch(object):
    def __init__(self, mean, std, img_size=32):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak1 = self.weak(x)
        # weak2 = self.weak(x)
        strong = self.strong(x)
        #strong = self.strong(x)
        #strong1 = self.strong(x)
        #return self.normalize(weak), self.normalize(strong), self.normalize(strong1)
        return self.normalize(weak1),self.normalize(strong)

class IMAGENETTE_Transform(object):
    def __init__(self, mean, std, img_size=64, crop_ratio=0.875):
        # Weak augmentation: random crop + horizontal flip
        self.weak = transforms.Compose([
            transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
            transforms.RandomCrop((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
        ])
        
        # Strong augmentation: weak + RandAugment
        self.strong = transforms.Compose([
            transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
            RandomResizedCropAndInterpolation((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            RandAugment(3, 10),
        ])
        
        # Normalization
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def __call__(self, x):
        weak1 = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak1), self.normalize(strong)

class TransformFixMatch_Tabular(object):
    def __init__(self):
        pass
    def __call__(self, x):
        x=torch.tensor(x).float()
        return x, x

class TransformPTarget(object):
    def __init__(self, positive_label_list):
        self.p_transform = lambda x: 1 if x in positive_label_list else 0

    def __call__(self, x):
        x = self.p_transform(x)
        if x == 0:
            raise NotImplementedError("Error when generating positive data")

        return x

class TransformUTarget(object):
    def __init__(self, positive_label_list):
        self.u_transform = lambda x: 0
        self.true_transform = lambda x: 1 if x in positive_label_list else 0

    def __call__(self, x):
        x_unlabeled = self.u_transform(x)
        x_true = self.true_transform(x)
        return x_unlabeled, x_true

class TransformTestTarget(object):
    def __init__(self, positive_label_list):
        self.true_transform = lambda x: 1 if x in positive_label_list else 0

    def __call__(self, x):
        x = self.true_transform(x)
        return x
