# Code for "[HAQ: Hardware-Aware Automated Quantization with Mixed Precision"
# Kuan Wang*, Zhijian Liu*, Yujun Lin*, Ji Lin, Song Han
# {kuanwang, zhijian, yujunlin, jilin, songhan}@mit.edu
#
# ---------------------------------------------------------------------------
# Project Omnia addendum (experimental_checklist.md Section 1: "Prepare the
# Standard Benchmarks") -- adds 'cifar100' and 'imagenet_mini' dataset_name
# options below, alongside the original imagenet/imagenet100/imagenet10
# branches. Both are needed to show mixed-precision assignment doesn't
# cause catastrophic accuracy loss on standard classification tasks,
# independent of the full ImageNet-1k results.
#
# Also adds get_calibration_loader() (bug fix from code review) -- see its
# docstring below. entropy_quantize.py previously built its calibration
# subset as `Subset(train_loader.dataset, list(range(calib_size)))`, which
# for ImageFolder-backed datasets pulls entirely from whichever class
# sorts first alphabetically (ImageNet-1k has ~1,300 images/class, so
# range(100) never left the first class), using train-time-augmented
# (RandomResizedCrop/RandomHorizontalFlip) images to boot.
# ---------------------------------------------------------------------------

import os
import numpy as np

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler


def get_dataset(dataset_name, batch_size, n_worker, data_root='data/imagenet', for_inception=False):
    print('==> Preparing data..')
    if dataset_name == 'imagenet':
        traindir = os.path.join(data_root, 'train')
        valdir = os.path.join(data_root, 'val')
        assert os.path.exists(traindir), traindir + ' not found'
        assert os.path.exists(valdir), valdir + ' not found'
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        input_size = 299 if for_inception else 224

        train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                traindir, transforms.Compose([
                    transforms.RandomResizedCrop(input_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])),
            batch_size=batch_size, shuffle=True,
            num_workers=n_worker, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(int(input_size / 0.875)),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=batch_size, shuffle=False,
            num_workers=n_worker, pin_memory=True)

        n_class = 1000
    elif dataset_name == 'imagenet100':
        traindir = os.path.join(data_root, 'train')
        valdir = os.path.join(data_root, 'val')
        assert os.path.exists(traindir), traindir + ' not found'
        assert os.path.exists(valdir), valdir + ' not found'
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        input_size = 299 if for_inception else 224

        train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                traindir, transforms.Compose([
                    transforms.RandomResizedCrop(input_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])),
            batch_size=batch_size, shuffle=True,
            num_workers=n_worker, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(int(input_size / 0.875)),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=batch_size, shuffle=False,
            num_workers=n_worker, pin_memory=True)

        n_class = 100
    elif dataset_name == 'imagenet10':
        traindir = os.path.join(data_root, 'train')
        valdir = os.path.join(data_root, 'val')
        assert os.path.exists(traindir), traindir + ' not found'
        assert os.path.exists(valdir), valdir + ' not found'
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        input_size = 299 if for_inception else 224

        train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                traindir, transforms.Compose([
                    transforms.RandomResizedCrop(input_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])),
            batch_size=batch_size, shuffle=True,
            num_workers=n_worker, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(int(input_size / 0.875)),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=batch_size, shuffle=False,
            num_workers=n_worker, pin_memory=True)

        n_class = 10
    elif dataset_name == 'cifar100':
        # CIFAR-100 is natively supported by torchvision.datasets, so we
        # download/cache it under data_root rather than expecting a
        # pre-existing train/val folder layout like the ImageNet variants
        # above. Compact vision models here (MobileNet*/ResNet/
        # EfficientNet-Lite) expect a 224x224 input, so every image is
        # resized up from CIFAR-100's native 32x32 -- this only needs to
        # be "big enough to run the same architecture / quantization
        # pipeline unmodified", not photorealistic, since the point of
        # this benchmark is to check the entropy-driven bit-assignment
        # doesn't cause catastrophic accuracy loss on a *second*,
        # architecturally-unrelated classification task.
        input_size = 224
        # CIFAR-100's own per-channel mean/std (not ImageNet's), since the
        # two datasets have different pixel statistics.
        normalize = transforms.Normalize(mean=[0.5071, 0.4865, 0.4409],
                                         std=[0.2673, 0.2564, 0.2762])

        train_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        test_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            normalize,
        ])

        trainset = datasets.CIFAR100(root=data_root, train=True, download=True,
                                     transform=train_transform)
        valset = datasets.CIFAR100(root=data_root, train=False, download=True,
                                   transform=test_transform)

        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=n_worker, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            valset, batch_size=batch_size, shuffle=False,
            num_workers=n_worker, pin_memory=True)

        n_class = 100
    elif dataset_name == 'imagenet_mini':
        # Kaggle's "ImageNet-Mini" (a ~34-class-per-split subset of
        # ImageNet-1k) ships already structured as train/ and val/ class
        # folders -- https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000
        # -- so, per the brief, this just points a standard ImageFolder-
        # backed DataLoader at data_root the same way the full-size
        # 'imagenet' branch above does; the only difference is the (much
        # smaller) number of images/classes actually present on disk.
        traindir = os.path.join(data_root, 'train')
        valdir = os.path.join(data_root, 'val')
        assert os.path.exists(traindir), (
            traindir + ' not found -- download the Kaggle ImageNet-Mini '
            'dataset and point --dataset_root at the folder containing '
            'train/ and val/ subdirectories')
        assert os.path.exists(valdir), valdir + ' not found'
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        input_size = 299 if for_inception else 224

        train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                traindir, transforms.Compose([
                    transforms.RandomResizedCrop(input_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])),
            batch_size=batch_size, shuffle=True,
            num_workers=n_worker, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(int(input_size / 0.875)),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=batch_size, shuffle=False,
            num_workers=n_worker, pin_memory=True)

        # ImageNet-Mini's class count depends on which Kaggle version you
        # grab (the common one is a ~1000-class subset with fewer images
        # per class, not fewer classes) -- infer it from the folder
        # structure rather than hardcoding, so this works across variants.
        n_class = len(train_loader.dataset.classes)
    else:
        # Add customized data here
        raise NotImplementedError
    return train_loader, val_loader, n_class


def get_split_train_dataset(dataset_name, batch_size, n_worker, val_size, train_size=None, random_seed=1,
                            data_root='data/imagenet', for_inception=False, shuffle=True):
    if shuffle:
        index_sampler = SubsetRandomSampler
    else:
        # use the same order
        class SubsetSequentialSampler(SubsetRandomSampler):
            def __iter__(self):
                return (self.indices[i] for i in torch.arange(len(self.indices)).int())
        index_sampler = SubsetSequentialSampler

    print('==> Preparing data..')
    if dataset_name == 'imagenet':

        traindir = os.path.join(data_root, 'train')
        valdir = os.path.join(data_root, 'val')
        assert os.path.exists(traindir), traindir + ' not found'
        assert os.path.exists(valdir), valdir + ' not found'
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        input_size = 299 if for_inception else 224
        train_transform = transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        test_transform = transforms.Compose([
                transforms.Resize(int(input_size/0.875)),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                normalize,
            ])

        trainset = datasets.ImageFolder(traindir, train_transform)
        valset = datasets.ImageFolder(traindir, test_transform)

        n_train = len(trainset)
        indices = list(range(n_train))
        # shuffle the indices
        np.random.seed(random_seed)
        np.random.shuffle(indices)
        assert val_size < n_train, 'val size should less than n_train'
        train_idx, val_idx = indices[val_size:], indices[:val_size]
        if train_size:
            train_idx = train_idx[:train_size]
        print('Data: train: {}, val: {}'.format(len(train_idx), len(val_idx)))

        train_sampler = index_sampler(train_idx)
        val_sampler = index_sampler(val_idx)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=train_sampler,
                                                   num_workers=n_worker, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, sampler=val_sampler,
                                                 num_workers=n_worker, pin_memory=True)
        n_class = 1000
    elif dataset_name == 'imagenet100':

        traindir = os.path.join(data_root, 'train')
        valdir = os.path.join(data_root, 'val')
        assert os.path.exists(traindir), traindir + ' not found'
        assert os.path.exists(valdir), valdir + ' not found'
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        input_size = 299 if for_inception else 224
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        test_transform = transforms.Compose([
            transforms.Resize(int(input_size/0.875)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ])

        trainset = datasets.ImageFolder(traindir, train_transform)
        valset = datasets.ImageFolder(traindir, test_transform)

        n_train = len(trainset)
        indices = list(range(n_train))
        # shuffle the indices
        np.random.seed(random_seed)
        np.random.shuffle(indices)
        assert val_size < n_train, 'val size should less than n_train'
        train_idx, val_idx = indices[val_size:], indices[:val_size]
        if train_size:
            train_idx = train_idx[:train_size]
        print('Data: train: {}, val: {}'.format(len(train_idx), len(val_idx)))

        train_sampler = index_sampler(train_idx)
        val_sampler = index_sampler(val_idx)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=train_sampler,
                                                   num_workers=n_worker, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, sampler=val_sampler,
                                                 num_workers=n_worker, pin_memory=True)
        n_class = 100
    else:
        raise NotImplementedError

    return train_loader, val_loader, n_class


def _get_normalize_and_size(dataset_name, for_inception=False):
    """
    Project Omnia addendum: factors the per-dataset Normalize()/input-size
    pair -- previously copy-pasted across every branch of get_dataset()
    above -- into one place, so get_calibration_loader() below reuses
    exactly the same normalization val_loader uses instead of risking it
    drifting out of sync over time.
    """
    input_size = 299 if for_inception else 224
    if dataset_name in ('imagenet', 'imagenet100', 'imagenet10', 'imagenet_mini'):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    elif dataset_name == 'cifar100':
        normalize = transforms.Normalize(mean=[0.5071, 0.4865, 0.4409],
                                         std=[0.2673, 0.2564, 0.2762])
        input_size = 224
    else:
        raise NotImplementedError(
            "_get_normalize_and_size does not know dataset_name=%r" % dataset_name)
    return normalize, input_size


def get_calibration_loader(dataset_name, data_root, calib_size, batch_size,
                           n_worker=4, seed=234, for_inception=False):
    """
    Project Omnia addendum (bug fix from code review) -- builds the "tiny
    calibration subset (100 sample images)" stream entropy_quantize.py
    uses for both the entropy pass and quantize_utils.calibrate()'s
    S_l/Z_l estimation.

    Two things this fixes relative to the original
    `Subset(train_loader.dataset, list(range(calib_size)))` approach:

      1. Representativeness: for the imagenet*/imagenet_mini branches,
         `train_loader.dataset` is a torchvision ImageFolder, which lists
         samples grouped by class in alphabetical order. `range(100)` on
         a dataset with ~1,300 images/class (ImageNet-1k) pulled entirely
         from whichever class sorts first -- not a representative sample
         of "typical" activations. This function instead draws
         `calib_size` indices *uniformly at random* (seeded, for
         reproducibility) from the entire training split.
      2. Preprocessing: train_loader's dataset uses training-time
         augmentation (RandomResizedCrop, RandomHorizontalFlip), so the
         same calibration image would contribute different, randomly
         perturbed pixels on every run. This function uses the same
         deterministic Resize+CenterCrop pipeline as val_loader, so
         calibration statistics are stable and reproducible for a given
         seed.

    Returns a DataLoader (shuffle=False -- the *selection* of indices is
    already randomized, so loader order doesn't matter) of `calib_size`
    images with batch_size=`batch_size`.
    """
    normalize, input_size = _get_normalize_and_size(dataset_name, for_inception)
    eval_transform = transforms.Compose([
        transforms.Resize(int(input_size / 0.875)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        normalize,
    ])

    if dataset_name in ('imagenet', 'imagenet100', 'imagenet10', 'imagenet_mini'):
        traindir = os.path.join(data_root, 'train')
        assert os.path.exists(traindir), traindir + ' not found'
        full_dataset = datasets.ImageFolder(traindir, eval_transform)
    elif dataset_name == 'cifar100':
        full_dataset = datasets.CIFAR100(root=data_root, train=True, download=True,
                                         transform=eval_transform)
    else:
        raise NotImplementedError(
            "get_calibration_loader does not support dataset_name=%r; add a "
            "branch here the same way get_dataset() has one." % dataset_name)

    n_total = len(full_dataset)
    if calib_size > n_total:
        raise ValueError(
            'calib_size=%d exceeds the %s training set size (%d)' %
            (calib_size, dataset_name, n_total))

    rng = np.random.RandomState(seed)
    calib_indices = rng.choice(n_total, size=calib_size, replace=False).tolist()
    calib_subset = torch.utils.data.Subset(full_dataset, calib_indices)

    return torch.utils.data.DataLoader(
        calib_subset, batch_size=batch_size, shuffle=False,
        num_workers=n_worker, pin_memory=True)
