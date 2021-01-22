import numpy as np
from torch.utils.data import DataLoader
import torchvision.datasets as tvd
import torchvision.transforms as tvt


__all__ = ["CIFAR10", "CIFAR100", "CIFAR100Coarse"]


class _BaseCIFAR:
    # mean = [0.4914, 0.4822, 0.4465]  # From own computations
    # stddev = [0.2470, 0.2435, 0.2616]  # other values are .2023, .1994, .2010
    mean = [0.485, 0.456, 0.406]  # From akaresnet
    stddev = [0.229, 0.224, 0.225]

    def __init__(self, data_folder, pin_memory=False):
        self.data_folder = data_folder
        self.pin_memory = pin_memory

    def get_train_loader(self, batch_size, shuffle=True, num_workers=0,
                         use_random_crops=True, use_hflips=True,
                         use_color_jitter=False):
        transforms = []
        if use_hflips:
            transforms.append(tvt.RandomHorizontalFlip())
        if use_random_crops:
            transforms.append(tvt.RandomCrop(32, padding=4))
        if use_color_jitter:
            transforms.append(tvt.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05))

        transforms.append(tvt.ToTensor())
        transforms.append(tvt.Normalize(_BaseCIFAR.mean, _BaseCIFAR.stddev))
        trainset = self._ds(root=self.data_folder, download=True,
                            train=True, transform=tvt.Compose(transforms))
        trainloader = DataLoader(dataset=trainset, batch_size=batch_size,
                                 shuffle=shuffle, num_workers=num_workers,
                                 pin_memory=self.pin_memory)
        return trainloader

    def get_test_loader(self, batch_size, num_workers=0):
        transforms = [tvt.ToTensor()]
        transforms.append(tvt.Normalize(_BaseCIFAR.mean, _BaseCIFAR.stddev))
        testset = self._ds(root=self.data_folder, download=False,
                           train=False, transform=tvt.Compose(transforms))
        testloader = DataLoader(dataset=testset, batch_size=batch_size,
                                shuffle=False, num_workers=num_workers,
                                pin_memory=self.pin_memory)
        return testloader


class CIFAR10(_BaseCIFAR):
    labels = [
        'plane', 'car', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]

    def __init__(self, data_folder, pin_memory=False):
        super().__init__(data_folder, pin_memory)
        self._ds = tvd.CIFAR10

    @staticmethod
    def get_labels():
        return CIFAR10.labels

    @staticmethod
    def get_num_classes():
        return len(CIFAR10.labels)


class CIFAR100(_BaseCIFAR):
    fine_labels = [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver',
        'bed', 'bee', 'beetle', 'bicycle', 'bottle',
        'bowl', 'boy', 'bridge', 'bus', 'butterfly',
        'camel', 'can', 'castle', 'caterpillar', 'cattle',
        'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach',
        'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox',
        'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
        'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard',
        'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
        'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid',
        'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
        'plain', 'plate', 'poppy', 'porcupine', 'possum',
        'rabbit', 'raccoon', 'ray', 'road', 'rocket',
        'rose', 'sea', 'seal', 'shark', 'shrew',
        'skunk', 'skyscraper', 'snail', 'snake', 'spider',
        'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
        'tank', 'telephone', 'television', 'tiger', 'tractor',
        'train', 'trout', 'tulip', 'turtle', 'wardrobe',
        'whale', 'willow_tree', 'wolf', 'woman', 'worm'
    ]

    coarse_labels = [
        'aquatic_mammals', 'fish',
        'flowers', 'food_containers',
        'fruit_and_vegetables', 'household_electrical_devices',
        'household_furniture', 'insects',
        'large_carnivores', 'large_man-made_outdoor_things',
        'large_natural_outdoor_scenes', 'large_omnivores_and_herbivores',
        'medium_mammals', 'non-insect_invertebrates',
        'people', 'reptiles',
        'small_mammals', 'trees',
        'vehicles_1', 'vehicles_2'
    ]

    fine_to_coarse = [
        4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
        3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
        6, 11,  5, 10,  7,  6, 13, 15,  3, 15,
        0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
        5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
        16, 4, 17,  4,  2,  0, 17,  4, 18, 17,
        10, 3,  2, 12, 12, 16, 12,  1,  9, 19,
        2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
        16, 19, 2,  4,  6, 19,  5,  5,  8, 19,
        18, 1,  2, 15,  6,  0, 17,  8, 14, 13
    ]

    def __init__(self, data_folder, pin_memory=False):
        super().__init__(data_folder, pin_memory)
        self._ds = tvd.CIFAR100

    @staticmethod
    def get_labels():
        return CIFAR100.fine_labels

    @staticmethod
    def get_num_classes():
        return len(CIFAR100.fine_labels)


class CIFAR100Coarse(CIFAR100):
    fine_to_coarse_np = np.array(CIFAR100.fine_to_coarse)

    class CIFAR100CoarseDataset(tvd.CIFAR100):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.targets = CIFAR100Coarse.fine_to_coarse_np[self.targets]

    def __init__(self, data_folder, pin_memory=False):
        super().__init__(data_folder, pin_memory)
        self._ds = CIFAR100Coarse.CIFAR100CoarseDataset

    @staticmethod
    def get_labels():
        return CIFAR100.coarse_labels

    @staticmethod
    def get_num_classes():
        return len(CIFAR100.coarse_labels)
