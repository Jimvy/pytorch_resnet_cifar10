from torch.utils.data import DataLoader
import torchvision.datasets as tvd
import torchvision.transforms as tvt


class CIFAR10:
    # mean = [0.4914, 0.4822, 0.4465]  # From own computations
    # stddev = [0.2470, 0.2435, 0.2616]  # other values are .2023, .1994, .2010
    mean = [0.485, 0.456, 0.406]  # From akaresnet
    stddev = [0.229, 0.224, 0.225]
    labels = [
        'plane', 'car', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]

    def __init__(self, data_folder, pin_memory=False):
        self.data_folder = data_folder
        self.pin_memory = pin_memory

    def get_train_loader(self, batch_size, shuffle=True, num_workers=0,
                         use_random_crops=True, use_hflips=True):
        transforms = []
        if use_hflips:
            transforms.append(tvt.RandomHorizontalFlip())
        if use_random_crops:
            transforms.append(tvt.RandomCrop(32, padding=4))
        transforms.append(tvt.ToTensor())
        transforms.append(tvt.Normalize(CIFAR10.mean, CIFAR10.stddev))
        trainset = tvd.CIFAR10(root=self.data_folder, download=True,
                               train=True, transform=tvt.Compose(transforms))
        trainloader = DataLoader(dataset=trainset, batch_size=batch_size,
                                 shuffle=shuffle, num_workers=num_workers,
                                 pin_memory=self.pin_memory)
        return trainloader

    def get_test_loader(self, batch_size, num_workers=0):
        transforms = [tvt.ToTensor()]
        transforms.append(tvt.Normalize(CIFAR10.mean, CIFAR10.stddev))
        testset = tvd.CIFAR10(root=self.data_folder, download=False,
                              train=False, transform=tvt.Compose(transforms))
        testloader = DataLoader(dataset=testset, batch_size=batch_size,
                                shuffle=False, num_workers=num_workers,
                                pin_memory=self.pin_memory)
        return testloader

    @staticmethod
    def get_labels():
        return CIFAR10.labels

    @staticmethod
    def get_num_classes():
        return len(CIFAR10.labels)


class CIFAR100:
    mean = [0.485, 0.456, 0.406]
    stddev = [0.229, 0.224, 0.225]
