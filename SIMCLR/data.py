import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler

class GaussianBlur(object):
    """Implements Gaussian blur as described in the SimCLR paper."""
    def __init__(self, kernel_size, min_val=0.1, max_val=2.0):
        self.min_val = min_val
        self.max_val = max_val
        self.kernel_size = max(3, int(kernel_size))
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1

    def __call__(self, sample):
        sample = np.array(sample)
        if np.random.random_sample() < 0.5:
            sigma = (self.max_val - self.min_val) * np.random.random_sample() + self.min_val
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)
        return sample

class SimCLRDataTransform:
    """Apply data transformations to create two augmented views."""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        xi = self.transform(sample)
        xj = self.transform(sample)
        return xi, xj

class DataSetWrapper:
    """Wrapper for loading datasets and applying transformations."""
    def __init__(self, batch_size, num_workers, valid_size, input_shape, strength):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.input_shape = input_shape
        self.strength = strength

    def get_data_loaders(self):
        data_augment = self._simclr_transform()
        train_dataset = datasets.ImageFolder(
            root='.data/', #Provide data path
            transform=SimCLRDataTransform(data_augment)
        )
        return self._get_train_validation_data_loaders(train_dataset)

    def _simclr_transform(self):
        s = self.strength
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        gaussian_blur = GaussianBlur(kernel_size=int(0.1 * 448))

        return transforms.Compose([
            transforms.Resize(448),
            transforms.CenterCrop(448),
            transforms.RandomApply([color_jitter], p=0.4),
            transforms.RandomGrayscale(p=0.2),
            gaussian_blur,
            transforms.ToTensor(),
        ])

    def _get_train_validation_data_loaders(self, train_dataset):
        num_train = len(train_dataset)
        indices = list(range(num_train))
        np.random.shuffle(indices)
        split = int(np.floor(self.valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=SubsetRandomSampler(train_idx),
            num_workers=self.num_workers, drop_last=True
        )
        valid_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=SubsetRandomSampler(valid_idx),
            num_workers=self.num_workers, drop_last=True
        )
        return train_loader, valid_loader
