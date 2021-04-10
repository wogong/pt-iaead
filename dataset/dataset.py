import torch.utils.data as data
import torchvision.transforms as transforms


class trainset_pytorch(data.Dataset):
    def __init__(self, train_data, train_labels, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

        self.train_data = train_data  # ndarray
        self.train_labels = train_labels

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.train_data[index], self.train_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        # img = Image.fromarray(img)  # used if the img is [H, W, C] and the dtype is uint8

        if self.transform is not None:
            # img_int = np.uint8(denormalize_minus1_1(img))
            # img = Image.fromarray(img_int)
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.train_data)


class trainset_rat(data.Dataset):
    def __init__(self, train_data, train_labels, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

        self.train_data = train_data  # ndarray
        self.train_labels = train_labels

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.train_data[index], self.train_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        # img = Image.fromarray(img)  # used if the img is [H, W, C] and the dtype is uint8

        RAT = transforms.Compose([transforms.ToPILImage(),
                                  transforms.RandomAffine(degrees=(-10, 10), translate=(0.4, 0.4), scale=(0.8, 1.2), shear=(-0.3, 0.3)),
                                  transforms.ToTensor(),])

        if self.transform is not None:
            # img_int = np.uint8(denormalize_minus1_1(img))
            # img = Image.fromarray(img_int)
            img = self.transform(img)
            img_rat = RAT(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, img_rat, target, index

    def __len__(self):
        return len(self.train_data)
