from torchvision.transforms import RandomAffine, ToPILImage, ColorJitter, RandomHorizontalFlip, ToTensor
import torchvision.transforms.functional as F
import numpy as np

class SyncToPILImage():

    def __init__(self):
        self.tpi = ToPILImage()

    def __call__(self, img, mask):
        return self.tpi(img), self.tpi(mask)

class SyncToTensor():
    def __init__(self):
        self.tt = ToTensor()

    def __call__(self, img, mask):
        return self.tt(img), self.tt(mask)


class SyncRandomAffine():
    def __init__(self, degrees, translate=None, scale=None, shear=None):
        self.trans = RandomAffine(degrees, translate, scale, shear)

    def __call__(self, img, mask):
        ret = self.trans.get_params(self.trans.degrees, self.trans.translate, self.trans.scale, self.trans.shear,
                                    img.size)

        img_prime = F.affine(img, *ret, resample=self.trans.resample, fillcolor=self.trans.fillcolor)
        mask_prime = F.affine(mask, *ret, resample=self.trans.resample, fillcolor=self.trans.fillcolor)

        return img_prime, mask_prime


class SyncColorJitter():
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.jitter = ColorJitter(brightness=brightness, contrast=contrast,
                                  saturation=saturation)  # hue - нету, foolproof

    def __call__(self, img, mask):
        img_prime = self.jitter(img)
        mask_prime = mask

        return img_prime, mask_prime


class SyncRandomHorizontalFlip():
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask):
        if np.random.rand() < self.p:
            img_prime = F.hflip(img)
            mask_prime = F.hflip(mask)

            return img_prime, mask_prime

        return img, mask


class SyncRandomVerticalFlip():
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask):
        if np.random.rand() < self.p:
            img_prime = F.vflip(img)
            mask_prime = F.vflip(mask)

            return img_prime, mask_prime

        return img, mask


class SyncCompose():
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for transform in self.transforms:
            img, mask = transform(img, mask)

        return img, mask
