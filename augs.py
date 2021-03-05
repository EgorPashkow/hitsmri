# RandomAffine
# deg = 360
# translate = 0.2, 0.2
# shear - не-а
# scale 0.8-1.2

from torchvision.transforms import RandomAffine, ToPILImage
import torchvision.transforms.functional as F

class SyncRandomAffine():
    def __init__(self, degrees, translate=None, scale=None, shear=None):
        self.trans = RandomAffine(degrees, translate, scale, shear)

    def __call__(self, img, mask):
        ret = self.trans.get_params(self.trans.degrees, self.trans.translate, self.trans.scale, self.trans.shear, img.size)

        img_prime = F.affine(img, *ret, resample=self.trans.resample, fillcolor=self.trans.fillcolor)
        mask_prime = F.affine(mask, *ret, resample=self.trans.resample, fillcolor=self.trans.fillcolor)

        return img_prime, mask_prime


class SyncToPILImage():
    def __init__(self):
        self.tpi = ToPILImage()

    def __call__(self, img, mask):
        return self.tpi(img), self.tpi(mask)