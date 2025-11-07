from torchvision.transforms import functional

import numpy as np


def _check_image_dimensions(image):
    if len(image.shape) < 3:
        image = np.expand_dims(image, axis=-1)
    return image


class RandomHorizontalFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        image_a, image_b, label = sample

        rv = np.random.rand()
        if rv > self.p:
            image_a = functional.hflip(image_a)
            image_b = functional.hflip(image_b)
            label = functional.hflip(label)
            
        return image_a, image_b, label
    

class RandomRotate(object):

    def __init__(self, interpolation=functional.InterpolationMode.NEAREST, expand=False, center=None, fill=0):
        self.interpolation = interpolation
        self.expand = expand
        self.center = center
        self.fill = fill

    def __call__(self, sample):
        image_a, image_b, label = sample

        rv = np.random.randint(low=-15, high=15)
        degree = rv
        
        c, *_ = functional.get_dimensions(image_a)
        image_a = functional.rotate(image_a, degree, self.interpolation, self.expand, self.center, fill=[float(self.fill) * c])
        
        c, *_ = functional.get_dimensions(image_b)
        image_b = functional.rotate(image_b, degree, self.interpolation, self.expand, self.center, fill=[float(self.fill) * c])
        
        c, *_ = functional.get_dimensions(label)
        label = functional.rotate(label, degree, self.interpolation, self.expand, self.center, fill=[float(self.fill) * c])

        return image_a, image_b, label
    

class RandomExchange(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        image_a, image_b, label = sample

        rv = np.random.rand()
        return (image_b, image_a, label) if rv > self.p else sample


class Normalize(object):

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, sample):
        image_a, image_b, label = sample
        image_a = functional.normalize(image_a, self.mean, self.std, self.inplace)
        image_b = functional.normalize(image_b, self.mean, self.std, self.inplace)
        
        return image_a, image_b, label
    

class ToTensor(object):

    def __init__(self):
        pass

    def __call__(self, sample):
        image_a, image_b, label = sample
        image_a = _check_image_dimensions(image_a)
        image_b = _check_image_dimensions(image_b)
        label = _check_image_dimensions(label)

        # image_a = image_a.transpose((2, 0, 1))
        # image_b = image_b.transpose((2, 0, 1))
        # label = label.transpose((2, 0, 1))

        image_a = image_a.astype(np.float32) / 255
        image_b = image_b.astype(np.float32) / 255
        label = label.astype(np.float32) / 255

        image_a = functional.to_tensor(image_a)
        image_b = functional.to_tensor(image_b)
        label = functional.to_tensor(label)
        
        return image_a, image_b, label
