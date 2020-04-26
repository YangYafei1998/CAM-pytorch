import torch
import numpy as np
import imgaug.augmenters as iaa

class DataAugmentation():
    def __init__(self):
        self.seq = iaa.Sequential([
            iaa.Sometimes(
                0.3,
                iaa.OneOf(
                    [iaa.GaussianBlur(sigma=(0, 0.05)),
                    iaa.MotionBlur(),]
                )
            ),
            iaa.Sometimes(
                0.3,
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05), per_channel=0.5),
            ),
            iaa.Cutout(nb_iterations=(0, 3), size=0.2, squared=False)
        ], random_order=True)

    def __call__(self, tensor_imgs):
        ## convert to np and then augment
        np_imgs = self.seq(images=np.asarray(tensor_imgs))
        ## convert to tensor
        return torch.from_numpy(np_imgs)


