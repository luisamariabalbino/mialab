"""The pre-processing module contains classes for image pre-processing.

Image pre-processing aims to improve the image quality (image intensities)
for subsequent pipeline steps.
"""

import numpy as np

import pymia.filtering.filter as pymia_fltr
import SimpleITK as sitk


class ImageNormalization(pymia_fltr.Filter):
    """Represents a normalization filter."""

    def __init__(self):
        """Initializes a new instance of the ImageNormalization class."""
        super().__init__()

    def execute(
            self,
            image: sitk.Image,
            params: pymia_fltr.FilterParams = None
    ) -> sitk.Image:
        """Executes a normalization on an image."""

        img_arr = sitk.GetArrayFromImage(image).astype(np.float32)

        mean = np.mean(img_arr)
        std = np.std(img_arr)

        if std > 0:
            img_arr = (img_arr - mean) / std

        img_out = sitk.GetImageFromArray(img_arr)
        img_out.CopyInformation(image)

        return img_out

    def __str__(self):
        return 'ImageNormalization:\n'.format(self=self)


class SkullStrippingParameters(pymia_fltr.FilterParams):
    """Skull-stripping parameters."""

    def __init__(self, img_mask: sitk.Image):
        """Initializes a new instance of the SkullStrippingParameters."""
        self.img_mask = img_mask


class SkullStripping(pymia_fltr.Filter):
    """Represents a skull-stripping filter."""

    def __init__(self):
        """Initializes a new instance of the SkullStripping class."""
        super().__init__()

    def execute(
            self,
            image: sitk.Image,
            params: SkullStrippingParameters = None
    ) -> sitk.Image:
        """Executes skull stripping using a brain mask."""

        mask = params.img_mask

        image = sitk.Mask(image, sitk.Cast(mask, sitk.sitkUInt8))

        return image

    def __str__(self):
        return 'SkullStripping:\n'.format(self=self)


class ImageRegistrationParameters(pymia_fltr.FilterParams):
    """Image registration parameters."""

    def __init__(
            self,
            atlas: sitk.Image,
            transformation: sitk.Transform,
            is_ground_truth: bool = False
    ):
        """Initializes a new instance of the ImageRegistrationParameters."""

        self.atlas = atlas
        self.transformation = transformation
        self.is_ground_truth = is_ground_truth


class ImageRegistration(pymia_fltr.Filter):
    """Represents a registration filter."""

    def __init__(self):
        """Initializes a new instance of the ImageRegistration class."""
        super().__init__()

    def execute(
            self,
            image: sitk.Image,
            params: ImageRegistrationParameters = None
    ) -> sitk.Image:
        """Registers an image to the atlas using the provided transform."""

        atlas = params.atlas
        transform = params.transformation
        is_ground_truth = params.is_ground_truth

        if is_ground_truth:
            interpolator = sitk.sitkNearestNeighbor
        else:
            interpolator = sitk.sitkLinear

        registered_image = sitk.Resample(
            image,
            atlas,
            transform,
            interpolator,
            0.0,
            image.GetPixelID()
        )

        return registered_image

    def __str__(self):
        return 'ImageRegistration:\n'.format(self=self)