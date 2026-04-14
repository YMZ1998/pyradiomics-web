# -- coding: utf-8 --
import SimpleITK as sitk
import os
import numpy as np
import glob


def window_transform(ct_array, windowWidth, windowCenter, normal=True):
    """
    return: trucated image according to window center and window width
    and normalized to [0,1]
    """
    minWindow = float(windowCenter) - 0.5 * float(windowWidth)
    newimg = (ct_array - minWindow) / float(windowWidth)
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    if not normal:
        newimg = (newimg * 255).astype('float32')
    return newimg


paths = glob.glob('./MyData/*/*/*.nii.gz')
# print(paths)
for path in paths:
    if 'seg' not in path:
        print('文件: ' + path)
        print("-" * 100)
        image = sitk.ReadImage(path, sitk.sitkFloat32)
        array = sitk.GetArrayFromImage(image)
        array = window_transform(array, 500, 100)
        new_image = sitk.GetImageFromArray(array)
        new_image.SetDirection(image.GetDirection())
        new_image.SetOrigin([0, 0, 0])
        new_image.SetSpacing([1, 1, 1])
        result_path = path.replace('MyData', 'DATA')
        sitk.WriteImage(new_image, result_path)
    else:
        print('文件: ' + path)
        print("-" * 100)
        image = sitk.ReadImage(path, sitk.sitkUInt8)
        array = sitk.GetArrayFromImage(image)
        new_image = sitk.GetImageFromArray(array)
        new_image.SetDirection(image.GetDirection())
        new_image.SetOrigin([0, 0, 0])
        new_image.SetSpacing([1, 1, 1])
        result_path = path.replace('MyData', 'DATA')
        sitk.WriteImage(new_image, result_path)
