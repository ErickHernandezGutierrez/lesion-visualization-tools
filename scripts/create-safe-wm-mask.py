#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_erosion


def create_ball_structuring_element(radius):
    """
    Create a spherical structuring element with the given radius.
    """
    diameter = 2 * radius + 1
    center = radius
    coords = np.ogrid[:diameter, :diameter, :diameter]
    distance = sum((c - center) ** 2 for c in coords)
    return distance <= radius ** 2


def main():
    parser = argparse.ArgumentParser(
        description='Erode a white matter mask to produce a safe WM mask.'
    )
    parser.add_argument('wm_mask', help='Path to the input white matter mask (.nii.gz).')
    parser.add_argument('output', help='Path for the output safe WM mask (.nii.gz).')
    parser.add_argument('--erosion_radius', type=int, default=1,
                        help='Radius (in voxels) of the spherical structuring element used for erosion. Default is 1.')
    args = parser.parse_args()

    img = nib.load(args.wm_mask)
    wm_mask = img.get_fdata().astype(np.uint8)

    struct = create_ball_structuring_element(args.erosion_radius)
    safe_wm_mask = binary_erosion(wm_mask, structure=struct).astype(np.uint8)

    out_img = nib.Nifti1Image(safe_wm_mask, img.affine, img.header)
    nib.save(out_img, args.output)
    print(f"Safe WM mask saved to {args.output}")


if __name__ == '__main__':
    main()
