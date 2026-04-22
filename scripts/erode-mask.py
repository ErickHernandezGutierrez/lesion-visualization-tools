import argparse
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_erosion

def erode_mask(input_path, output_path, iterations=1):
    """
    Load a binary mask in .nii.gz format, erode it, and save the result.
    
    Parameters:
    - input_path: Path to the input .nii.gz file
    - output_path: Path to save the eroded mask
    - iterations: Number of erosion iterations (default: 1)
    """
    # Load the NIfTI file
    img = nib.load(input_path)
    data = img.get_fdata().astype(np.uint8)
    affine = img.affine
    header = img.header
    
    # Binarize the data (in case it's not strictly 0/1)
    binary_data = (data > 0).astype(np.uint8)
    
    # Perform binary erosion
    eroded_data = binary_erosion(binary_data, iterations=iterations)
    
    # Convert back to uint8
    eroded_data = eroded_data.astype(np.uint8)
    
    # Save the eroded mask
    eroded_img = nib.Nifti1Image(eroded_data, affine, header)
    nib.save(eroded_img, output_path)
    print(f"Eroded mask saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Erode binary mask')
    parser.add_argument('in_mask', help='Path to the input binary mask')
    parser.add_argument('out_mask', help='Path to the output binary mask')
    parser.add_argument('--n_iterations', type=int, default=1, help='Number of iterations. Default 1')
    args = parser.parse_args()

    erode_mask(args.in_mask, args.out_mask, iterations=args.n_iterations)
