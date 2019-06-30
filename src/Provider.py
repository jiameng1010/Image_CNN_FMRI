import nibabel as nib
import numpy as np
import skimage
from mayavi import mlab
import nilearn
import scipy.io as sio

#from .Plot_3D import construct_mesh, construct_mesh_in_mask
import Plot_3D

# environment
DATA_PATH = "/home/mjia/Downloads/Image_CNN_FMRI/sceneViewingYork/"
#DATA_PATH = "/home/mjia/Image_CNN_FMRI/sceneViewingYork/"



def main():
    mask_file = DATA_PATH + 'BetaTimeSeriesToMengSean/s03/mask.nii'
    img_file = DATA_PATH + 'BetaTimeSeriesToMengSean/s03/beta_0001.nii'
    l_ROI = DATA_PATH + 'HippRois/03/L-HPC.nii'
    r_ROI = DATA_PATH + 'HippRois/03/R-HPC.nii'

    img = nib.load(img_file)
    mask_img = nib.load(mask_file)
    hipp_l = nib.load(l_ROI)
    hipp_r = nib.load(r_ROI)

    mask = mask_img.get_fdata()
    mask = np.nan_to_num(mask)
    #Plot_3D.construct_mesh(mask)

    image = img.get_fdata()
    image = np.nan_to_num(image)
    image = np.asarray(image)

    hipp_l_mask = hipp_l.get_fdata()
    hipp_r_mask = hipp_r.get_fdata()
    Plot_3D.construct_mesh_in_mask(hipp_l_mask, mask, hipp_l.affine, img.affine,)
    Plot_3D.construct_mesh_in_mask(hipp_r_mask, mask, hipp_r.affine, img.affine,)

    print(' ')

if __name__ == '__main__':
    main()