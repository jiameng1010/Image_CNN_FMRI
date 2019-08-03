import os
import nibabel as nib
import skimage
import pymesh
from scipy import linalg as sciLA
import scipy
from skimage.exposure import equalize_hist
import numpy as np
import sys
import h5py as h5
import copy
sys.path.insert(0, '..')
import Plot_3D
from test import there_is_out
from mayavi import mlab

NUM_CUT = 6
NUM_SCAN = 632
HIPP_ROI_DIR = "/home/mjia/Downloads/Image_CNN_FMRI/sceneViewingYork"
SUBJECT_SCAN_DIR = '/home/mjia/Downloads/Image_CNN_FMRI/sceneViewingYork/BetaTimeSeriesToMengSean'

def apply_affine2destination(verts, affine, affine_destination):
    verts = np.concatenate([np.expand_dims(verts[0], axis=1),
                            np.expand_dims(verts[1], axis=1),
                            np.expand_dims(verts[2], axis=1)], axis=1)
    v = np.concatenate([verts, np.ones(shape=(verts.shape[0], 1))], axis=1)
    v = np.matmul(affine, v.T)
    v = np.matmul(np.linalg.inv(affine_destination), v)
    v = v.T
    v = v / np.expand_dims(v[:, 3], 1)
    v = np.round(v)
    v = v.astype(int)
    v = v[:, :3]
    v = np.unique(v, axis=0)
    return (v[:, 0], v[:, 1], v[:, 2])

def get_mean(scan_name, L_Seg_ROI, R_Seg_ROI):
    scan = nib.load(scan_name)
    scan_img = scan.get_fdata()
    L_hipp_seg = L_Seg_ROI.get_fdata()
    R_hipp_seg = R_Seg_ROI.get_fdata()

    mean_L = np.zeros(shape=(NUM_CUT))
    mean_R = np.zeros(shape=(NUM_CUT))
    for i in range(NUM_CUT):
        seg_index = i + 1
        i_th_seg_L_index = np.where(np.abs(L_hipp_seg - seg_index) < 0.01)
        if i_th_seg_L_index[0].shape[0] == 0:
            return 0, 0
        i_th_seg_L_index = apply_affine2destination(i_th_seg_L_index, L_Seg_ROI.affine, scan.affine)
        i_th_seg_L = scan_img[i_th_seg_L_index]
        i_th_seg_L = i_th_seg_L[~np.isnan(i_th_seg_L)]
        mean_L[i] = np.mean(i_th_seg_L)

        i_th_seg_R_index = np.where(np.abs(R_hipp_seg - seg_index) < 0.01)
        if i_th_seg_R_index[0].shape[0] == 0:
            return 0, 0
        i_th_seg_R_index = apply_affine2destination(i_th_seg_R_index, R_Seg_ROI.affine, scan.affine)
        i_th_seg_R = scan_img[i_th_seg_R_index]
        i_th_seg_R = i_th_seg_R[~np.isnan(i_th_seg_R)]
        mean_R[i] = np.mean(i_th_seg_R)

    return mean_L, mean_R

def main():
    for root, dirs, files in os.walk(HIPP_ROI_DIR):
        for i in files:
            if i == "L-HPC-Seg-6.nii":
                L_full_filename = root + '/' + i
                R_full_filename = root + '/R' + i[1:]
                L_Seg_ROI = nib.load(L_full_filename)
                R_Seg_ROI = nib.load(R_full_filename)

                id = L_full_filename.split('/')[-2]
                if id == '09':
                    continue
                all_means_L = np.zeros(shape=[NUM_SCAN, NUM_CUT])
                all_means_R = np.zeros(shape=[NUM_SCAN, NUM_CUT])
                for scan in range(NUM_SCAN):
                    scan_name = 'beta_' + str(scan+1).zfill(4) + '.nii'
                    scan_name = SUBJECT_SCAN_DIR + '/s' + id + '/' + scan_name
                    means_L, means_R = get_mean(scan_name, L_Seg_ROI, R_Seg_ROI)
                    if isinstance(means_L, int) and isinstance(means_R, int):
                        print(L_full_filename)
                        print(R_full_filename)
                        break
                    all_means_L[scan, :] = means_L
                    all_means_R[scan, :] = means_R

                h5_filename = SUBJECT_SCAN_DIR + '/s' + id + '/hipp_seg_means.h5'
                h5_file = h5.File(h5_filename, 'w')
                h5_file.create_dataset('left_hipp_means', data=all_means_L)
                h5_file.create_dataset('right_hipp_means', data=all_means_R)
                h5_file.close()
                print("s" + str(id) + ' saved')


if __name__ == '__main__':
    main()