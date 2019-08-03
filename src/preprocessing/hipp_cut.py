import os
import nibabel as nib
import skimage
import pymesh
from scipy import linalg as sciLA
import scipy
from skimage.exposure import equalize_hist
import numpy as np
import sys
import copy
sys.path.insert(0, '..')
import Plot_3D
from test import there_is_out
from mayavi import mlab

NUM_CUT = 6
HIPP_ROI_DIR = "/home/mjia/Downloads/Image_CNN_FMRI/sceneViewingYork"





def segment(eigen):
    output = np.zeros_like(eigen)
    eigen_eq = equalize_hist(eigen)
    min = np.min(eigen_eq)
    range_total = np.max(eigen_eq) - min
    interval = range_total / NUM_CUT
    for i in range(NUM_CUT):
        output[np.where(eigen_eq>(min+i*interval))] = i
    return output

def hipp_cut(hipp_filename):
    hipp_img = nib.load(hipp_filename)
    hipp_matrix = hipp_img.get_fdata()
    hipp_matrix_s = scipy.ndimage.filters.gaussian_filter(hipp_matrix, sigma=1)
    #mesh it
    verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(hipp_matrix_s, level=0.5)
    hipp_mesh = pymesh.form_mesh(verts, faces)
    hipp_mesh, info = pymesh.remove_duplicated_faces(hipp_mesh)
    assembler = pymesh.Assembler(hipp_mesh)
    L = assembler.assemble('laplacian').toarray()
    eigen = sciLA.eigh(L)
    for i in range(50):
        #print(eigen[0][i])
        #Plot_3D.plot_eigh(hipp_mesh, eigen[1][:, i])
        if (eigen[0][i] > 0.0000000001):
            eigen_cut = segment(eigen[1][:, i])
            mean_first = np.mean(verts[np.where(eigen_cut==0), :], axis=1)
            mean_last = np.mean(verts[np.where(eigen_cut==(NUM_CUT-1)), :], axis=1)
            diff = -mean_first[0, 1]    + mean_last[0, 1] + mean_first[0, 2] - mean_last[0, 2]
            eigen_cut_output = copy.copy(eigen_cut)
            if diff < 0:
                for i in range(NUM_CUT):
                    eigen_cut_output[np.where(eigen_cut==i)] = NUM_CUT-i-1
            #Plot_3D.plot_eigh(hipp_mesh, eigen_cut)
            break
    return eigen_cut_output, hipp_mesh

def save_output(cut, hipp_mesh, full_filename):
    save_filename = full_filename[:-4] + '-Seg-6.nii'
    hipp_ROI = nib.load(full_filename)

    hipp_ROI_matrix = hipp_ROI.get_fdata()
    seg_matrix = np.zeros_like(hipp_ROI_matrix)
    index = np.where(hipp_ROI_matrix != 0)
    index_3d = np.concatenate([np.expand_dims(index[0], axis=1),
                               np.expand_dims(index[1], axis=1),
                               np.expand_dims(index[2], axis=1)], axis=1)
    distance2vert = np.expand_dims(index_3d, axis=1) - np.expand_dims(hipp_mesh.vertices, axis=0)
    distance2vert = np.sum(distance2vert * distance2vert, axis=2)
    NNV = np.argmin(distance2vert, axis=1)
    seg_matrix[index] = cut[NNV] + 1

    segmented_img = nib.Nifti1Image(seg_matrix, hipp_ROI.affine, hipp_ROI.header)
    segmented_img.to_filename(save_filename)



def main():
    for root, dirs, files in os.walk(HIPP_ROI_DIR):
        for i in files:
            if i == "L-HPC.nii":
                L_full_filename = root + '/' + i
                L_cut, L_mesh = hipp_cut(L_full_filename)
                R_full_filename = root + '/R' + i[1:]
                R_cut, R_mesh = hipp_cut(R_full_filename)

                id = L_full_filename.split('/')[-2]
                brain_filename = '/home/mjia/Downloads/Image_CNN_FMRI/sceneViewingYork/BetaTimeSeriesToMengSean/s' + id + '/mask.nii'
                Plot_3D.plot_seg(L_full_filename,
                                 R_full_filename,
                                 brain_filename,
                                 L_cut, R_cut, level1=0.5, level2=0.5, level3=0.1)



                #L_out, R_out = there_is_out(L_full_filename, R_full_filename, brain_filename)
                #print(id + ' --- left: ' + str(L_out) + '   right: ' + str(R_out))
                #save_output(L_cut, L_mesh, L_full_filename)
                #save_output(R_cut, R_mesh, R_full_filename)


if __name__ == '__main__':
    main()