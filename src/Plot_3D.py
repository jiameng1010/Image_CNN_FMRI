import numpy as np
import skimage
from mayavi import mlab
import nibabel as nib
import scipy


def apply_affine(verts, affine):
    v = np.concatenate([verts, np.ones(shape=(verts.shape[0], 1))], axis=1)
    v = np.matmul(affine, v.T)
    v = v.T
    v = v / np.expand_dims(v[:, 3], 1)
    return v[:, :3]

def construct_mesh(matrix):
    verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(matrix, level=0.005)
    # verts, faces = skimage.measure.marching_cubes_classic(self.occupy_matrix[i])

    # rescale
    verts = verts - np.asarray([[128, 128, 128]])
    verts = 1.1 * verts / (128 * 2)

    mlab.triangular_mesh([vert[0] for vert in verts],
                         [vert[1] for vert in verts],
                         [vert[2] for vert in verts],
                         faces)
    mlab.show()


def construct_mesh_in_mask(ROI_matrix, mask_matrix, affine1, affine2, level1=0.0, level2=0.0):
    verts1, faces1, normals1, values1 = skimage.measure.marching_cubes_lewiner(ROI_matrix, level=level1)
    verts2, faces2, normals2, values2 = skimage.measure.marching_cubes_lewiner(mask_matrix, level=level2)
    # verts, faces = skimage.measure.marching_cubes_classic(self.occupy_matrix[i])
    verts1 = apply_affine(verts1, affine1)
    verts2 = apply_affine(verts2, affine2)

    color1 = -0.6*np.ones(shape=[verts1.shape[0]])
    color2 = 0.6*np.ones(shape=[verts2.shape[0]])
    color1[0] = -1
    color1[-1] = 1

    mlab.triangular_mesh([vert[0] for vert in verts1],
                         [vert[1] for vert in verts1],
                         [vert[2] for vert in verts1],
                         faces1,
                         opacity=0.9,
                         scalars=color1)

    mlab.triangular_mesh([vert[0] for vert in verts2],
                         [vert[1] for vert in verts2],
                         [vert[2] for vert in verts2],
                         faces2,
                         opacity=0.4,
                         scalars=color2)
    mlab.show()

def plot_seg(filename1, filename2, filename3, color1, color2, level1=0.1, level2=0.1, level3=0.1):
    f1 = nib.load(filename1)
    f2 = nib.load(filename2)
    f3 = nib.load(filename3)
    ROI_L = f1.get_fdata()
    ROI_L = scipy.ndimage.filters.gaussian_filter(ROI_L, sigma=1)
    ROI_R = f2.get_fdata()
    ROI_R = scipy.ndimage.filters.gaussian_filter(ROI_R, sigma=1)
    mask_matrix = f3.get_fdata()
    verts1, faces1, normals1, values1 = skimage.measure.marching_cubes_lewiner(ROI_L, level=level1)
    verts2, faces2, normals2, values2 = skimage.measure.marching_cubes_lewiner(ROI_R, level=level2)
    verts3, faces3, normals3, values3 = skimage.measure.marching_cubes_lewiner(mask_matrix, level=level3)
    # verts, faces = skimage.measure.marching_cubes_classic(self.occupy_matrix[i])
    verts1 = apply_affine(verts1, f1.affine)
    verts2 = apply_affine(verts2, f2.affine)
    verts3 = apply_affine(verts3, f3.affine)

    color3 = 0.6*np.ones(shape=[verts3.shape[0]])
    color3[0] = -1
    color3[-1] = 1

    mlab.triangular_mesh([vert[0] for vert in verts1],
                         [vert[1] for vert in verts1],
                         [vert[2] for vert in verts1],
                         faces1,
                         opacity=1,
                         scalars=color1)

    mlab.triangular_mesh([vert[0] for vert in verts2],
                         [vert[1] for vert in verts2],
                         [vert[2] for vert in verts2],
                         faces2,
                         opacity=1,
                         scalars=color2)
    mlab.triangular_mesh([vert[0] for vert in verts3],
                         [vert[1] for vert in verts3],
                         [vert[2] for vert in verts3],
                         faces3,
                         opacity=0.5,
                         scalars=color3)
    mlab.show()


def plot_eigh(mesh, eigh):
    mlab.triangular_mesh([vert[0] for vert in mesh.vertices],
                         [vert[1] for vert in mesh.vertices],
                         [vert[2] for vert in mesh.vertices],
                         mesh.faces,
                         opacity=1,
                         scalars=eigh)
    mlab.show()

def main():
    return 0