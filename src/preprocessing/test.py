import numpy as np
import skimage
from mayavi import mlab
import nibabel as nib

def apply_affine(verts, affine):
    v = np.concatenate([verts, np.ones(shape=(verts.shape[0], 1))], axis=1)
    v = np.matmul(affine, v.T)
    v = v.T
    v = v / np.expand_dims(v[:, 3], 1)
    return v[:, :3]

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
    np.unique(v, axis=0)
    return (v[:, 0], v[:, 1], v[:, 2])

def there_is_out(filename1, filename2, filename3):
    f1 = nib.load(filename1)
    f2 = nib.load(filename2)
    f3 = nib.load(filename3)
    ROI_L = f1.get_fdata()
    ROI_R = f2.get_fdata()
    mask = f3.get_fdata()

    #L
    index_in_mask = apply_affine2destination(np.where(ROI_L != 0), f1.affine, f3.affine)
    flagL = mask[index_in_mask]

    #R
    index_in_mask = apply_affine2destination(np.where(ROI_R != 0), f2.affine, f3.affine)
    flagR = mask[index_in_mask]

    if np.min(flagL)==0 or np.min(flagR)==0:
        return np.sum(1*flagL==0)/flagL.shape[0], np.sum(1*flagR==0)/flagR.shape[0]
    else:
        return 0, 0