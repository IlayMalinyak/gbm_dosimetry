# __author:IlayK
# data:12/04/2024

import numpy as np
import scipy.interpolate as spi
import scipy.io
import matplotlib.pyplot as plt

DOSE_MAT_PATH2 = 'GBM dose calculator/DART2D_sol_LRn030_LPb020_LBi002_PleakPb080_Time30d_l10_R0185_20-Jun-2023.mat'



def generate_dose_tables(dose_mat_path, num_r=200, num_theta=400):
    mat = scipy.io.loadmat(dose_mat_path)

    # Importing specific data from the simulation
    Gamma_Ra0 = mat['DART2D_sol'][0][0][10][0][0] / 3.7e4  # Convert to uCi
    L_Rn = mat['DART2D_sol'][0][0][19][0][0]
    L_Pb = mat['DART2D_sol'][0][0][20][0][0]
    dose_DART2D = np.asarray(mat['DART2D_sol'][0][0][33])
    r_DART2D = mat['DART2D_sol'][0][0][0][0]
    z_DART2D = mat['DART2D_sol'][0][0][2][0]
    R0_DART2D = mat['DART2D_sol'][0][0][12][0][0]
    l_DART2D = mat['DART2D_sol'][0][0][13][0][0]
    leak_Pb = mat['DART2D_sol'][0][0][8][0][0]

    # Filling the seed itself with the maximal dose
    dose_DART2D[dose_DART2D == 0] = np.amax(dose_DART2D)
    rmax = min(max(r_DART2D), max(z_DART2D))  # max radius for the r, theta lookup table (LUT)
    # Prepare r, theta dose lookup table
    R_DART2D, Z_DART2D = np.meshgrid(r_DART2D, z_DART2D)
    r_LUT = np.linspace(0.01, rmax, num_r)
    theta_LUT = np.linspace(0, 180, num_theta)
    R_LUT, THETA_LUT = np.meshgrid(r_LUT, theta_LUT)
    X = np.multiply(R_LUT, np.sin(THETA_LUT * np.pi / 180))
    Z = np.multiply(R_LUT, np.cos(THETA_LUT * np.pi / 180))

    # Changing the arrays to 1d for the interpolation
    X_1D = X.ravel()
    Z_1D = Z.ravel()
    dose_DART2D_1D = dose_DART2D.ravel()

    # Create a 2D interpolation function using griddata
    points = np.column_stack((R_DART2D.ravel(), Z_DART2D.ravel()))
    print("interpolating...")
    dose_LUT_1D = spi.griddata(points, dose_DART2D_1D, (X_1D, Z_1D), method='linear')
    print("finished")
    # Reshape dose_LUT back to the original shape
    dose_LUT = dose_LUT_1D.reshape(R_LUT.shape)
    return dose_LUT, R_LUT, THETA_LUT


def create_grid(w, wz, dx, dy, dz):
    xvec = np.arange(-w, w + dx, dx)
    yvec = np.arange(-w, w + dy, dy)
    zvec = np.arange(-wz, wz + dz, dz)
    xmat, ymat, zmat = np.meshgrid(xvec, yvec, zvec)
    return xmat.astype(np.float32), ymat.astype(np.float32), zmat.astype(np.float32)

def prepare_r_theta(seed, gridx, gridy, gridz):
    Lseed = np.linalg.norm(seed[0] - seed[-1])
    grid = np.vstack([gridx[None], gridy[None], gridz[None]])
    nseed = (seed[-1] - seed[0]) / Lseed
    cm = (seed[0] + seed[-1]) / 2
    # unit vector from seed center to point of interest + radial distance from seed center
    r = np.sqrt(np.sum((grid - cm[:,None,None,None])**2,axis=0))
    n = (grid - cm[:,None,None,None]) / (r + 1e-6)
    cos_theta = n[0]*nseed[0] + n[1]*nseed[1] + n[2]*nseed[2]
    theta = 180 / np.pi * np.real(np.arccos(cos_theta))
    return r, theta
def calc_dose_from_seed(seed,dose_LUT, R_LUT, THETA_LUT,
                        gridx,gridy, gridz, doseXYZ, eps=1e-6, rmax=10):
    r, theta = prepare_r_theta(seed, gridx, gridy, gridz)
    i = np.where(r < rmax)
    r_i, theta_i = r[i], theta[i]
    # print(f"interpolating seed {n}...")
    dose_interp = spi.griddata((R_LUT.ravel(), THETA_LUT.ravel()),
                                    dose_LUT.ravel(), (r_i, theta_i), method='linear')
    doseXYZ[i] += 0.5 * dose_interp
    return doseXYZ

def check_dose_at_seed(seed, gridx, gridy, gridz, doseXYZ, tol=1):
    r, theta = prepare_r_theta(seed, gridx, gridy, gridz)
    i = np.where(r < tol)
    return doseXYZ[i], i
def calc_dose_multiple_seeds(seeds,dose_LUT, R_LUT, THETA_LUT,
                             gridx, gridy, gridz, doseXYZ):
    for s in range(seeds.shape[-1]):
        doseXYZ = calc_dose_from_seed(seeds[...,s],dose_LUT, R_LUT, THETA_LUT,
                                      gridx, gridy, gridz, doseXYZ)
        doseXYZ = np.nan_to_num(doseXYZ, nan=0.0, posinf=0.0, neginf=0.0)
    return doseXYZ

# dose_LUT, R_LUT, THETA_LUT = generate_dose_tables(DOSE_MAT_PATH2, num_r=50, num_theta=50)
# x,y,z = create_grid(8,25,0.1,0.1,0.1)
# seed = np.array([[10,5,5],[20,5,5]])
# doseXYZ = np.zeros(x.shape)
# doseXYZ = calc_dose_from_seed(seed, x, y, z, doseXYZ)
# doseXYZ = np.nan_to_num(doseXYZ, nan=0.0, posinf=0.0, neginf=0.0)
# print(doseXYZ.sum())
# print(dose_LUT.shape, R_LUT.shape, THETA_LUT.shape)
# print(x.shape, y.shape, z.shape)
# plt.imshow(dose_LUT)
# plt.show()
# plt.imshow(R_LUT)
# plt.show()