import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from HW1 import mask_visualization
from HW1 import normal_visualization
from HW1 import depth_visualization
from HW1 import save_ply
from HW1 import show_ply
import scipy.sparse as sp
path = "./test/venus/"

IMG1 = cv2.imread(path + 'pic1.bmp', cv2.IMREAD_GRAYSCALE)
IMG2 = cv2.imread(path + 'pic2.bmp', cv2.IMREAD_GRAYSCALE)
IMG3 = cv2.imread(path + 'pic3.bmp', cv2.IMREAD_GRAYSCALE)
IMG4 = cv2.imread(path + 'pic4.bmp', cv2.IMREAD_GRAYSCALE)
IMG5 = cv2.imread(path + 'pic5.bmp', cv2.IMREAD_GRAYSCALE)
IMG6 = cv2.imread(path + 'pic6.bmp', cv2.IMREAD_GRAYSCALE)
print(IMG1.shape)
lights = []
with open(path + "LightSource.txt", 'r', encoding='utf8') as f:
    for i in f.readlines():
        each_light = i[7: -2].split(',')
        #each_light = np.array(each_light)
        lights.append(each_light)
    
lights = np.array(lights).astype(np.float)
unit_lights = []
for i in lights:
    unit_lights.append(i/ np.linalg.norm(i))
unit_lights = np.array(unit_lights)
normal_image = np.zeros((IMG1.shape[0], IMG1.shape[1], 3))
LTL_inv = np.linalg.inv(np.dot(unit_lights.T, unit_lights)) 
for i in range(IMG1.shape[0]):
    for j in range(IMG1.shape[1]):
        I = np.array(
            [IMG1[i][j],
             IMG2[i][j],
             IMG3[i][j],
             IMG4[i][j],
             IMG5[i][j],
             IMG6[i][j]]
        )
        LTI = np.dot(unit_lights.T, I)
        n = np.dot(LTL_inv, LTI)
        norm = np.linalg.norm(n)
        if norm != 0:
            n /= norm
        #print(n.shape)
            normal_image[i][j] = n

normal_visualization(normal_image, IMG1.shape[0], IMG1.shape[1], "venus_normal.png")
print(normal_image[50][50])        
#make mask
mask = np.zeros(IMG1.shape)
for i in range(IMG1.shape[0]):
    for j in range(IMG1.shape[1]):
        #print(normal_image[i][j].all())
        if normal_image[i][j].all() != 0:
            mask[i][j] = 1
mask_visualization(mask, IMG1.shape[0], IMG1.shape[1])

nonzero_h, nonzero_w = np.where(mask != 0)

pix_num = len(nonzero_h)
print("no_pix: ", pix_num)
#giving those pixels whose value are not zero IDs
idxplane = np.zeros(mask.shape)
for i in range(len(nonzero_h)):
    idxplane[nonzero_h[i], nonzero_w[i]] = i

# Mz = v
M = sp.lil_matrix((2*pix_num, pix_num))
V = np.zeros((2*pix_num, 1))

for i in range(pix_num):
    # get position
    h = nonzero_h[i]
    w = nonzero_w[i]
    # get normal value
    n_x = normal_image[h, w, 0]
    n_y = normal_image[h, w, 1]
    n_z = normal_image[h, w, 2]

    # z_(x+1, y) - z(x, y) = -nx / nz
    row_i = i * 2
    if mask[h, w+1]:
        idx_horiz = idxplane[h, w+1]
        M[row_i, i] = -1
        M[row_i, idx_horiz] = 1
        V[row_i] = -n_x / n_z

    elif mask[h, w-1]:
        idx_horiz = idxplane[h, w-1]
        M[row_i, idx_horiz] = -1
        M[row_i, i] = 1
        V[row_i] = -n_x / n_z

    # z_(x, y+1) - z(x, y) = -ny / nz
    row_i = i * 2 + 1
    if mask[h+1, w]:
        idx_vert = idxplane[h+1, w]
        M[row_i, i] = 1
        M[row_i, idx_vert] = -1
        V[row_i] = -n_y / n_z
        
    elif mask[h-1, w]:
        idx_vert = idxplane[h-1, w]
        M[row_i, idx_vert] = 1
        M[row_i, i] = -1
        V[row_i] = -n_y / n_z
MTM = M.T @ M
MTV = M.T @ V
z = sp.linalg.spsolve(MTM, MTV)

std_z = np.std(z, ddof=1)
mean_z = np.mean(z)
z_zscore = (z - mean_z) / std_z

# solve singular point problem
outlier_ind = np.abs(z_zscore) > 10
z_min = np.min(z[~outlier_ind])
z_max = np.max(z[~outlier_ind])

#back to 2D
Z = mask.astype('float')
for i in range(pix_num):
    h = nonzero_h[i]
    w = nonzero_w[i]
    Z[h, w] = (z[i] - z_min) / (z_max - z_min) * 255

depth_visualization(Z, IMG1.shape[0], IMG1.shape[1], "venus_depth.png")
save_ply(Z, "venus.ply", IMG1.shape[0], IMG1.shape[1])
show_ply("venus.ply")