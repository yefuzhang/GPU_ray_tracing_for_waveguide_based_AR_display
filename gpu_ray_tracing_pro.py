import numpy as np
import time
import math
from numba import cuda, int32, float32
from couplers_coor import couplers_coor
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.colors import LogNorm
from scipy.io import savemat
from GPU_ray_tracing_functions import process_rays_kernel, process_rays_IC_kernel, process_rays_IC_kernel2, generate_points_in_polygon, pack_active_to_front, reset_counter_kernel, process_rays_IC_kernel_pro, process_rays_kernel_pro


print("Initializing ......")
print("Import waveguide design and couplers' coordinates ......")
num_FOV_x = 121
num_FOV_y = 81

IC, FC, FC_offset, OC, OC_offset, \
eff_reg1, eff_reg2, eff_reg_FOV, eff_reg_FOV_range, \
lut_TIR, lut_gap, \
Lambda_ic, phi_ic, Lambda_fc, phi_fc, Lambda_oc, phi_oc, n_g, lmd, \
th_in_ic, phi_in_ic, th_out_ic, phi_out_ic, th_out_fc, phi_out_fc, \
th_out_ic2, phi_out_ic2, th_out_oc, phi_out_oc = couplers_coor(num_FOV_x=num_FOV_x, num_FOV_y=num_FOV_y)

# load couplers' LUT
lut_ic1 = np.load('lut_ic1.npy')
lut_ic2 = np.load('lut_ic2.npy')
lut_ic3 = np.load('lut_ic3.npy')
lut_fc1 = np.load('lut_fc1.npy')
lut_fc2 = np.load('lut_fc2.npy')
lut_oc1 = np.load('lut_oc1.npy')
lut_oc2 = np.load('lut_oc2.npy')

# generating eyebox matrix
matrix_EB = np.zeros((num_FOV_y, num_FOV_x, 80, 120), dtype=np.float32)

# transfer variables to GPU
d_IC = cuda.to_device(IC)
d_FC = cuda.to_device(FC)
d_FC_offset = cuda.to_device(FC_offset)
d_OC = cuda.to_device(OC)
d_OC_offset = cuda.to_device(OC_offset)
d_eff_reg1 = cuda.to_device(eff_reg1)
d_eff_reg2 = cuda.to_device(eff_reg2)
d_eff_reg_FOV = cuda.to_device(eff_reg_FOV)
d_eff_reg_FOV_range = cuda.to_device(eff_reg_FOV_range)
d_lut_ic1 = cuda.to_device(lut_ic1)
d_lut_ic2 = cuda.to_device(lut_ic2)
d_lut_ic3 = cuda.to_device(lut_ic3)
d_lut_fc1 = cuda.to_device(lut_fc1)
d_lut_fc2 = cuda.to_device(lut_fc2)
d_lut_oc1  = cuda.to_device(lut_oc1)
d_lut_oc2  = cuda.to_device(lut_oc2)
d_lut_TIR = cuda.to_device(lut_TIR)
d_lut_gap = cuda.to_device(lut_gap)

# generating huge ray matrix
num_iter = 1
num_rays_per_FoV = 30000 # the number of rays per FoV
matrix_size = num_rays_per_FoV*num_FOV_x*num_FOV_y
num_rays = matrix_size

x = np.zeros((matrix_size), dtype=np.float32)
y = np.zeros((matrix_size), dtype=np.float32)
gap_x = np.zeros((matrix_size), dtype=np.float32)
gap_y = np.zeros((matrix_size), dtype=np.float32)
pol = np.zeros((matrix_size), dtype=np.float32)
azi = np.zeros((matrix_size), dtype=np.float32)
m = np.zeros((matrix_size), dtype=np.float32)
n = np.zeros((matrix_size), dtype=np.float32)
te = np.zeros((matrix_size), dtype=np.float32)
tm = np.zeros((matrix_size), dtype=np.float32)
delta_phase = np.zeros((matrix_size), dtype=np.float32)

# generate points within in-coupler
points = generate_points_in_polygon(IC, int(num_rays_per_FoV/2))
num = 0
for ii in range(num_FOV_x):
    for jj in range(num_FOV_y):
        start = int(num*num_rays_per_FoV)
        end = int(start+num_rays_per_FoV/2)
        # initialize ray properties
        x[start:end] = points[:,0] # position of ray (x,y)
        y[start:end] = points[:,1] # position of ray (x,y)
        gap_x[start:end] = 0 # position of ray (x,y) # propagation distance for each TIR (gap_x, gap_y)
        gap_y[start:end] = 0 # position of ray (x,y) # propagation distance for each TIR (gap_x, gap_y)
        pol[start:end] = 0 # polar angle of ray (global coordinates)
        azi[start:end] = 0 # azimuth angle of ray (global coordinates)
        m[start:end] = ii # LUT position: m
        n[start:end] = jj # LUT position: n
        te[start:end] = 1 # E-field: te
        tm[start:end] = 0 # E-field: tm
        delta_phase[start:end] = 0 # phase difference between te and tm (phi_tm - phi_te)  range: (-pi, pi]

        start = int(start+num_rays_per_FoV/2)
        end = int(start+num_rays_per_FoV/2)
        x[start:end] = points[:,0] # position of ray (x,y)
        y[start:end] = points[:,1] # position of ray (x,y)
        gap_x[start:end] = 0 # position of ray (x,y) # propagation distance for each TIR (gap_x, gap_y)
        gap_y[start:end] = 0 # position of ray (x,y) # propagation distance for each TIR (gap_x, gap_y)
        pol[start:end] = 0 # polar angle of ray (global coordinates)
        azi[start:end] = 0 # azimuth angle of ray (global coordinates)
        m[start:end] = ii # LUT position: m
        n[start:end] = jj # LUT position: n
        te[start:end] = 0 # E-field: te
        tm[start:end] = 1 # E-field: tm
        delta_phase[start:end] = 0 # phase difference between te and tm (phi_tm - phi_te)  range: (-pi, pi]
        num += 1

# points = generate_points_in_polygon(IC, int(num_rays/2))
# start = 0
# end = int(num_rays/2)
# x[start:end] = points[:,0] # position of ray (x,y)
# y[start:end] = points[:,1] # position of ray (x,y)
# gap_x[start:end] = 0 # position of ray (x,y) # propagation distance for each TIR (gap_x, gap_y)
# gap_y[start:end] = 0 # position of ray (x,y) # propagation distance for each TIR (gap_x, gap_y)
# pol[start:end] = 0 # polar angle of ray (global coordinates)
# azi[start:end] = 0 # azimuth angle of ray (global coordinates)
# m[start:end] = 25 # LUT position: m
# n[start:end] = 18 # LUT position: n
# te[start:end] = 1 # E-field: te
# tm[start:end] = 0 # E-field: tm
# delta_phase[start:end] = 0 # phase difference between te and tm (phi_tm - phi_te)  range: (-pi, pi]
# start = end
# end = int(num_rays)
# x[start:end] = points[:,0] # position of ray (x,y)
# y[start:end] = points[:,1] # position of ray (x,y)
# gap_x[start:end] = 0 # position of ray (x,y) # propagation distance for each TIR (gap_x, gap_y)
# gap_y[start:end] = 0 # position of ray (x,y) # propagation distance for each TIR (gap_x, gap_y)
# pol[start:end] = 0 # polar angle of ray (global coordinates)
# azi[start:end] = 0 # azimuth angle of ray (global coordinates)
# m[start:end] = 25 # LUT position: m
# n[start:end] = 18 # LUT position: n
# te[start:end] = 0 # E-field: te
# tm[start:end] = 1 # E-field: tm
# delta_phase[start:end] = 0 # phase difference between te and tm (phi_tm - phi_te)  range: (-pi, pi]

d_x = cuda.to_device(x)
d_y = cuda.to_device(y)
d_gap_x = cuda.to_device(gap_x)
d_gap_y = cuda.to_device(gap_y)
d_pol = cuda.to_device(pol)
d_azi = cuda.to_device(azi)
d_m = cuda.to_device(m)
d_n = cuda.to_device(n)
d_te = cuda.to_device(te)
d_tm = cuda.to_device(tm)
d_delta_phase = cuda.to_device(delta_phase)

d_rng_states = cuda.to_device((np.uint32(0x9E3779B9) * (np.arange(matrix_size, dtype=np.uint32) + np.uint32(1))))
d_matrix_EB = cuda.to_device(matrix_EB)
threads_per_block = 256

print("Finish initializaton!")

blocks_per_grid = (num_rays + threads_per_block - 1) // threads_per_block
start = time.perf_counter()
for _ in range(num_iter):
    process_rays_kernel_pro[blocks_per_grid, threads_per_block](
        d_x, d_y, d_gap_x, d_gap_y, d_pol, d_azi, d_m, d_n, d_te, d_tm, d_delta_phase, 
        d_rng_states, 
        d_IC, d_FC, d_FC_offset, d_OC, d_OC_offset, n_g,
        d_eff_reg1, d_eff_reg2, d_eff_reg_FOV, d_eff_reg_FOV_range,
        d_lut_ic1, d_lut_ic2, d_lut_ic3, d_lut_fc1, d_lut_fc2, d_lut_oc1, d_lut_oc2,
        d_lut_TIR, d_lut_gap,
        d_matrix_EB
    )
cuda.synchronize()  # <-- make kernel timing accurate
kern_time = time.perf_counter() - start
print(f"Kernels time: {kern_time:.2f} s")

matrix_EB = d_matrix_EB.copy_to_host()
A = np.sum(matrix_EB, axis=(-2, -1))/num_rays/num_iter
A2 = matrix_EB[18, 25, :, :]/num_rays/num_iter

efficiency = np.sum(A)
print(f"Effciency: {efficiency*100:.3f} %")

A = np.flipud(A)/np.max(A)
A2 = np.flipud(A2)/np.max(A2)

# plots
fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # 1 row, 2 columns

im1 = axes[0].imshow(A, cmap='jet')
im1.set_clim(0, 1)
fig.colorbar(im1, ax=axes[0], label='Intensity')
axes[0].set_title('FoV')

im2 = axes[1].imshow(A2, cmap='jet')
fig.colorbar(im2, ax=axes[1], label='Intensity')
axes[1].set_title('Eyebox')

plt.tight_layout()
plt.show()

savemat("matrix.mat", {"A": A})