import numpy as np
import time
from numba import cuda, int32, float32
from couplers_coor import couplers_coor_full_color
import matplotlib.pyplot as plt
from scipy.io import savemat
import GPU_ray_tracing_functions as GRTF
from AR_system_evaluation_functions import evaluation
import cv2

print("\n" + "="*60)
print("🧠  Initializing System Components ...")
print("="*60 + "\n")
print("⚙️  Loading waveguide design and coupler coordinates ...")

num_FOV_x = 100
num_FOV_y = 75

IC, FC, FC_offset, OC, OC_offset, \
eff_reg1, eff_reg2, eff_reg_FOV, eff_reg_FOV_range, \
lut_TIR, lut_gap, lut_Fresnel, \
Lambda_ic, phi_ic, Lambda_fc, phi_fc, Lambda_oc, phi_oc, n_g, lmd, \
th_in_ic, phi_in_ic, th_out_ic, phi_out_ic, th_out_fc, phi_out_fc, \
th_out_ic2, phi_out_ic2, th_out_oc, phi_out_oc, th_out_oc_glow,\
kx_ic, ky_ic, kx_fc, ky_fc, kx_oc, ky_oc  = couplers_coor_full_color(num_FOV_x=num_FOV_x, num_FOV_y=num_FOV_y)

# load couplers' LUT
lut_ic1 = np.load('lut_ic1_fullColor.npy')
lut_ic2 = np.load('lut_ic2_fullColor.npy')
lut_ic3 = np.load('lut_ic3_fullColor.npy')
lut_fc1 = np.load('lut_fc1_fullColor.npy')
lut_fc2 = np.load('lut_fc2_fullColor.npy')
lut_oc1 = np.load('lut_oc1_fullColor.npy')
lut_oc2 = np.load('lut_oc2_fullColor.npy')

# generating eyebox matrix
matrix_EB = np.zeros((len(lmd), num_FOV_y, num_FOV_x, 80, 120), dtype=np.float32)

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
num_iter = 4
num_rays_per_FoV = 5000 # the number of rays per FoV
matrix_size = num_rays_per_FoV*num_FOV_x*num_FOV_y*len(lmd)
num_rays = matrix_size

x = np.zeros((matrix_size), dtype=np.float32)
y = np.zeros((matrix_size), dtype=np.float32)
gap_x = np.zeros((matrix_size), dtype=np.float32)
gap_y = np.zeros((matrix_size), dtype=np.float32)
pol = np.zeros((matrix_size), dtype=np.float32)
azi = np.zeros((matrix_size), dtype=np.float32)
m = np.zeros((matrix_size), dtype=np.float32)
n = np.zeros((matrix_size), dtype=np.float32)
lmd_num = np.zeros((matrix_size), dtype=np.float32)
te = np.zeros((matrix_size), dtype=np.float32)
tm = np.zeros((matrix_size), dtype=np.float32)
delta_phase = np.zeros((matrix_size), dtype=np.float32)

# generate points within in-coupler
points = GRTF.generate_points_in_polygon(IC, int(num_rays_per_FoV/2))
num = 0

for ii in range(num_FOV_x):
    for jj in range(num_FOV_y):
        for num_lmd in range(len(lmd)):
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
            lmd_num[start:end] = num_lmd # wavelength
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
            lmd_num[start:end] = num_lmd # wavelength
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
d_lmd_num = cuda.to_device(lmd_num)
d_te = cuda.to_device(te)
d_tm = cuda.to_device(tm)
d_delta_phase = cuda.to_device(delta_phase)

d_rng_states = cuda.to_device((np.uint32(0x9E3779B9) * (np.arange(matrix_size, dtype=np.uint32) + np.uint32(1))))
d_matrix_EB = cuda.to_device(matrix_EB)
threads_per_block = 256

print("✅ Initialization Complete!\n")

print("="*60)
print("🚀  START GPU RAY TRACING")
print("="*60 + "\n")
blocks_per_grid = (num_rays + threads_per_block - 1) // threads_per_block
start = time.perf_counter()
for _ in range(num_iter):
    GRTF.process_rays_kernel_pro_fullColor[blocks_per_grid, threads_per_block](
        d_x, d_y, d_gap_x, d_gap_y, d_pol, d_azi, d_m, d_n, d_lmd_num, d_te, d_tm, d_delta_phase, 
        d_rng_states, 
        d_IC, d_FC, d_FC_offset, d_OC, d_OC_offset, n_g,
        d_eff_reg1, d_eff_reg2, d_eff_reg_FOV, d_eff_reg_FOV_range,
        d_lut_ic1, d_lut_ic2, d_lut_ic3, d_lut_fc1, d_lut_fc2, d_lut_oc1, d_lut_oc2,
        d_lut_TIR, d_lut_gap,
        d_matrix_EB)
cuda.synchronize()  # <-- make kernel timing accurate
kern_time = time.perf_counter() - start
print("🎯  Simulation Finished Successfully!\n")
print(f"💡 Number of rays traced : {num_rays * num_iter:,}")
print(f"🕒  GPU Calculation Time : {kern_time:.2f} s\n")

# result analysis
matrix_EB = d_matrix_EB.copy_to_host()
A = np.sum(matrix_EB, axis=(-2, -1))/num_rays/num_iter
# A2 = matrix_EB[1, 18, 25, :, :]/num_rays/num_iter
# A2 = np.flipud(A2)/np.max(A2)

efficiency_R = np.sum(A[2, :, :]*3)
efficiency_G = np.sum(A[1, :, :]*3)
efficiency_B = np.sum(A[0, :, :]*3)
print("\n" + "="*60)
print(f"{'RESULTS':^60}")
print("="*60 + "\n")

matrix_EB2 = matrix_EB/num_rays_per_FoV/num_iter
delta_e, U_fov, U_EB, output_image = evaluation(matrix_EB2)
n_FOVy, n_FOVx, _, n_epy, n_epx = output_image.shape
final_image_uint8 = (output_image[:, :, :, 0, n_epx-1]*255).astype(np.uint8)
final_image_uint8 = np.flipud(final_image_uint8)
final_image_bgr = cv2.cvtColor(final_image_uint8, cv2.COLOR_RGB2BGR)
cv2.imwrite(f'Eyebox Center View.png', final_image_bgr)

print("🔴 Efficiency (Red)     : {:8.3f} %".format(efficiency_R * 100))
print("🟢 Efficiency (Green)   : {:8.3f} %".format(efficiency_G * 100))
print("🔵 Efficiency (Blue)    : {:8.3f} %".format(efficiency_B * 100))
print("🎨 Color dispersion     : {:8.2f}".format(delta_e))
print("🌈 FoV uniformity       : {:8.2f} %".format(U_fov * 100))
print("👁️  Eyebox uniformity    : {:8.2f} %".format(U_EB * 100))


FOV_R = np.flipud(A[2, :, :])/np.max(A[2, :, :])
FOV_G = np.flipud(A[1, :, :])/np.max(A[1, :, :])
FOV_B = np.flipud(A[0, :, :])/np.max(A[0, :, :])

# plots
fig, axes = plt.subplots(1, 3, figsize=(10, 4))  # 1 row, 2 columns

im1 = axes[0].imshow(FOV_R, cmap='jet')
im1.set_clim(0, 1)
fig.colorbar(im1, ax=axes[0], label='Intensity')
axes[0].set_title('Normalized FoV Efficiency R')

im2 = axes[1].imshow(FOV_G, cmap='jet')
im2.set_clim(0, 1)
fig.colorbar(im2, ax=axes[1], label='Intensity')
axes[1].set_title('Normalized FoV Efficiency G')

im3 = axes[2].imshow(FOV_B, cmap='jet')
im3.set_clim(0, 1)
fig.colorbar(im3, ax=axes[2], label='Intensity')
axes[2].set_title('Normalized FoV Efficiency B')

plt.tight_layout()

plt.show()



