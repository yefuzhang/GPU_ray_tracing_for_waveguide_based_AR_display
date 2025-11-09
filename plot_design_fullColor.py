import numpy as np
import math
from numba import cuda, int32, float32
from couplers_coor import couplers_coor_full_color
import matplotlib.pyplot as plt
from matplotlib.path import Path
from scipy.spatial import ConvexHull
import alphashape

print("Initializing ......")
print("Import waveguide design and couplers' coordinates ......")
num_FOV_x = 100
num_FOV_y = 75

IC, FC, FC_offset, OC, OC_offset, \
eff_reg1, eff_reg2, eff_reg_FOV, eff_reg_FOV_range, \
lut_TIR, lut_gap, lut_Fresnel, \
Lambda_ic, phi_ic, Lambda_fc, phi_fc, Lambda_oc, phi_oc, n_g, lmd, \
th_in_ic, phi_in_ic, th_out_ic, phi_out_ic, th_out_fc, phi_out_fc, \
th_out_ic2, phi_out_ic2, th_out_oc, phi_out_oc, th_out_oc_glow,\
kx_ic, ky_ic, kx_fc, ky_fc, kx_oc, ky_oc  = couplers_coor_full_color(num_FOV_x=num_FOV_x, num_FOV_y=num_FOV_y)
print("Complete!")

# plot K-diagram
theta_max = np.arctan(4 / 2 / 0.7)  # maximum TIR angle if pupil is continuous at largest TIR angle (4mm is the diameter of in-coupler, 0.7mm is thickness of waveguide) 
t_ic = np.linspace(0, 2 * np.pi, 200)
c = [0, 0]
r1 = 1
r2 = n_g
r3 = n_g * np.sin(theta_max)
x1 = c[0] + r1 * np.sin(t_ic)
y1 = c[1] + r1 * np.cos(t_ic)
x2 = c[0] + r2 * np.sin(t_ic)
y2 = c[1] + r2 * np.cos(t_ic)
x3 = c[0] + r3 * np.sin(t_ic)
y3 = c[1] + r3 * np.cos(t_ic)

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 16,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "axes.labelsize": 24,
    "axes.titlesize": 26,
    "axes.linewidth": 1.2,
    "xtick.direction": "in",
    "ytick.direction": "in",
})

fig, ax = plt.subplots(figsize=(8, 6))

# Base curves
ax.plot(x1, y1, label='Air boundary', lw=1.6)
ax.plot(x2, y2, label='Glass boundary', lw=1.6)
ax.plot(x3, y3, '--', label='Max TIR angle', lw=1.4)

# Regions
k0 = 2 * np.pi / lmd
colors = ["tab:blue", "tab:green", "tab:red"]

# IC region (using the first wavelength)
points_ic = np.vstack((kx_ic[0, :] / k0[0], ky_ic[0, :] / k0[0]))
hull_ic = ConvexHull(points_ic.T)
ax.fill(points_ic[0, hull_ic.vertices], points_ic[1, hull_ic.vertices],
        color="#cccccc", alpha=0.5)

# FC/OC regions
for i in range(len(lmd)):
    fc = np.vstack((kx_fc[i, :] / k0[i], ky_fc[i, :] / k0[i]))
    oc = np.vstack((kx_oc[i, :] / k0[i], ky_oc[i, :] / k0[i]))
    hull_fc = ConvexHull(fc.T)
    hull_oc = ConvexHull(oc.T)
    ax.fill(fc[0, hull_fc.vertices], fc[1, hull_fc.vertices],
            color=colors[i], alpha=0.35)
    ax.fill(oc[0, hull_oc.vertices], oc[1, hull_oc.vertices],
            color=colors[i], alpha=0.35)

# Axes/labels
ax.set_aspect('equal', adjustable='box')
ax.grid(False)
ax.set_xticks(np.arange(-2, 2.1, 1))
ax.set_yticks(np.arange(-2, 2.1, 1))
ax.set_title('k-diagram', fontweight='bold')
ax.set_xlabel(r'$k_x/k_0$', fontweight='bold')
ax.set_ylabel(r'$k_y/k_0$', fontweight='bold')
ax.legend(loc='best')
plt.tight_layout()

fig, ax = plt.subplots(figsize=(8, 6))
for i in range(len(OC_offset)-1):
    start = OC_offset[i]
    end = OC_offset[i+1]
    ax.fill(OC[start:end,0], OC[start:end,1], 'b', alpha=0.4, edgecolor='black')
    # print(OC[start:end,:])

for i in range(len(FC_offset)-1):
    start = FC_offset[i]
    end = FC_offset[i+1]
    ax.fill(FC[start:end,0], FC[start:end,1], 'g', alpha=0.4, edgecolor='black')

ax.fill(IC[:,0], IC[:,1], 'r', alpha=0.4, edgecolor='black')
# Lens size
width = 58   # mm
height = 42  # mm
a = width / 2
b = height / 2
n = 4  # Superellipse exponent

# Superellipse full shape
theta_main = np.linspace(0, 2 * np.pi, 500)
x_main = a * np.sign(np.cos(theta_main)) * np.abs(np.cos(theta_main))**(2/n)
y_main = b * np.sign(np.sin(theta_main)) * np.abs(np.sin(theta_main))**(2/n)

# Half-circle notch on left side
r = b / 2
theta_half = np.linspace(np.pi/2, 3*np.pi/2, 100)
x_half = -a + r * np.cos(theta_half)+6
y_half = r * np.sin(theta_half)+1.5

# Combine all candidate boundary points
x_all = np.concatenate([x_main, x_half])
y_all = np.concatenate([y_main, y_half])
points = np.vstack((x_all, y_all)).T

# Compute convex hull to extract outer edge
hull = ConvexHull(points)
hull_points = points[hull.vertices]

ax.fill(hull_points[:, 0], hull_points[:, 1]+13, color='lightblue', edgecolor='black', alpha=0.3)
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(-35, 30)
ax.set_ylim(-10, 35)
ax.set_xticks(np.arange(-35, 31, 10))
ax.set_yticks(np.arange(-10, 36, 10))
ax.set_xlabel('x (mm)', fontweight='bold')
ax.set_ylabel('y (mm)', fontweight='bold')
ax.set_title('Waveguide Design', fontweight='bold', pad=8)
ax.grid(False)
plt.tight_layout()

# angular spectrum
deg = np.pi/180
fig, ax = plt.subplots(figsize=(8, 6))
labels = np.array(["Blue (465nm)", "Green (532nm)", "Red (630nm)"])
for num in range(len(lmd)):
    points = np.column_stack(((th_out_ic[2-num,:,:]/deg).flatten(), (phi_out_ic[2-num,:,:]/deg).flatten()))
    alpha = 0.1  # smaller alpha = more detail
    shape = alphashape.alphashape(points, alpha)
    x, y = shape.exterior.xy
    ax.fill(x, y, color=colors[2-num], alpha=0.35, edgecolor=colors[2-num], label=labels[2-num])

plt.xlabel("Polar angle (°)", fontweight='bold')
plt.ylabel("Azimuth angle (°)", fontweight='bold')
plt.title("Angular Response (lower right)", fontweight='bold')
plt.grid(True)
plt.legend()
plt.tight_layout()

fig, ax = plt.subplots(figsize=(8, 6))
labels = np.array(["Blue (465nm)", "Green (532nm)", "Red (630nm)"])
for num in range(len(lmd)):
    points = np.column_stack(((th_out_fc[2-num,:,:]/deg).flatten(), (phi_out_fc[2-num,:,:]/deg).flatten()))
    alpha = 0.1  # smaller alpha = more detail
    shape = alphashape.alphashape(points, alpha)
    x, y = shape.exterior.xy
    ax.fill(x, y, color=colors[2-num], alpha=0.35, edgecolor=colors[2-num], label=labels[2-num])

plt.xlabel("Polar angle (°)", fontweight='bold')
plt.ylabel("Azimuth angle (°)", fontweight='bold')
plt.title("Angular Response (upper right)", fontweight='bold')
plt.grid(True)
plt.legend()
plt.tight_layout()

fig, ax = plt.subplots(figsize=(8, 6))
labels = np.array(["Blue (465nm)", "Green (532nm)", "Red (630nm)"])
for num in range(len(lmd)):
    points = np.column_stack(((th_out_ic2[2-num,:,:]/deg).flatten(), (phi_out_ic2[2-num,:,:]/deg).flatten()))
    alpha = 0.1  # smaller alpha = more detail
    shape = alphashape.alphashape(points, alpha)
    x, y = shape.exterior.xy
    ax.fill(x, y, color=colors[2-num], alpha=0.35, edgecolor=colors[2-num], label=labels[2-num])

plt.xlabel("Polar angle (°)", fontweight='bold')
plt.ylabel("Azimuth angle (°)", fontweight='bold')
plt.title("Angular Response (upper left)", fontweight='bold')
plt.grid(True)
plt.legend()
plt.tight_layout()

fig, ax = plt.subplots(figsize=(8, 6))
labels = np.array(["Blue (465nm)", "Green (532nm)", "Red (630nm)"])
for num in range(len(lmd)):
    points = np.column_stack(((th_out_oc[2-num,:,:]/deg).flatten(), (phi_out_oc[2-num,:,:]/deg).flatten()))
    alpha = 0.1  # smaller alpha = more detail
    shape = alphashape.alphashape(points, alpha)
    x, y = shape.exterior.xy
    ax.fill(x, y, color=colors[2-num], alpha=0.35, edgecolor=colors[2-num], label=labels[2-num])

plt.xlabel("Polar angle (°)", fontweight='bold')
plt.ylabel("Azimuth angle (°)", fontweight='bold')
plt.title("Angular Response (lower left)", fontweight='bold')
plt.grid(True)
plt.legend()
plt.tight_layout()

fig, ax = plt.subplots(figsize=(8, 6))
labels = np.array(["Blue (465nm)", "Green (532nm)", "Red (630nm)"])
x = np.linspace(-9,9,3)
y = np.linspace(-6.75,6.75,3)
[X,Y] = np.meshgrid(x,y)
points = np.column_stack((X.flatten(), Y.flatten()))
alpha = 0.1  # smaller alpha = more detail
shape = alphashape.alphashape(points, alpha)
x, y = shape.exterior.xy
ax.fill(x, y, color=colors[1], alpha=0.35, edgecolor=colors[1])

plt.xlabel(r'$FoV_x$' " (°)", fontweight='bold')
plt.ylabel(r'$FoV_y$' " (°)", fontweight='bold')
ax.set_xlim(-10.1, 10.1)
ax.set_ylim(-8.1, 8.1)
ax.set_xticks(np.arange(-10, 10.1, 2))
ax.set_yticks(np.arange(-8, 8.1, 2))
plt.title("Angular Response (Center)", fontweight='bold')
plt.grid(True)
plt.tight_layout()
ax.set_aspect('equal', adjustable='box')
plt.show()