import numpy as np
import time
import math
from numba import cuda, int32, float32, uint32
from couplers_coor import couplers_coor
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.colors import LogNorm
from scipy.io import savemat
from numba.cuda.random import xoroshiro128p_uniform_float32


def generate_points_in_polygon(polygon_vertices, num_points):
    """
    Generates a specified number of random points uniformly distributed inside a given polygon.

    Args:
        polygon_vertices (np.ndarray): A NumPy array of shape (n, 2) with the [x, y] coordinates
                                       of the polygon's vertices.
        num_points (int): The desired number of points to generate.

    Returns:
        np.ndarray: An array of shape (num_points, 2) containing the generated [x, y] points.
    """
    # Find the bounding box of the polygon
    xmin, ymin = np.min(polygon_vertices, axis=0)
    xmax, ymax = np.max(polygon_vertices, axis=0)
    
    # Create a Path object for efficient point-in-polygon checks
    polygon_path = Path(polygon_vertices)
    
    points_inside = []
    # Loop until the desired number of points is generated
    while len(points_inside) < num_points:
        # Generate a batch of random points within the bounding box
        # We generate more points than needed to account for rejections
        num_to_generate = (num_points - len(points_inside)) * 2
        random_points = np.random.uniform(low=[xmin, ymin], high=[xmax, ymax], size=(num_to_generate, 2))
        
        # Check which of the random points are inside the polygon
        is_inside = polygon_path.contains_points(random_points)
        
        # Append the valid points to our list
        points_inside.extend(random_points[is_inside])
        
    # Trim the list to exactly the number of points required and return as an array
    return np.array(points_inside[:num_points])


@cuda.jit(device=True)
def get_uniform_random_number(rng_states, index):
    # load state
    s = rng_states[index]

    # avoid the all-zero trap (optional reseed)
    if s == 0:
        s = uint32(0x6D2B79F5) ^ uint32(index + 1)

    # xorshift32 with 32-bit masking
    s ^= (s << 13) & uint32(0xFFFFFFFF)
    s ^= (s >> 17)
    s ^= (s << 5)  & uint32(0xFFFFFFFF)

    # store back
    rng_states[index] = s

    # scale to [0,1): multiply by 1/2**32
    return (s & uint32(0xFFFFFFFF)) * (1.0 / 4294967296.0)


@cuda.jit(device=True)
def is_inside_polygon(x, y, poly, start, end):
    """
    Checks if a point (x, y) is inside a polygon defined by a slice of 'poly'.
    """
    n_vertices = end - start # Calculate vertex count from bounds
    
    inside = False
    j = n_vertices - 1
    for i in range(n_vertices):
        # Access x and y coordinates directly from the main 'poly' array
        # using the 'start' offset.
        xi = poly[start + i, 0]
        yi = poly[start + i, 1]
        xj = poly[start + j, 0]
        yj = poly[start + j, 1]
        
        # The core ray-casting logic remains the same
        if ((yi > y) != (yj > y)) and \
           (x < (xj - xi) * (y - yi) / (yj - yi + 1e-20) + xi):
            inside = not inside
            
        j = i
    return inside


@cuda.jit(device=True)
def point_on_segment(px, py, poly, idx_a, idx_b, tol):
    """
    Checks if a point (px, py) is on the line segment between two points
    in the 'poly' array, identified by their indices.
    """
    # Look up the coordinates for the start and end of the segment
    x1 = poly[idx_a, 0]
    y1 = poly[idx_a, 1]
    x2 = poly[idx_b, 0]
    y2 = poly[idx_b, 1]
    
    # Check if the point is within the bounding box of the segment (with tolerance)
    if (px < min(x1, x2) - tol) or (px > max(x1, x2) + tol) or \
       (py < min(y1, y2) - tol) or (py > max(y1, y2) + tol):
        return False
        
    # Check if the point is collinear with the segment (with tolerance)
    # This checks if the area of the triangle formed by the three points is close to zero.
    return abs((x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)) <= tol


@cuda.jit(device=True)
def is_inside_or_on_edge(px, py, poly, start, end):
    n_vertices = end - start
    
    # Check if the point is on any of the polygon's edges
    # Note: We must also modify point_on_segment
    j = n_vertices - 1
    for i in range(n_vertices):
        # Pass the full array and indices to the helper function
        if point_on_segment(px, py, poly, start + j, start + i, 1e-12):
            return True
        j = i
        
    # Check if the point is inside the polygon
    # Note: We must also modify is_inside_polygon
    return is_inside_polygon(px, py, poly, start, end)


@cuda.jit(device=True)
def point_on_segment_4d(px, py, fov_array, m, n, idx_a, idx_b, tol):
    """
    Checks if a point (px, py) is on a line segment within a 4D polygon array.
    The segment is defined by vertex indices idx_a and idx_b for polygon (m, n).
    """
    # Look up the coordinates for the start of the segment
    x1 = fov_array[m, n, idx_a, 0]
    y1 = fov_array[m, n, idx_a, 1]
    
    # Look up the coordinates for the end of the segment
    x2 = fov_array[m, n, idx_b, 0]
    y2 = fov_array[m, n, idx_b, 1]
    
    # Check if the point is within the bounding box of the segment (with tolerance)
    if (px < min(x1, x2) - tol) or (px > max(x1, x2) + tol) or \
       (py < min(y1, y2) - tol) or (py > max(y1, y2) + tol):
        return False
        
    # Check if the point is collinear with the segment (cross-product check)
    return abs((x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)) <= tol


@cuda.jit(device=True)
def is_inside_polygon_4d(x, y, fov_array, m, n):
    """
    Checks if a point (x, y) is inside a polygon from a 4D array,
    identified by indices m and n.
    """
    n_vertices = fov_array.shape[2] # Get N from the shape (m, n, N, 2)
    
    inside = False
    j = n_vertices - 1
    for i in range(n_vertices):
        # Access x and y coordinates directly from the 4D array
        xi = fov_array[m, n, i, 0]
        yi = fov_array[m, n, i, 1]
        xj = fov_array[m, n, j, 0]
        yj = fov_array[m, n, j, 1]
        
        # Core ray-casting logic
        if ((yi > y) != (yj > y)) and \
           (x < (xj - xi) * (y - yi) / (yj - yi + 1e-20) + xi):
            inside = not inside
            
        j = i
        
    return inside


@cuda.jit(device=True)
def is_inside_or_on_edge_4d(px, py, fov_array, m, n):
    """
    Top-level check to see if a point is inside or on the edge of a polygon
    from a 4D array, identified by indices m and n.
    """
    n_vertices = fov_array.shape[2]
    
    # 1. Check if the point lies on any of the polygon's edges
    j = n_vertices - 1
    for i in range(n_vertices):
        # Pass the 4D array and all necessary indices to the helper
        if point_on_segment_4d(px, py, fov_array, m, n, j, i, 1e-12):
            return True
        j = i
        
    # 2. If not on an edge, check if it's strictly inside
    return is_inside_polygon_4d(px, py, fov_array, m, n)


@cuda.jit(device=True)
def LUT_find_position(lut, theta, phi):
    J = lut.shape[1]
    K = lut.shape[2]
    best_d2 = 1e-5

    for j in range(J):
        for k in range(K):
            dt = theta - lut[0, j, k]
            dp = phi - lut[1, j, k]
            d2 = dt*dt + dp*dp
            if d2 < best_d2:
                return j, k, True

    return -1, -1, False


@cuda.jit(device=True)
def _wrap_minus_pi_to_pi(x):
    """Wrap angle to (-pi, pi]."""
    two_pi = 2.0 * math.pi
    x = x + math.pi
    x = x - two_pi * math.floor(x / two_pi)
    x = x - math.pi
    return x


@cuda.jit(device=True)
def E_field_cal(
    Ete_abs, Etm_abs, delta_tem,
    E_te_te,  # TE_out from TE_in  (a)
    E_te_tm,  # TM_out from TE_in  (c)
    E_tm_te,  # TE_out from TM_in  (b)
    E_tm_tm   # TM_out from TM_in  (d)
):
    """
    Inputs:
    Ete_abs, Etm_abs : float
        Input field magnitudes |E_TE|, |E_TM|.
    delta_tem : float
        Input phase difference (phi_TM - phi_TE). Global TE phase taken as 0.
    E_te_te, E_te_tm, E_tm_te, E_tm_tm : complex
        1st-order transmission complex amplitudes:
          - E_te_te: TE_out for unit TE_in
          - E_te_tm: TM_out for unit TE_in
          - E_tm_te: TE_out for unit TM_in
          - E_tm_tm: TM_out for unit TM_in

    Returns:
    (Ete_out_abs, Etm_out_abs, delta_out) :
        |E_TE,out|, |E_TM,out|, and (phi_TM,out - phi_TE,out), wrapped to (-pi, pi].
    """
    # e^{i * delta_tem}
    phase = complex(math.cos(delta_tem), math.sin(delta_tem))

    # Input Jones vector: [E_TE; E_TM] with TE phase = 0, TM phase = delta_tem
    te_in = complex(Ete_abs, 0.0)
    tm_in = phase * Etm_abs

    # Transmission matrix (rows: [TE_out, TM_out], cols: [TE_in, TM_in])
    a = E_te_te
    b = E_tm_te
    c = E_te_tm
    d = E_tm_tm

    # Multiply: E_out = T * E_in
    Ete_out = a * te_in + b * tm_in   # TE component out
    Etm_out = c * te_in + d * tm_in   # TM component out

    # Magnitudes
    Ete_out_abs = math.hypot(Ete_out.real, Ete_out.imag)
    Etm_out_abs = math.hypot(Etm_out.real, Etm_out.imag)

    # Phases (guard tiny magnitudes)
    eps = 1e-20
    phi_te = math.atan2(Ete_out.imag, Ete_out.real) if Ete_out_abs >= eps else 0.0
    phi_tm = math.atan2(Etm_out.imag, Etm_out.real) if Etm_out_abs >= eps else 0.0

    # Output phase difference (TM - TE), wrapped to (-pi, pi]
    delta_out = _wrap_minus_pi_to_pi(phi_tm - phi_te)

    return Ete_out_abs, Etm_out_abs, delta_out


@cuda.jit(device=True)
def add_to_EB_atomic_val(matrix_EB, m, n, x, y,
                         xmin, xmax, ymin, ymax, value):
    Ny = matrix_EB.shape[2]
    Nx = matrix_EB.shape[3]

    dx = (xmax - xmin) / Nx
    dy = (ymax - ymin) / Ny

    ix = int(math.floor((x - xmin) / dx))
    iy = int(math.floor((y - ymin) / dy))

    cuda.atomic.add(matrix_EB, (n, m, iy, ix), value)
    return True

@cuda.jit
def zero_out_kernel(array):
    idx = cuda.grid(1)
    if idx < array.size:
        array.flat[idx] = 0

@cuda.jit
def reset_counter_kernel(counter):
    if cuda.threadIdx.x == 0 and cuda.blockIdx.x == 0:
        counter[0] = 0


@cuda.jit
def pack_active_to_front(src, dst, src_len, out_count):
    i = cuda.grid(1)
    if i >= src_len:
        return
    Ete = src[i, 8]
    Etm = src[i, 9]
    efficiency = Ete * Ete + Etm * Etm
    if src[i, 12] != 0 and efficiency > 0:
        pos = cuda.atomic.add(out_count, 0, 1)
        # write packed row at the front of dst
        for j in range(src.shape[1]):
            dst[pos, j] = src[i, j]


@cuda.jit
def process_rays_kernel(
    vectors, useful_count_in, d_total_ray_counter,
    MAX_STEPS, 
    IC, FC, FC_offset, OC, OC_offset,
    eff_reg1, eff_reg2, eff_reg_FOV, eff_reg_FOV_range,
    lut_ic1, lut_ic2, lut_fc1, lut_fc2, lut_oc,
    lut_TIR, lut_gap,
    matrix_EB):

    idx = cuda.grid(1)
    N = int(useful_count_in)

    if idx >= N:
        return
    
    if vectors[idx, 12] == 0:
        return
    
    # Read ray property in vectors
    x = vectors[idx, 0] # position of ray
    y = vectors[idx, 1]
    gap_x = vectors[idx, 2] # propagation distance for each TIR
    gap_y = vectors[idx, 3]
    theta = vectors[idx, 4] # polar and azimuth angle of ray (global coordinates)
    phi = vectors[idx, 5]
    m = int(vectors[idx, 6]) # LUT position
    n = int(vectors[idx, 7])
    Ete = vectors[idx, 8] # E-field
    Etm = vectors[idx, 9]
    delta_phase = vectors[idx, 10] # phase difference between te and tm (phi_tm - phi_te)  range: (-pi, pi]
    region_state = vectors[idx, 11] # region state
    flag = vectors[idx, 12] # 1: useful, 0: useless
    FOV_total = eff_reg_FOV.shape[0]

    if region_state == 0: # have not interacted with IC yet
        # check LUT and change propagation direction and electrical field here (TIR phase shift considered)
        # the parameters need to change: Es, Ep, delta_phase, theta, phi, gap_x, gap_y, energy
        theta = lut_ic2[m, n, 0]
        phi = lut_ic2[m, n, 1]
        Ete, Etm, delta_phase = E_field_cal(Ete, Etm, delta_phase,
                                            lut_ic1[m, n, 8], lut_ic1[m, n, 11], # unit TE in, TE&TM out (+1st tran)
                                            lut_ic1[m, n, 20], lut_ic1[m, n, 23]) # unit TM in, TE&TM out (+1st tran)
        delta_phase += lut_TIR[m, n, 0] # phase shift due to TIR (TM-TE)
        gap_x = lut_gap[m, n, 0] # assign the propagation distance
        gap_y = lut_gap[m, n, 1]
        x += gap_x
        y += gap_y
        region_state = 1
    
    if region_state == 1: # already in waveguide but still in IC region
        for _ in range(int(MAX_STEPS)):
            # check if the ray is still inside the IC. If not, enter the region 2 (exist IC and interact with FC)\
            if not is_inside_or_on_edge(x, y, IC, 0, IC.shape[0]): # two situations will occur: 1. enter FC region 2. does not enter FC yet
                # 1. have already entered FC region
                for i in range(len(FC_offset)-1): # check all the FC overlap
                    start_FC = FC_offset[i]
                    end_FC = FC_offset[i+1]
                    if is_inside_or_on_edge(x, y, FC, start_FC, end_FC):
                        # check LUT and change propagation direction and electrical field here (TIR phase shift considered)
                        # the parameters need to change: Es, Ep, delta_phase, theta, phi, gap_x, gap_y, energy
                        vectors[idx, 8], vectors[idx, 9], vectors[idx, 10] = E_field_cal(
                            Ete, Etm, delta_phase,
                            lut_fc1[i, m, n, 3], lut_fc1[i, m, n, 6],   # unit TE in, TE&TM out (0th ref)
                            lut_fc1[i, m, n, 15], lut_fc1[i, m, n, 18]) # unit TM in, TE&TM out (0th ref)
                        vectors[idx, 10] += lut_TIR[m, n, 0] # phase shift due to TIR (TM-TE)
                        vectors[idx, 0] = x + gap_x # position of ray
                        vectors[idx, 1] = y + gap_y
                        vectors[idx, 2] = gap_x # propagation distance for each TIR
                        vectors[idx, 3] = gap_y
                        vectors[idx, 4] = theta.real # polar and azimuth angle of ray (global coordinates)
                        vectors[idx, 5] = phi.real
                        vectors[idx, 6] = m # LUT position
                        vectors[idx, 7] = n
                        vectors[idx, 11] = 2 # region state
                        vectors[idx, 12] = 1 # 1: useful, 0: useless

                        ######################################################################
                        new_ray_index = cuda.atomic.add(d_total_ray_counter, 0, 1)
                        vectors[new_ray_index, 8], vectors[new_ray_index, 9], vectors[new_ray_index, 10] = E_field_cal(
                            Ete, Etm, delta_phase,
                            lut_fc1[i, m, n, 4], lut_fc1[i, m, n, 7],   # unit TE in, TE&TM out (+1st ref)
                            lut_fc1[i, m, n, 16], lut_fc1[i, m, n, 19]) # unit TM in, TE&TM out (+1st ref)
                        vectors[new_ray_index, 10] += lut_TIR[m, n, 1] # phase shift due to TIR (TM-TE)
                        vectors[new_ray_index, 0] = x + lut_gap[m, n, 2] # position of ray
                        vectors[new_ray_index, 1] = y + lut_gap[m, n, 3]
                        vectors[new_ray_index, 2] = lut_gap[m, n, 2] # propagation distance for each TIR
                        vectors[new_ray_index, 3] = lut_gap[m, n, 3]
                        vectors[new_ray_index, 4] = lut_fc2[i, m, n, 0].real # polar and azimuth angle of ray (global coordinates)
                        vectors[new_ray_index, 5] = lut_fc2[i, m, n, 1].real
                        vectors[new_ray_index, 6] = m # LUT position
                        vectors[new_ray_index, 7] = n
                        vectors[new_ray_index, 11] = 3 # region state (change to 3, +1st order)
                        vectors[new_ray_index, 12] = 1 # 1: useful, 0: useless
                        return
                # 2. have not entered FC region yet
                delta_phase += 2*lut_TIR[m, n, 0] # phase shift due to TIR (TM-TE) 2 times
                x += gap_x
                y += gap_y
            else: # still inside IC
                # check LUT, change propagation direction and electrical field here (TIR phase shift considered)
                # the parameters need to change: Es, Ep, delta_phase, theta, phi, gap_x, gap_y, energy
                Ete, Etm, delta_phase = E_field_cal(
                    Ete, Etm, delta_phase,
                    lut_ic2[m, n, 3], lut_ic2[m, n, 6],   # unit TE in, TE&TM out (0th ref)
                    lut_ic2[m, n, 15], lut_ic2[m, n, 18]) # unit TM in, TE&TM out (0th ref)
                delta_phase += lut_TIR[m, n, 0] # phase shift due to TIR (TM-TE)
                x += gap_x
                y += gap_y
        # if still region==1 after loop, we didn’t exit IC in MAX_STEPS -> treat as lost
        if region_state == 1:
            vectors[idx, 12] = 0
            return

    if region_state == 2 or region_state == 3: # exist IC and interact with FC
        # check if the ray is still inside the entire effective region. If not, set flag to 0 (waisted ray).
        if not is_inside_or_on_edge(x, y, eff_reg1, 0, eff_reg1.shape[0]):
            # mark output inactive and exit
            vectors[idx, 12] = 0
            return

        for _ in range(int(MAX_STEPS)):
            for i in range(len(FC_offset)-1):
                start_FC = FC_offset[i]
                end_FC = FC_offset[i+1]
                if is_inside_or_on_edge(x, y, FC, start_FC, end_FC):
                    # check LUT and change propagation direction and electrical field here (TIR phase shift considered)
                    # the parameters need to change: Es, Ep, delta_phase, theta, phi, gap_x, gap_y, energy
                    if region_state == 2: # propagation direction is along FC
                        vectors[idx, 8], vectors[idx, 9], vectors[idx, 10] = E_field_cal(
                            Ete, Etm, delta_phase,
                            lut_fc1[i, m, n, 3], lut_fc1[i, m, n, 6],   # unit TE in, TE&TM out (0th ref)
                            lut_fc1[i, m, n, 15], lut_fc1[i, m, n, 18]) # unit TM in, TE&TM out (0th ref)
                        vectors[idx, 10] += lut_TIR[m, n, 0] # phase shift due to TIR (TM-TE)
                        vectors[idx, 0] = x + gap_x # position of ray
                        vectors[idx, 1] = y + gap_y
                        vectors[idx, 2] = gap_x # propagation distance for each TIR
                        vectors[idx, 3] = gap_y
                        vectors[idx, 4] = theta.real # polar and azimuth angle of ray (global coordinates)
                        vectors[idx, 5] = phi.real
                        vectors[idx, 6] = m # LUT position
                        vectors[idx, 7] = n
                        vectors[idx, 11] = region_state # region state (keep on 2, 0th order)
                        vectors[idx, 12] = 1 # 1: useful, 0: useless

                        ######################################################################
                        new_ray_index = cuda.atomic.add(d_total_ray_counter, 0, 1)
                        vectors[new_ray_index, 8], vectors[new_ray_index, 9], vectors[new_ray_index, 10] = E_field_cal(
                            Ete, Etm, delta_phase,
                            lut_fc1[i, m, n, 4], lut_fc1[i, m, n, 7],   # unit TE in, TE&TM out (+1st ref)
                            lut_fc1[i, m, n, 16], lut_fc1[i, m, n, 19]) # unit TM in, TE&TM out (+1st ref)
                        vectors[new_ray_index, 10] += lut_TIR[m, n, 1] # phase shift due to TIR (TM-TE)
                        vectors[new_ray_index, 0] = x + lut_gap[m, n, 2] # position of ray
                        vectors[new_ray_index, 1] = y + lut_gap[m, n, 3]
                        vectors[new_ray_index, 2] = lut_gap[m, n, 2] # propagation distance for each TIR
                        vectors[new_ray_index, 3] = lut_gap[m, n, 3]
                        vectors[new_ray_index, 4] = lut_fc2[i, m, n, 0].real # polar and azimuth angle of ray (global coordinates)
                        vectors[new_ray_index, 5] = lut_fc2[i, m, n, 1].real
                        vectors[new_ray_index, 6] = m # LUT position
                        vectors[new_ray_index, 7] = n
                        vectors[new_ray_index, 11] = 3 # region state (change to 3, +1st order)
                        vectors[new_ray_index, 12] = 1 # 1: useful, 0: useless
                        return
                    
                    elif region_state == 3: # propagation direction is to OC
                        vectors[idx, 8], vectors[idx, 9], vectors[idx, 10] = E_field_cal(
                            Ete, Etm, delta_phase,
                            lut_fc2[i, m, n, 3], lut_fc2[i, m, n, 6],   # unit TE in, TE&TM out (0th ref)
                            lut_fc2[i, m, n, 15], lut_fc2[i, m, n, 18]) # unit TM in, TE&TM out (0th ref)
                        vectors[idx, 10] += lut_TIR[m, n, 1] # phase shift due to TIR (TM-TE)
                        vectors[idx, 0] = x + gap_x # position of ray
                        vectors[idx, 1] = y + gap_y
                        vectors[idx, 2] = gap_x # propagation distance for each TIR
                        vectors[idx, 3] = gap_y
                        vectors[idx, 4] = theta.real # polar and azimuth angle of ray (global coordinates)
                        vectors[idx, 5] = phi.real
                        vectors[idx, 6] = m # LUT position
                        vectors[idx, 7] = n
                        vectors[idx, 11] = region_state # region state (keep on 3, 0th order)
                        vectors[idx, 12] = 1 # 1: useful, 0: useless

                        ######################################################################
                        new_ray_index = cuda.atomic.add(d_total_ray_counter, 0, 1)
                        vectors[new_ray_index, 8], vectors[new_ray_index, 9], vectors[new_ray_index, 10] = E_field_cal(
                            Ete, Etm, delta_phase,
                            lut_fc2[i, m, n, 2], lut_fc2[i, m, n, 5],   # unit TE in, TE&TM out (-1st ref)
                            lut_fc2[i, m, n, 14], lut_fc2[i, m, n, 17]) # unit TM in, TE&TM out (-1st ref)
                        vectors[new_ray_index, 10] += lut_TIR[m, n, 0] # phase shift due to TIR (TM-TE)
                        vectors[new_ray_index, 0] = x + lut_gap[m, n, 0] # position of ray
                        vectors[new_ray_index, 1] = y + lut_gap[m, n, 1]
                        vectors[new_ray_index, 2] = lut_gap[m, n, 0] # propagation distance for each TIR
                        vectors[new_ray_index, 3] = lut_gap[m, n, 1]
                        vectors[new_ray_index, 4] = lut_fc1[i, m, n, 0].real # polar and azimuth angle of ray (global coordinates)
                        vectors[new_ray_index, 5] = lut_fc1[i, m, n, 1].real
                        vectors[new_ray_index, 6] = m # LUT position
                        vectors[new_ray_index, 7] = n
                        vectors[new_ray_index, 11] = 2 # region state (change to 2, -1st order)
                        vectors[new_ray_index, 12] = 1 # 1: useful, 0: useless
                        return
            # There is no overlap with any FC ("hit = False"), which means the ray has not enter the FC yet. 
            # Continue propagate and consider TIR phase shift:
            # TIR phase shift (twice):
            if not is_inside_or_on_edge(x, y, eff_reg2, 0, eff_reg2.shape[0]):
                if region_state == 3:
                    region_state = 4
                    break
                else:
                    vectors[idx, 12] = 0
                    return
            else:
                delta_phase += 2*lut_TIR[m, n, 0] # phase shift due to TIR (TM-TE) 2 times
                x += gap_x
                y += gap_y

    
    if region_state == 4:  # exist FC and start propagate to OC
        for _ in range(int(MAX_STEPS)):
            if not is_inside_or_on_edge(x, y, eff_reg1, 0, eff_reg1.shape[0]):
                vectors[idx, 12] = 0
                return

            hit = False
            for i in range(len(OC_offset) - 1):
                start_OC = OC_offset[i]
                end_OC = OC_offset[i+1]
                if is_inside_or_on_edge(x, y, OC, start_OC, end_OC):
                    hit = True
                    # If inside the FOV region for this FOV_num, accumulate to EB
                    if is_inside_or_on_edge_4d(x, y, eff_reg_FOV, m, n):
                        Ete_out, Etm_out, _ = E_field_cal(
                            Ete, Etm, delta_phase,
                            lut_oc[i, m, n, 10], lut_oc[i, m, n, 13],  # +1st tran, TE in -> TE/TM
                            lut_oc[i, m, n, 22], lut_oc[i, m, n, 25])  # +1st tran, TM in -> TE/TM
                        # squared magnitude (no sqrt)
                        efficiency = Ete_out * Ete_out + Etm_out * Etm_out

                        if efficiency > 0:
                            # Add atomically into the EB matrix bin for this FOV
                            xmin = eff_reg_FOV_range[m, n, 0]
                            xmax = eff_reg_FOV_range[m, n, 1]
                            ymin = eff_reg_FOV_range[m, n, 2]
                            ymax = eff_reg_FOV_range[m, n, 3]
                            add_to_EB_atomic_val(matrix_EB, m, n, x, y, xmin, xmax, ymin, ymax, efficiency)

                    # Then continue propagation with 0th order
                    Ete, Etm, delta_phase = E_field_cal(
                        Ete, Etm, delta_phase,
                        lut_oc[i, m, n, 3],  lut_oc[i, m, n, 6],   # 0th ref
                        lut_oc[i, m, n, 15], lut_oc[i, m, n, 18])  # 0th ref
                    delta_phase += lut_TIR[m, n, 1]
                    x += gap_x
                    y += gap_y
                    efficiency = Ete * Ete + Etm * Etm
                    if efficiency < 0:
                        vectors[idx, 12] = 0
                        return
                    break

            # have not enter OC yet, continue propagate
            if not hit:
                delta_phase += 2 * lut_TIR[m, n, 1]
                x += gap_x
                y += gap_y


@cuda.jit
def process_rays_kernel2(
    vectors, useful_count_in, d_total_ray_counter, 
    IC, FC, FC_offset, OC, OC_offset,
    eff_reg1, eff_reg2, eff_reg_FOV, eff_reg_FOV_range,
    lut_ic1, lut_ic2, lut_ic3, lut_fc1, lut_fc2, lut_oc1, lut_oc2,
    lut_TIR, lut_gap,
    matrix_EB):

    idx = cuda.grid(1)
    N = int(useful_count_in)

    if idx >= N:
        return
    
    # Read ray property in vectors
    x = vectors[idx, 0] # position of ray
    y = vectors[idx, 1]
    gap_x = vectors[idx, 2] # propagation distance for each TIR
    gap_y = vectors[idx, 3]
    theta = vectors[idx, 4] # polar and azimuth angle of ray (global coordinates)
    phi = vectors[idx, 5]
    m = int(vectors[idx, 6]) # LUT position
    n = int(vectors[idx, 7])
    Ete = vectors[idx, 8] # E-field
    Etm = vectors[idx, 9]
    delta_phase = vectors[idx, 10] # phase difference between te and tm (phi_tm - phi_te)  range: (-pi, pi]
    region_state = vectors[idx, 11] # region state

    threshold_IC = 1e-3
    threshold = 1e-6
    MAX_STEPS = 50
    n_g = 1.52

    if region_state == 0: # have not interact with IC yet
        # check LUT and change propagation direction and electrical field here (TIR phase shift considered)
        # the parameters need to change: Es, Ep, delta_phase, theta, phi, gap_x, gap_y, energy
        # order to FC direction
        theta = lut_ic2[m, n, 0]
        phi = lut_ic2[m, n, 1]
        Ete1, Etm1, delta_phase1 = E_field_cal(
            Ete, Etm, delta_phase,
            lut_ic1[m, n, 13], lut_ic1[m, n, 18], # unit TE in, TE&TM out (-1st tran) (to FC)
            lut_ic1[m, n, 33], lut_ic1[m, n, 38]) # unit TM in, TE&TM out (-1st tran) (to FC)
        delta_phase1 += lut_TIR[m, n, 0] # phase shift due to TIR (TM-TE)
        gap_x = lut_gap[m, n, 0] # assign the propagation distance
        gap_y = lut_gap[m, n, 1]
        x1 = x + gap_x
        y1 = y + gap_y
        if not is_inside_or_on_edge(x1, y1, IC, 0, IC.shape[0]): # two situations will occur: 1. exist IC region 2. does not exist IC yet
            vectors[idx, 0] = x1 # position
            vectors[idx, 1] = y1
            vectors[idx, 2] = gap_x # propagation direction
            vectors[idx, 3] = gap_y
            vectors[idx, 4] = theta.real # polar and azimuth angles
            vectors[idx, 5] = phi.real
            vectors[idx, 6] = m # FOV
            vectors[idx, 7] = n
            vectors[idx, 8] = Ete1 # E-field
            vectors[idx, 9] = Etm1
            vectors[idx, 10] = delta_phase1
            vectors[idx, 11] = 3 # region state
            vectors[idx, 12] = 1 # flag
        else: # still inside IC (order to FC)
            vectors[idx, 0] = x1 # position
            vectors[idx, 1] = y1
            vectors[idx, 2] = gap_x # propagation direction
            vectors[idx, 3] = gap_y 
            vectors[idx, 4] = theta.real # polar and azimuth angles
            vectors[idx, 5] = phi.real
            vectors[idx, 6] = m # FOV
            vectors[idx, 7] = n
            vectors[idx, 8] = Ete1 # E-field
            vectors[idx, 9] = Etm1
            vectors[idx, 10] = delta_phase1
            vectors[idx, 11] = 1 # region state
            vectors[idx, 12] = 1 # flag
        
        # order to opposite FC direction
        theta = lut_ic3[m, n, 0]
        phi = lut_ic3[m, n, 1]
        gap_x = lut_gap[m, n, 4] # assign the propagation distance
        gap_y = lut_gap[m, n, 5]
        x2 = x + gap_x
        y2 = y + gap_y
        if is_inside_or_on_edge(x2, y2, IC, 0, IC.shape[0]): # two situations will occur: 1. exist IC region (waisted) 2. does not exist IC yet
            Ete2, Etm2, delta_phase2 = E_field_cal(
                Ete, Etm, delta_phase,
                lut_ic1[m, n, 15], lut_ic1[m, n, 20], # unit TE in, TE&TM out (+1st tran) (opposite to FC)
                lut_ic1[m, n, 35], lut_ic1[m, n, 40]) # unit TM in, TE&TM out (+1st tran) (opposite to FC)
            delta_phase2 += lut_TIR[m, n, 2] # phase shift due to TIR (TM-TE)
            new_ray_index = cuda.atomic.add(d_total_ray_counter, 0, 1)
            vectors[new_ray_index, 0] = x2 # position
            vectors[new_ray_index, 1] = y2
            vectors[new_ray_index, 2] = gap_x # propagation direction
            vectors[new_ray_index, 3] = gap_y 
            vectors[new_ray_index, 4] = theta.real # polar and azimuth angles
            vectors[new_ray_index, 5] = phi.real
            vectors[new_ray_index, 6] = m # FOV
            vectors[new_ray_index, 7] = n
            vectors[new_ray_index, 8] = Ete2 # E-field
            vectors[new_ray_index, 9] = Etm2
            vectors[new_ray_index, 10] = delta_phase2
            vectors[new_ray_index, 11] = 2 # region state
            vectors[new_ray_index, 12] = 1 # flag
            return

    if region_state == 1: # propagate to FC direction
        Ete1, Etm1, delta_phase1 = E_field_cal(
            Ete, Etm, delta_phase,
            lut_ic2[m, n, 4], lut_ic2[m, n, 9], # unit TE in, TE&TM out (0th ref) (to FC)
            lut_ic2[m, n, 24], lut_ic2[m, n, 29]) # unit TM in, TE&TM out (0th ref) (to FC)
        delta_phase1 += lut_TIR[m, n, 0] # phase shift due to TIR (TM-TE)
        x1 = x + gap_x
        y1 = y + gap_y
        if not is_inside_or_on_edge(x1, y1, IC, 0, IC.shape[0]): # two situations will occur: 1. exist IC region 2. does not exist IC yet
            efficiency = (Ete1 * Ete1 + Etm1 * Etm1) * math.cos(lut_ic2[m, n, 0].real) / math.cos(lut_ic1[m, n, 0].real) * n_g
            if efficiency < threshold_IC:
                vectors[idx, 12] = 0 # flag
            else:
                vectors[idx, 0] = x1 # position
                vectors[idx, 1] = y1
                vectors[idx, 2] = gap_x # propagation direction
                vectors[idx, 3] = gap_y
                vectors[idx, 4] = theta.real # polar and azimuth angles
                vectors[idx, 5] = phi.real
                vectors[idx, 6] = m # FOV
                vectors[idx, 7] = n
                vectors[idx, 8] = Ete1 # E-field
                vectors[idx, 9] = Etm1
                vectors[idx, 10] = delta_phase1
                vectors[idx, 11] = 3 # region state
                vectors[idx, 12] = 1 # flag
        else: # still inside IC (order to FC)
            efficiency = (Ete1 * Ete1 + Etm1 * Etm1) * math.cos(lut_ic2[m, n, 0].real) / math.cos(lut_ic1[m, n, 0].real) * n_g
            if efficiency < threshold_IC:
                vectors[idx, 12] = 0 # flag
            else:
                vectors[idx, 0] = x1 # position
                vectors[idx, 1] = y1
                vectors[idx, 2] = gap_x # propagation direction
                vectors[idx, 3] = gap_y 
                vectors[idx, 4] = theta.real # polar and azimuth angles
                vectors[idx, 5] = phi.real
                vectors[idx, 6] = m # FOV
                vectors[idx, 7] = n
                vectors[idx, 8] = Ete1 # E-field
                vectors[idx, 9] = Etm1
                vectors[idx, 10] = delta_phase1
                vectors[idx, 11] = 1 # region state
                vectors[idx, 12] = 1 # flag
        
        # order to opposite FC direction
        theta = lut_ic3[m, n, 0]
        phi = lut_ic3[m, n, 1]
        gap_x = lut_gap[m, n, 4] # assign the propagation distance
        gap_y = lut_gap[m, n, 5]
        x2 = x + gap_x
        y2 = y + gap_y
        if is_inside_or_on_edge(x2, y2, IC, 0, IC.shape[0]): # two situations will occur: 1. exist IC region 2. does not exist IC yet
            Ete2, Etm2, delta_phase2 = E_field_cal(
                Ete, Etm, delta_phase,
                lut_ic2[m, n, 6], lut_ic2[m, n, 11], # unit TE in, TE&TM out (+2nd ref) (opposite to FC)
                lut_ic2[m, n, 26], lut_ic2[m, n, 31]) # unit TM in, TE&TM out (+2nd ref) (opposite to FC)
            efficiency = (Ete2 * Ete2 + Etm2 * Etm2) * math.cos(lut_ic3[m, n, 0].real) / math.cos(lut_ic1[m, n, 0].real) * n_g
            if efficiency < threshold_IC:
                return
            delta_phase2 += lut_TIR[m, n, 2] # phase shift due to TIR (TM-TE)
            new_ray_index = cuda.atomic.add(d_total_ray_counter, 0, 1)
            vectors[new_ray_index, 0] = x2 # position
            vectors[new_ray_index, 1] = y2
            vectors[new_ray_index, 2] = gap_x # propagation direction
            vectors[new_ray_index, 3] = gap_y 
            vectors[new_ray_index, 4] = theta.real # polar and azimuth angles
            vectors[new_ray_index, 5] = phi.real
            vectors[new_ray_index, 6] = m # FOV
            vectors[new_ray_index, 7] = n
            vectors[new_ray_index, 8] = Ete2 # E-field
            vectors[new_ray_index, 9] = Etm2
            vectors[new_ray_index, 10] = delta_phase2
            vectors[new_ray_index, 11] = 2 # region state
            vectors[new_ray_index, 12] = 1 # flag
            return
    
    if region_state == 2: # propagate to opposite FC direction
        theta = lut_ic3[m, n, 0]
        phi = lut_ic3[m, n, 1]
        Ete1, Etm1, delta_phase1 = E_field_cal(
            Ete, Etm, delta_phase,
            lut_ic3[m, n, 4], lut_ic3[m, n, 9], # unit TE in, TE&TM out (0th ref) (opposite to FC)
            lut_ic3[m, n, 24], lut_ic3[m, n, 29]) # unit TM in, TE&TM out (0th ref) (opposite to FC)
        delta_phase1 += lut_TIR[m, n, 2] # phase shift due to TIR (TM-TE)
        gap_x = lut_gap[m, n, 4] # assign the propagation distance
        gap_y = lut_gap[m, n, 5]
        x1 = x + gap_x
        y1 = y + gap_y
        if is_inside_or_on_edge(x1, y1, IC, 0, IC.shape[0]): # two situations will occur: 1. exist IC region 2. does not exist IC yet
            efficiency = (Ete1 * Ete1 + Etm1 * Etm1) * math.cos(lut_ic3[m, n, 0].real) / math.cos(lut_ic1[m, n, 0].real) * n_g
            if efficiency < threshold_IC:
                vectors[idx, 12] = 0 # flag
            else:
                vectors[idx, 0] = x1 # position
                vectors[idx, 1] = y1
                vectors[idx, 2] = gap_x # propagation direction
                vectors[idx, 3] = gap_y 
                vectors[idx, 4] = theta.real # polar and azimuth angles
                vectors[idx, 5] = phi.real
                vectors[idx, 6] = m # FOV
                vectors[idx, 7] = n
                vectors[idx, 8] = Ete1 # E-field
                vectors[idx, 9] = Etm1
                vectors[idx, 10] = delta_phase1
                vectors[idx, 11] = 2 # region state
                vectors[idx, 12] = 1 # flag
        else:
            vectors[idx, 12] = 0 # flag
        
        # order to FC direction
        theta = lut_ic2[m, n, 0]
        phi = lut_ic2[m, n, 1]
        gap_x = lut_gap[m, n, 0] # assign the propagation distance
        gap_y = lut_gap[m, n, 1]
        x2 = x + gap_x
        y2 = y + gap_y
        if is_inside_or_on_edge(x2, y2, IC, 0, IC.shape[0]): # two situations will occur: 1. exist IC region 2. does not exist IC yet
            Ete2, Etm2, delta_phase2 = E_field_cal(
                Ete, Etm, delta_phase,
                lut_ic3[m, n, 2], lut_ic3[m, n, 22], # unit TE in, TE&TM out (-2nd ref) (to FC)
                lut_ic3[m, n, 7], lut_ic3[m, n, 27]) # unit TM in, TE&TM out (-2nd ref) (to FC)
            efficiency = (Ete2 * Ete2 + Etm2 * Etm2) * math.cos(lut_ic2[m, n, 0].real) / math.cos(lut_ic1[m, n, 0].real) * n_g
            if efficiency < threshold_IC:
                return
            delta_phase2 += lut_TIR[m, n, 0] # phase shift due to TIR (TM-TE)
            new_ray_index = cuda.atomic.add(d_total_ray_counter, 0, 1)
            vectors[new_ray_index, 0] = x2 # position
            vectors[new_ray_index, 1] = y2
            vectors[new_ray_index, 2] = gap_x # propagation direction
            vectors[new_ray_index, 3] = gap_y 
            vectors[new_ray_index, 4] = theta.real # polar and azimuth angles
            vectors[new_ray_index, 5] = phi.real
            vectors[new_ray_index, 6] = m # FOV
            vectors[new_ray_index, 7] = n
            vectors[new_ray_index, 8] = Ete2 # E-field
            vectors[new_ray_index, 9] = Etm2
            vectors[new_ray_index, 10] = delta_phase2
            vectors[new_ray_index, 11] = 1 # region state
            vectors[new_ray_index, 12] = 1 # flag
            return
        else:
            Ete2, Etm2, delta_phase2 = E_field_cal(
                Ete, Etm, delta_phase,
                lut_ic3[m, n, 2], lut_ic3[m, n, 22], # unit TE in, TE&TM out (-2nd ref) (to FC)
                lut_ic3[m, n, 7], lut_ic3[m, n, 27]) # unit TM in, TE&TM out (-2nd ref) (to FC)
            efficiency = (Ete2 * Ete2 + Etm2 * Etm2) * math.cos(lut_ic2[m, n, 0].real) / math.cos(lut_ic1[m, n, 0].real) * n_g
            if efficiency < threshold_IC:
                return
            delta_phase2 += lut_TIR[m, n, 0] # phase shift due to TIR (TM-TE)
            new_ray_index = cuda.atomic.add(d_total_ray_counter, 0, 1)
            vectors[new_ray_index, 0] = x2 # position
            vectors[new_ray_index, 1] = y2
            vectors[new_ray_index, 2] = gap_x # propagation direction
            vectors[new_ray_index, 3] = gap_y 
            vectors[new_ray_index, 4] = theta.real # polar and azimuth angles
            vectors[new_ray_index, 5] = phi.real
            vectors[new_ray_index, 6] = m # FOV
            vectors[new_ray_index, 7] = n
            vectors[new_ray_index, 8] = Ete2 # E-field
            vectors[new_ray_index, 9] = Etm2
            vectors[new_ray_index, 10] = delta_phase2
            vectors[new_ray_index, 11] = 3 # region state
            vectors[new_ray_index, 12] = 1 # flag

    if region_state == 3 or region_state == 4: # exist IC and interact with FC
        # check if the ray is still inside the entire effective region. If not, set flag to 0 (waisted ray).
        if not is_inside_or_on_edge(x, y, eff_reg1, 0, eff_reg1.shape[0]):
            # mark output inactive and exit
            vectors[idx, 12] = 0
            return

        for _ in range(int(MAX_STEPS)):
            for i in range(len(FC_offset)-1):
                start_FC = FC_offset[i]
                end_FC = FC_offset[i+1]
                if is_inside_or_on_edge(x, y, FC, start_FC, end_FC):
                    # check LUT and change propagation direction and electrical field here (TIR phase shift considered)
                    # the parameters need to change: Es, Ep, delta_phase, theta, phi, gap_x, gap_y, energy
                    if region_state == 3: # propagation direction is along FC
                        Ete1, Etm1, delta_phase1 = E_field_cal(
                            Ete, Etm, delta_phase,
                            lut_fc1[i, m, n, 3], lut_fc1[i, m, n, 6],   # unit TE in, TE&TM out (0th ref)
                            lut_fc1[i, m, n, 15], lut_fc1[i, m, n, 18]) # unit TM in, TE&TM out (0th ref)
                        efficiency = (Ete1 * Ete1 + Etm1 * Etm1) * math.cos(lut_fc1[i, m, n, 0].real) / math.cos(lut_ic1[m, n, 0].real) * n_g
                        if efficiency < threshold:
                            vectors[idx, 12] = 0 # flag
                        else:
                            vectors[idx, 0] = x + gap_x # position of ray
                            vectors[idx, 1] = y + gap_y
                            vectors[idx, 2] = gap_x # propagation distance for each TIR
                            vectors[idx, 3] = gap_y
                            vectors[idx, 4] = theta.real # polar and azimuth angle of ray (global coordinates)
                            vectors[idx, 5] = phi.real
                            vectors[idx, 6] = m # LUT position
                            vectors[idx, 7] = n
                            vectors[idx, 8] = Ete1
                            vectors[idx, 9] = Etm1
                            vectors[idx, 10] = delta_phase1 + lut_TIR[m, n, 0] # phase shift due to TIR (TM-TE)
                            vectors[idx, 11] = 3 # region state (keep on 3, 0th order)
                            vectors[idx, 12] = 1 # 1: useful, 0: useless

                        ######################################################################
                        Ete2, Etm2, delta_phase2 = E_field_cal(
                            Ete, Etm, delta_phase,
                            lut_fc1[i, m, n, 2], lut_fc1[i, m, n, 5],   # unit TE in, TE&TM out (-1st ref)
                            lut_fc1[i, m, n, 14], lut_fc1[i, m, n, 17]) # unit TM in, TE&TM out (-1st ref)
                        efficiency = (Ete2 * Ete2 + Etm2 * Etm2) * math.cos(lut_fc2[i, m, n, 0].real) / math.cos(lut_ic1[m, n, 0].real) * n_g
                        if efficiency < threshold:
                            return
                        new_ray_index = cuda.atomic.add(d_total_ray_counter, 0, 1)
                        vectors[new_ray_index, 0] = x + lut_gap[m, n, 2] # position of ray
                        vectors[new_ray_index, 1] = y + lut_gap[m, n, 3]
                        vectors[new_ray_index, 2] = lut_gap[m, n, 2] # propagation distance for each TIR
                        vectors[new_ray_index, 3] = lut_gap[m, n, 3]
                        vectors[new_ray_index, 4] = lut_fc2[i, m, n, 0].real # polar and azimuth angle of ray (global coordinates)
                        vectors[new_ray_index, 5] = lut_fc2[i, m, n, 1].real
                        vectors[new_ray_index, 6] = m # LUT position
                        vectors[new_ray_index, 7] = n
                        vectors[new_ray_index, 8] = Ete2
                        vectors[new_ray_index, 9] = Etm2
                        vectors[new_ray_index, 10] = delta_phase2 + lut_TIR[m, n, 1] # phase shift due to TIR (TM-TE)
                        vectors[new_ray_index, 11] = 4 # region state (change to 4, -1st order)
                        vectors[new_ray_index, 12] = 1 # 1: useful, 0: useless
                        return
                    
                    elif region_state == 4: # propagation direction is to OC
                        Ete1, Etm1, delta_phase1 = E_field_cal(
                            Ete, Etm, delta_phase,
                            lut_fc2[i, m, n, 3], lut_fc2[i, m, n, 6],   # unit TE in, TE&TM out (0th ref)
                            lut_fc2[i, m, n, 15], lut_fc2[i, m, n, 18]) # unit TM in, TE&TM out (0th ref)
                        efficiency = (Ete1 * Ete1 + Etm1 * Etm1) * math.cos(lut_fc2[i, m, n, 0].real) / math.cos(lut_ic1[m, n, 0].real) * n_g
                        if efficiency < threshold:
                            vectors[idx, 12] = 0 # flag
                        else:
                            vectors[idx, 0] = x + gap_x # position of ray
                            vectors[idx, 1] = y + gap_y
                            vectors[idx, 2] = gap_x # propagation distance for each TIR
                            vectors[idx, 3] = gap_y
                            vectors[idx, 4] = theta.real # polar and azimuth angle of ray (global coordinates)
                            vectors[idx, 5] = phi.real
                            vectors[idx, 6] = m # LUT position
                            vectors[idx, 7] = n
                            vectors[idx, 8] = Ete1
                            vectors[idx, 9] = Etm1
                            vectors[idx, 10] = delta_phase1 + lut_TIR[m, n, 1] # phase shift due to TIR (TM-TE)
                            vectors[idx, 11] = 4 # region state (keep on 4, 0th order)
                            vectors[idx, 12] = 1 # 1: useful, 0: useless

                        ######################################################################
                        Ete2, Etm2, delta_phase2 = E_field_cal(
                            Ete, Etm, delta_phase,
                            lut_fc2[i, m, n, 4], lut_fc2[i, m, n, 7],   # unit TE in, TE&TM out (+1st ref)
                            lut_fc2[i, m, n, 16], lut_fc2[i, m, n, 19]) # unit TM in, TE&TM out (+1st ref)
                        efficiency = (Ete2 * Ete2 + Etm2 * Etm2) * math.cos(lut_fc1[i, m, n, 0].real) / math.cos(lut_ic1[m, n, 0].real) * n_g
                        if efficiency < threshold:
                            return
                        new_ray_index = cuda.atomic.add(d_total_ray_counter, 0, 1)
                        vectors[new_ray_index, 0] = x + lut_gap[m, n, 0] # position of ray
                        vectors[new_ray_index, 1] = y + lut_gap[m, n, 1]
                        vectors[new_ray_index, 2] = lut_gap[m, n, 0] # propagation distance for each TIR
                        vectors[new_ray_index, 3] = lut_gap[m, n, 1]
                        vectors[new_ray_index, 4] = lut_fc1[i, m, n, 0].real # polar and azimuth angle of ray (global coordinates)
                        vectors[new_ray_index, 5] = lut_fc1[i, m, n, 1].real
                        vectors[new_ray_index, 6] = m # LUT position
                        vectors[new_ray_index, 7] = n
                        vectors[new_ray_index, 8] = Ete2
                        vectors[new_ray_index, 9] = Etm2
                        vectors[new_ray_index, 10] = delta_phase2 + lut_TIR[m, n, 1] # phase shift due to TIR (TM-TE)
                        vectors[new_ray_index, 11] = 3 # region state (change to 3, +1st order)
                        vectors[new_ray_index, 12] = 1 # 1: useful, 0: useless
                        return
            # There is no overlap with any FC ("hit = False"), which means the ray 1. has not enter the FC yet or 2. has left the FC. 
            if not is_inside_or_on_edge(x, y, eff_reg2, 0, eff_reg2.shape[0]):
                if region_state == 4:
                    region_state = 5
                    break
                else:
                    vectors[idx, 12] = 0
                    return
            # Continue propagate and consider TIR phase shift:
            # TIR phase shift (twice):
            else:
                delta_phase += 2*lut_TIR[m, n, 0] # phase shift due to TIR (TM-TE) 2 times
                x += gap_x
                y += gap_y

    if region_state == 5 or region_state == 6:  # exist FC and interact with OC
        for _ in range(int(MAX_STEPS)):
            if not is_inside_or_on_edge(x, y, eff_reg1, 0, eff_reg1.shape[0]):
                vectors[idx, 12] = 0
                return

            hit = False
            for i in range(len(OC_offset) - 1):
                start_OC = OC_offset[i]
                end_OC = OC_offset[i+1]
                if is_inside_or_on_edge(x, y, OC, start_OC, end_OC):
                    hit = True
                    if region_state == 5:
                        # If inside the FOV region for this FOV_num, accumulate to EB
                        if is_inside_or_on_edge_4d(x, y, eff_reg_FOV, m, n):
                            Ete_out, Etm_out, _ = E_field_cal(
                                Ete, Etm, delta_phase,
                                lut_oc1[i, m, n, 13], lut_oc1[i, m, n, 18],  # -1st tran, TE in -> TE/TM
                                lut_oc1[i, m, n, 33], lut_oc1[i, m, n, 38])  # -1st tran, TM in -> TE/TM
                            # squared magnitude (no sqrt)
                            efficiency = Ete_out * Ete_out + Etm_out * Etm_out
                            # Add atomically into the EB matrix bin for this FOV
                            xmin = eff_reg_FOV_range[m, n, 0]
                            xmax = eff_reg_FOV_range[m, n, 1]
                            ymin = eff_reg_FOV_range[m, n, 2]
                            ymax = eff_reg_FOV_range[m, n, 3]
                            add_to_EB_atomic_val(matrix_EB, m, n, x, y, xmin, xmax, ymin, ymax, efficiency)

                        # Then continue propagation with 0th order
                        Ete1, Etm1, delta_phase1 = E_field_cal(
                            Ete, Etm, delta_phase,
                            lut_oc1[i, m, n, 4],  lut_oc1[i, m, n, 9],   # 0th ref
                            lut_oc1[i, m, n, 24], lut_oc1[i, m, n, 29])  # 0th ref
                        efficiency = (Ete1 * Ete1 + Etm1 * Etm1) * math.cos(lut_oc1[i, m, n, 0].real) / math.cos(lut_ic1[m, n, 0].real) * n_g
                        if efficiency < threshold:
                            vectors[idx, 12] = 0 # flag
                        else:
                            vectors[idx, 0] = x + gap_x # position of ray
                            vectors[idx, 1] = y + gap_y
                            vectors[idx, 2] = gap_x # propagation distance for each TIR
                            vectors[idx, 3] = gap_y
                            vectors[idx, 4] = theta.real # polar and azimuth angle of ray (global coordinates)
                            vectors[idx, 5] = phi.real
                            vectors[idx, 6] = m # LUT position
                            vectors[idx, 7] = n
                            vectors[idx, 8] = Ete1
                            vectors[idx, 9] = Etm1
                            vectors[idx, 10] = delta_phase1 + lut_TIR[m, n, 1] # phase shift due to TIR (TM-TE)
                            vectors[idx, 11] = 5 # region state (keep on 5, 0th order)
                            vectors[idx, 12] = 1 # 1: useful, 0: useless
                        
                        ######################################################################
                        Ete2, Etm2, delta_phase2 = E_field_cal(
                            Ete, Etm, delta_phase,
                            lut_oc1[i, m, n, 2], lut_oc1[i, m, n, 7],   # unit TE in, TE&TM out (-2nd ref)
                            lut_oc1[i, m, n, 22], lut_oc1[i, m, n, 27]) # unit TM in, TE&TM out (-2nd ref)
                        efficiency = (Ete2 * Ete2 + Etm2 * Etm2) * math.cos(lut_oc2[i, m, n, 0].real) / math.cos(lut_ic1[m, n, 0].real) * n_g
                        if efficiency < threshold:
                            return
                        new_ray_index = cuda.atomic.add(d_total_ray_counter, 0, 1)
                        vectors[new_ray_index, 0] = x + lut_gap[m, n, 6] # position of ray
                        vectors[new_ray_index, 1] = y + lut_gap[m, n, 7]
                        vectors[new_ray_index, 2] = lut_gap[m, n, 6] # propagation distance for each TIR
                        vectors[new_ray_index, 3] = lut_gap[m, n, 7]
                        vectors[new_ray_index, 4] = lut_oc2[i, m, n, 0].real # polar and azimuth angle of ray (global coordinates)
                        vectors[new_ray_index, 5] = lut_oc2[i, m, n, 1].real
                        vectors[new_ray_index, 6] = m # LUT position
                        vectors[new_ray_index, 7] = n
                        vectors[new_ray_index, 8] = Ete2
                        vectors[new_ray_index, 9] = Etm2
                        vectors[new_ray_index, 10] = delta_phase2 + lut_TIR[m, n, 3] # phase shift due to TIR (TM-TE)
                        vectors[new_ray_index, 11] = 6 # region state (change to 6, -2nd order)
                        vectors[new_ray_index, 12] = 1 # 1: useful, 0: useless
                        return
                    
                    elif region_state == 6:
                        # If inside the FOV region for this FOV_num, accumulate to EB
                        if is_inside_or_on_edge_4d(x, y, eff_reg_FOV, m, n):
                            Ete_out, Etm_out, _ = E_field_cal(
                                Ete, Etm, delta_phase,
                                lut_oc2[i, m, n, 15], lut_oc2[i, m, n, 20],  # +1st tran, TE in -> TE/TM
                                lut_oc2[i, m, n, 35], lut_oc2[i, m, n, 40])  # +1st tran, TM in -> TE/TM
                            # squared magnitude (no sqrt)
                            efficiency = Ete_out * Ete_out + Etm_out * Etm_out
                            # Add atomically into the EB matrix bin for this FOV
                            xmin = eff_reg_FOV_range[m, n, 0]
                            xmax = eff_reg_FOV_range[m, n, 1]
                            ymin = eff_reg_FOV_range[m, n, 2]
                            ymax = eff_reg_FOV_range[m, n, 3]
                            add_to_EB_atomic_val(matrix_EB, m, n, x, y, xmin, xmax, ymin, ymax, efficiency)

                        # Then continue propagation with 0th order
                        Ete1, Etm1, delta_phase1 = E_field_cal(
                            Ete, Etm, delta_phase,
                            lut_oc2[i, m, n, 4],  lut_oc2[i, m, n, 9],   # 0th ref
                            lut_oc2[i, m, n, 24], lut_oc2[i, m, n, 29])  # 0th ref
                        efficiency = (Ete1 * Ete1 + Etm1 * Etm1) * math.cos(lut_oc2[i, m, n, 0].real) / math.cos(lut_ic1[m, n, 0].real) * n_g
                        if efficiency < threshold:
                            vectors[idx, 12] = 0 # flag
                        else:
                            vectors[idx, 0] = x + gap_x # position of ray
                            vectors[idx, 1] = y + gap_y
                            vectors[idx, 2] = gap_x # propagation distance for each TIR
                            vectors[idx, 3] = gap_y
                            vectors[idx, 4] = theta.real # polar and azimuth angle of ray (global coordinates)
                            vectors[idx, 5] = phi.real
                            vectors[idx, 6] = m # LUT position
                            vectors[idx, 7] = n
                            vectors[idx, 8] = Ete1
                            vectors[idx, 9] = Etm1
                            vectors[idx, 10] = delta_phase1 + lut_TIR[m, n, 3] # phase shift due to TIR (TM-TE)
                            vectors[idx, 11] = 6 # region state (keep on 6, 0th order)
                            vectors[idx, 12] = 1 # 1: useful, 0: useless
                        
                        ######################################################################
                        Ete2, Etm2, delta_phase2 = E_field_cal(
                            Ete, Etm, delta_phase,
                            lut_oc2[i, m, n, 6], lut_oc2[i, m, n, 11],   # unit TE in, TE&TM out (+2nd ref)
                            lut_oc2[i, m, n, 26], lut_oc2[i, m, n, 31]) # unit TM in, TE&TM out (+2nd ref)
                        efficiency = (Ete2 * Ete2 + Etm2 * Etm2) * math.cos(lut_oc1[i, m, n, 0].real) / math.cos(lut_ic1[m, n, 0].real) * n_g
                        if efficiency < threshold:
                            return
                        new_ray_index = cuda.atomic.add(d_total_ray_counter, 0, 1)
                        vectors[new_ray_index, 0] = x + lut_gap[m, n, 2] # position of ray
                        vectors[new_ray_index, 1] = y + lut_gap[m, n, 3]
                        vectors[new_ray_index, 2] = lut_gap[m, n, 2] # propagation distance for each TIR
                        vectors[new_ray_index, 3] = lut_gap[m, n, 3]
                        vectors[new_ray_index, 4] = lut_oc1[i, m, n, 0].real # polar and azimuth angle of ray (global coordinates)
                        vectors[new_ray_index, 5] = lut_oc1[i, m, n, 1].real
                        vectors[new_ray_index, 6] = m # LUT position
                        vectors[new_ray_index, 7] = n
                        vectors[new_ray_index, 8] = Ete2
                        vectors[new_ray_index, 9] = Etm2
                        vectors[new_ray_index, 10] = delta_phase2 + lut_TIR[m, n, 1] # phase shift due to TIR (TM-TE)
                        vectors[new_ray_index, 11] = 5 # region state (change to 5, +2nd order)
                        vectors[new_ray_index, 12] = 1 # 1: useful, 0: useless
                        return

            # have not enter OC yet, continue propagate
            if not hit:
                if region_state == 5:
                    delta_phase += 2 * lut_TIR[m, n, 1]
                    x += gap_x
                    y += gap_y
                if region_state == 6:
                    vectors[idx, 12] = 0
                    return


@cuda.jit
def process_rays_kernel_pro( # probability tracing
    x_v, y_v, gap_x_v, gap_y_v, pol_v, azi_v, m_v, n_v, te_v, tm_v, delta_phase_v,
    rng_states,
    IC, FC, FC_offset, OC, OC_offset, n_g,
    eff_reg1, eff_reg2, eff_reg_FOV, eff_reg_FOV_range,
    lut_ic1, lut_ic2, lut_ic3, lut_fc1, lut_fc2, lut_oc1, lut_oc2,
    lut_TIR, lut_gap,
    matrix_EB):

    idx = cuda.grid(1)
    N = x_v.shape[0]

    if idx >= N:
        return
    
    # Read ray property in vectors
    x = x_v[idx] # position of ray
    y = y_v[idx]
    gap_x = gap_x_v[idx] # propagation distance for each TIR
    gap_y = gap_y_v[idx]
    theta = pol_v[idx] # polar and azimuth angle of ray (global coordinates)
    phi = azi_v[idx]
    m = int(m_v[idx]) # LUT position
    n = int(n_v[idx])
    Ete = te_v[idx] # E-field
    Etm = tm_v[idx]
    delta_phase = delta_phase_v[idx] # phase difference between te and tm (phi_tm - phi_te)  range: (-pi, pi]

    ener = 1
    threshold = 1e-15

    # have not interact with IC yet
    # check LUT and change propagation direction and electrical field here (TIR phase shift considered)
    # the parameters need to change: Es, Ep, delta_phase, theta, phi, gap_x, gap_y, energy
    # order to FC direction
    Ete1, Etm1, delta_phase1 = E_field_cal(
        Ete, Etm, delta_phase,
        lut_ic1[m, n, 13], lut_ic1[m, n, 18], # unit TE in, TE&TM out (-1st tran) (to FC)
        lut_ic1[m, n, 33], lut_ic1[m, n, 38]) # unit TM in, TE&TM out (-1st tran) (to FC)
    Ete2, Etm2, delta_phase2 = E_field_cal(
        Ete, Etm, delta_phase,
        lut_ic1[m, n, 15], lut_ic1[m, n, 20], # unit TE in, TE&TM out (+1st tran) (opposite to FC)
        lut_ic1[m, n, 35], lut_ic1[m, n, 40]) # unit TM in, TE&TM out (+1st tran) (opposite to FC)
    efficiency1 = (Ete1 * Ete1 + Etm1 * Etm1) * math.cos(lut_ic2[m, n, 0].real) / math.cos(lut_ic1[m, n, 0].real) * n_g
    efficiency2 = (Ete2 * Ete2 + Etm2 * Etm2) * math.cos(lut_ic3[m, n, 0].real) / math.cos(lut_ic1[m, n, 0].real) * n_g
    rand_val = get_uniform_random_number(rng_states, idx)
    if rand_val <= efficiency1:
        theta = lut_ic2[m, n, 0]
        phi = lut_ic2[m, n, 1]
        norm = math.sqrt(Ete1*Ete1 + Etm1*Etm1)
        Ete = Ete1/norm
        Etm = Etm1/norm
        delta_phase = delta_phase1 + lut_TIR[m, n, 0] # phase shift due to TIR (TM-TE)
        gap_x = lut_gap[m, n, 0] # assign the propagation distance
        gap_y = lut_gap[m, n, 1]
        x += gap_x
        y += gap_y
        ener *= efficiency1
        if not is_inside_or_on_edge(x, y, IC, 0, IC.shape[0]): # two situations will occur: 1. exist IC region 2. does not exist IC yet
            region_state = 2
        else:
            region_state = 0
    elif rand_val <= efficiency1+efficiency2:
        theta = lut_ic3[m, n, 0]
        phi = lut_ic3[m, n, 1]
        norm = math.sqrt(Ete2*Ete2 + Etm2*Etm2)
        Ete = Ete2/norm
        Etm = Etm2/norm
        delta_phase = delta_phase2 + lut_TIR[m, n, 2] # phase shift due to TIR (TM-TE)
        gap_x = lut_gap[m, n, 4] # assign the propagation distance
        gap_y = lut_gap[m, n, 5]
        x += gap_x
        y += gap_y
        ener *= efficiency2
        if not is_inside_or_on_edge(x, y, IC, 0, IC.shape[0]): # two situations will occur: 1. exist IC region 2. does not exist IC yet
            return
        else:
            region_state = 1
    else:
        return
    
    for _ in range(1e5):
        if not is_inside_or_on_edge(x, y, eff_reg1, 0, eff_reg1.shape[0]):
            return
        
        if region_state == 0:
            Ete1, Etm1, delta_phase1 = E_field_cal(
                Ete, Etm, delta_phase,
                lut_ic2[m, n, 4], lut_ic2[m, n, 9], # unit TE in, TE&TM out (0th ref) (to FC)
                lut_ic2[m, n, 24], lut_ic2[m, n, 29]) # unit TM in, TE&TM out (0th ref) (to FC)
            Ete2, Etm2, delta_phase2 = E_field_cal(
                Ete, Etm, delta_phase,
                lut_ic2[m, n, 6], lut_ic2[m, n, 11], # unit TE in, TE&TM out (+2nd ref) (opposite to FC)
                lut_ic2[m, n, 26], lut_ic2[m, n, 31]) # unit TM in, TE&TM out (+2nd ref) (opposite to FC)
            efficiency1 = (Ete1 * Ete1 + Etm1 * Etm1) * math.cos(lut_ic2[m, n, 0].real) / math.cos(theta.real) # 0th ref
            efficiency2 = (Ete2 * Ete2 + Etm2 * Etm2) * math.cos(lut_ic3[m, n, 0].real) / math.cos(theta.real) # +2nd ref
            rand_val = get_uniform_random_number(rng_states, idx)
            if rand_val <= efficiency1:
                theta = lut_ic2[m, n, 0]
                phi = lut_ic2[m, n, 1]
                norm = math.sqrt(Ete1*Ete1 + Etm1*Etm1)
                Ete = Ete1/norm
                Etm = Etm1/norm
                delta_phase = delta_phase1 + lut_TIR[m, n, 0] # phase shift due to TIR (TM-TE)
                gap_x = lut_gap[m, n, 0] # assign the propagation distance
                gap_y = lut_gap[m, n, 1]
                x += gap_x
                y += gap_y
                ener *= efficiency1
                if not is_inside_or_on_edge(x, y, IC, 0, IC.shape[0]): # two situations will occur: 1. exist IC region 2. does not exist IC yet
                    region_state = 2
                else:
                    region_state = 0
            elif rand_val <= efficiency1+efficiency2:
                theta = lut_ic3[m, n, 0]
                phi = lut_ic3[m, n, 1]
                norm = math.sqrt(Ete2*Ete2 + Etm2*Etm2)
                Ete = Ete2/norm
                Etm = Etm2/norm
                delta_phase = delta_phase2 + lut_TIR[m, n, 2] # phase shift due to TIR (TM-TE)
                gap_x = lut_gap[m, n, 4] # assign the propagation distance
                gap_y = lut_gap[m, n, 5]
                x += gap_x
                y += gap_y
                ener *= efficiency2
                if not is_inside_or_on_edge(x, y, IC, 0, IC.shape[0]): # two situations will occur: 1. exist IC region 2. does not exist IC yet
                    return
                else:
                    region_state = 1
            else:
                return
        
        elif region_state == 1:
            Ete1, Etm1, delta_phase1 = E_field_cal(
                Ete, Etm, delta_phase,
                lut_ic3[m, n, 2], lut_ic3[m, n, 22], # unit TE in, TE&TM out (-2nd ref) (to FC)
                lut_ic3[m, n, 7], lut_ic3[m, n, 27]) # unit TM in, TE&TM out (-2nd ref) (to FC)
            Ete2, Etm2, delta_phase2 = E_field_cal(
                Ete, Etm, delta_phase,
                lut_ic3[m, n, 4], lut_ic3[m, n, 9], # unit TE in, TE&TM out (0th ref) (opposite to FC)
                lut_ic3[m, n, 24], lut_ic3[m, n, 29]) # unit TM in, TE&TM out (0th ref) (opposite to FC)
            efficiency1 = (Ete1 * Ete1 + Etm1 * Etm1) * math.cos(lut_ic2[m, n, 0].real) / math.cos(theta.real)
            efficiency2 = (Ete2 * Ete2 + Etm2 * Etm2) * math.cos(lut_ic3[m, n, 0].real) / math.cos(theta.real)
            rand_val = get_uniform_random_number(rng_states, idx)
            if rand_val <= efficiency1:
                theta = lut_ic2[m, n, 0]
                phi = lut_ic2[m, n, 1]
                norm = math.sqrt(Ete1*Ete1 + Etm1*Etm1)
                Ete = Ete1/norm
                Etm = Etm1/norm
                delta_phase = delta_phase1 + lut_TIR[m, n, 0] # phase shift due to TIR (TM-TE)
                gap_x = lut_gap[m, n, 0] # assign the propagation distance
                gap_y = lut_gap[m, n, 1]
                x += gap_x
                y += gap_y
                ener *= efficiency1
                if not is_inside_or_on_edge(x, y, IC, 0, IC.shape[0]): # two situations will occur: 1. exist IC region 2. does not exist IC yet
                    region_state = 2
                else:
                    region_state = 0
            elif rand_val <= efficiency1+efficiency2:
                theta = lut_ic3[m, n, 0]
                phi = lut_ic3[m, n, 1]
                norm = math.sqrt(Ete2*Ete2 + Etm2*Etm2)
                Ete = Ete2/norm
                Etm = Etm2/norm
                delta_phase = delta_phase2 + lut_TIR[m, n, 2] # phase shift due to TIR (TM-TE)
                gap_x = lut_gap[m, n, 4] # assign the propagation distance
                gap_y = lut_gap[m, n, 5]
                x += gap_x
                y += gap_y
                ener *= efficiency2
                if not is_inside_or_on_edge(x, y, IC, 0, IC.shape[0]): # two situations will occur: 1. exist IC region 2. does not exist IC yet
                    return
                else:
                    region_state = 1
            else:
                return
        
        elif region_state == 2: # exist IC and interact with FC (to FC direction)
            hit = False
            for i in range(len(FC_offset)-1):
                start_FC = FC_offset[i]
                end_FC = FC_offset[i+1]
                if is_inside_or_on_edge(x, y, FC, start_FC, end_FC):
                    hit = True
                    Ete1, Etm1, delta_phase1 = E_field_cal(
                        Ete, Etm, delta_phase,
                        lut_fc1[i, m, n, 3], lut_fc1[i, m, n, 6],   # unit TE in, TE&TM out (0th ref)
                        lut_fc1[i, m, n, 15], lut_fc1[i, m, n, 18]) # unit TM in, TE&TM out (0th ref)
                    Ete2, Etm2, delta_phase2 = E_field_cal(
                        Ete, Etm, delta_phase,
                        lut_fc1[i, m, n, 2], lut_fc1[i, m, n, 5],   # unit TE in, TE&TM out (-1st ref)
                        lut_fc1[i, m, n, 14], lut_fc1[i, m, n, 17]) # unit TM in, TE&TM out (-1st ref)
                    efficiency1 = (Ete1 * Ete1 + Etm1 * Etm1) * math.cos(lut_fc1[i, m, n, 0].real) / math.cos(theta.real)
                    efficiency2 = (Ete2 * Ete2 + Etm2 * Etm2) * math.cos(lut_fc2[i, m, n, 0].real) / math.cos(theta.real)
                    ener1 = ener * efficiency1
                    ener2 = ener * efficiency2
                    rand_val = get_uniform_random_number(rng_states, idx)
                    if rand_val <= efficiency1 and ener1 > threshold:
                        theta = lut_fc1[i, m, n, 0]
                        phi = lut_fc1[i, m, n, 1]
                        norm = math.sqrt(Ete1*Ete1 + Etm1*Etm1)
                        Ete = Ete1/norm
                        Etm = Etm1/norm
                        delta_phase = delta_phase1 + lut_TIR[m, n, 0] # phase shift due to TIR (TM-TE)
                        gap_x = lut_gap[m, n, 0] # assign the propagation distance
                        gap_y = lut_gap[m, n, 1]
                        x += gap_x
                        y += gap_y
                        ener = ener1 * 1.0
                        region_state = 2
                    elif rand_val <= efficiency1+efficiency2 and ener2 > threshold:
                        theta = lut_fc2[i, m, n, 0]
                        phi = lut_fc2[i, m, n, 1]
                        norm = math.sqrt(Ete2*Ete2 + Etm2*Etm2)
                        Ete = Ete2/norm
                        Etm = Etm2/norm
                        delta_phase = delta_phase2 + lut_TIR[m, n, 1] # phase shift due to TIR (TM-TE)
                        gap_x = lut_gap[m, n, 2] # assign the propagation distance
                        gap_y = lut_gap[m, n, 3]
                        x += gap_x
                        y += gap_y
                        ener = ener2 * 1.0
                        region_state = 3
                    else:
                        return
                    break
            if not hit:
                x += gap_x
                y += gap_y
                delta_phase += 2*lut_TIR[m, n, 0]
        
        elif region_state == 3: # exist IC and interact with FC (to OC direction)
            hit = False
            for i in range(len(FC_offset)-1):
                start_FC = FC_offset[i]
                end_FC = FC_offset[i+1]
                if is_inside_or_on_edge(x, y, FC, start_FC, end_FC):
                    hit = True
                    Ete1, Etm1, delta_phase1 = E_field_cal(
                        Ete, Etm, delta_phase,
                        lut_fc2[i, m, n, 4], lut_fc2[i, m, n, 7],   # unit TE in, TE&TM out (+1st ref)
                        lut_fc2[i, m, n, 16], lut_fc2[i, m, n, 19]) # unit TM in, TE&TM out (+1st ref)
                    Ete2, Etm2, delta_phase2 = E_field_cal(
                        Ete, Etm, delta_phase,
                        lut_fc2[i, m, n, 3], lut_fc2[i, m, n, 6],   # unit TE in, TE&TM out (0th ref)
                        lut_fc2[i, m, n, 15], lut_fc2[i, m, n, 18]) # unit TM in, TE&TM out (0th ref)
                    efficiency1 = (Ete1 * Ete1 + Etm1 * Etm1) * math.cos(lut_fc1[i, m, n, 0].real) / math.cos(theta.real)
                    efficiency2 = (Ete2 * Ete2 + Etm2 * Etm2) * math.cos(lut_fc2[i, m, n, 0].real) / math.cos(theta.real)
                    ener1 = ener * efficiency1
                    ener2 = ener * efficiency2
                    rand_val = get_uniform_random_number(rng_states, idx)
                    if rand_val <= efficiency1 and ener1 > threshold:
                        theta = lut_fc1[i, m, n, 0]
                        phi = lut_fc1[i, m, n, 1]
                        norm = math.sqrt(Ete1*Ete1 + Etm1*Etm1)
                        Ete = Ete1/norm
                        Etm = Etm1/norm
                        delta_phase = delta_phase1 + lut_TIR[m, n, 0] # phase shift due to TIR (TM-TE)
                        gap_x = lut_gap[m, n, 0] # assign the propagation distance
                        gap_y = lut_gap[m, n, 1]
                        x += gap_x
                        y += gap_y
                        ener = ener1 * 1.0
                        region_state = 2
                    elif rand_val <= efficiency1+efficiency2 and ener2 > threshold:
                        theta = lut_fc2[i, m, n, 0]
                        phi = lut_fc2[i, m, n, 1]
                        norm = math.sqrt(Ete2*Ete2 + Etm2*Etm2)
                        Ete = Ete2/norm
                        Etm = Etm2/norm
                        delta_phase = delta_phase2 + lut_TIR[m, n, 1] # phase shift due to TIR (TM-TE)
                        gap_x = lut_gap[m, n, 2] # assign the propagation distance
                        gap_y = lut_gap[m, n, 3]
                        x += gap_x
                        y += gap_y
                        ener = ener2 * 1.0
                        region_state = 3
                    else:
                        return
                    break
            if not hit:
                if not is_inside_or_on_edge(x, y, eff_reg2, 0, eff_reg2.shape[0]):
                    region_state = 4
                else:
                    x += gap_x
                    y += gap_y
                    delta_phase += 2*lut_TIR[m, n, 1]

        elif region_state == 4: # exist FC and interact with OC (to OC direction)
            hit = False
            for i in range(len(OC_offset)-1):
                start_OC = OC_offset[i]
                end_OC = OC_offset[i+1]
                if is_inside_or_on_edge(x, y, OC, start_OC, end_OC):
                    hit = True
                    Ete1, Etm1, delta_phase1 = E_field_cal(
                        Ete, Etm, delta_phase,
                        lut_oc1[i, m, n, 4],  lut_oc1[i, m, n, 9],   # 0th ref
                        lut_oc1[i, m, n, 24], lut_oc1[i, m, n, 29])  # 0th ref
                    Ete2, Etm2, delta_phase2 = E_field_cal(
                        Ete, Etm, delta_phase,
                        lut_oc1[i, m, n, 2], lut_oc1[i, m, n, 7],   # -2nd ref
                        lut_oc1[i, m, n, 22], lut_oc1[i, m, n, 27]) # -2nd ref
                    Ete3, Etm3, delta_phase3 = E_field_cal(
                        Ete, Etm, delta_phase,
                        lut_oc1[i, m, n, 13], lut_oc1[i, m, n, 18],  # -1st tran
                        lut_oc1[i, m, n, 33], lut_oc1[i, m, n, 38])  # -1st tran
                    efficiency1 = (Ete1 * Ete1 + Etm1 * Etm1) * math.cos(lut_oc1[i, m, n, 0].real) / math.cos(theta.real)
                    efficiency2 = (Ete2 * Ete2 + Etm2 * Etm2) * math.cos(lut_oc2[i, m, n, 0].real) / math.cos(theta.real)
                    efficiency3 = (Ete3 * Ete3 + Etm3 * Etm3) * math.cos(lut_ic1[m, n, 0].real) / math.cos(theta.real) / n_g
                    ener1 = ener * efficiency1
                    ener2 = ener * efficiency2
                    ener3 = ener * efficiency3
                    rand_val = get_uniform_random_number(rng_states, idx)
                    if rand_val <= efficiency1 and ener1 > threshold:
                        theta = lut_oc1[i, m, n, 0]
                        phi = lut_oc1[i, m, n, 1]
                        norm = math.sqrt(Ete1*Ete1 + Etm1*Etm1)
                        Ete = Ete1/norm
                        Etm = Etm1/norm
                        delta_phase = delta_phase1 + lut_TIR[m, n, 1] # phase shift due to TIR (TM-TE)
                        gap_x = lut_gap[m, n, 2] # assign the propagation distance
                        gap_y = lut_gap[m, n, 3]
                        x += gap_x
                        y += gap_y
                        ener = ener1 * 1.0
                        region_state = 4
                    elif rand_val <= efficiency1+efficiency2 and ener2 > threshold:
                        theta = lut_oc2[i, m, n, 0]
                        phi = lut_oc2[i, m, n, 2]
                        norm = math.sqrt(Ete2*Ete2 + Etm2*Etm2)
                        Ete = Ete2/norm
                        Etm = Etm2/norm
                        delta_phase = delta_phase2 + lut_TIR[m, n, 3] # phase shift due to TIR (TM-TE)
                        gap_x = lut_gap[m, n, 6] # assign the propagation distance
                        gap_y = lut_gap[m, n, 7]
                        x += gap_x
                        y += gap_y
                        ener = ener2 * 1.0
                        region_state = 5
                    elif rand_val <= efficiency1+efficiency2+efficiency3 and ener3 > threshold:
                        if is_inside_or_on_edge_4d(x, y, eff_reg_FOV, m, n):
                            xmin = eff_reg_FOV_range[m, n, 0]
                            xmax = eff_reg_FOV_range[m, n, 1]
                            ymin = eff_reg_FOV_range[m, n, 2]
                            ymax = eff_reg_FOV_range[m, n, 3]
                            add_to_EB_atomic_val(matrix_EB, m, n, x, y, xmin, xmax, ymin, ymax, 1.0)
                            return
                        else:
                            return
                    else:
                        return
                    break
            if not hit:
                x += gap_x
                y += gap_y
                delta_phase += 2*lut_TIR[m, n, 1]

        elif region_state == 5: # exist FC and interact with OC (opposite OC direction)
            hit = False
            for i in range(len(OC_offset)-1):
                start_OC = OC_offset[i]
                end_OC = OC_offset[i+1]
                if is_inside_or_on_edge(x, y, OC, start_OC, end_OC):
                    hit = True
                    Ete1, Etm1, delta_phase1 = E_field_cal(
                        Ete, Etm, delta_phase,
                        lut_oc2[i, m, n, 6], lut_oc2[i, m, n, 11],   # +2nd ref
                        lut_oc2[i, m, n, 26], lut_oc2[i, m, n, 31])  # +2nd ref
                    Ete2, Etm2, delta_phase2 = E_field_cal(
                        Ete, Etm, delta_phase,
                        lut_oc2[i, m, n, 4],  lut_oc2[i, m, n, 9],   # 0th ref
                        lut_oc2[i, m, n, 24], lut_oc2[i, m, n, 29])  # 0th ref
                    Ete3, Etm3, delta_phase3 = E_field_cal(
                        Ete, Etm, delta_phase,
                        lut_oc2[i, m, n, 15], lut_oc2[i, m, n, 20],  # +1st tran
                        lut_oc2[i, m, n, 35], lut_oc2[i, m, n, 40])  # +1st tran
                    efficiency1 = (Ete1 * Ete1 + Etm1 * Etm1) * math.cos(lut_oc1[i, m, n, 0].real) / math.cos(theta.real)
                    efficiency2 = (Ete2 * Ete2 + Etm2 * Etm2) * math.cos(lut_oc2[i, m, n, 0].real) / math.cos(theta.real)
                    efficiency3 = (Ete3 * Ete3 + Etm3 * Etm3) * math.cos(lut_ic1[m, n, 0].real) / math.cos(theta.real) / n_g
                    ener1 = ener * efficiency1
                    ener2 = ener * efficiency2
                    ener3 = ener * efficiency3
                    rand_val = get_uniform_random_number(rng_states, idx)
                    if rand_val <= efficiency1 and ener1 > threshold:
                        theta = lut_oc1[i, m, n, 0]
                        phi = lut_oc1[i, m, n, 1]
                        norm = math.sqrt(Ete1*Ete1 + Etm1*Etm1)
                        Ete = Ete1/norm
                        Etm = Etm1/norm
                        delta_phase = delta_phase1 + lut_TIR[m, n, 1] # phase shift due to TIR (TM-TE)
                        gap_x = lut_gap[m, n, 2] # assign the propagation distance
                        gap_y = lut_gap[m, n, 3]
                        x += gap_x
                        y += gap_y
                        ener = ener1 * 1.0
                        region_state = 4
                    elif rand_val <= efficiency1+efficiency2 and ener2 > threshold:
                        theta = lut_oc2[i, m, n, 0]
                        phi = lut_oc2[i, m, n, 2]
                        norm = math.sqrt(Ete2*Ete2 + Etm2*Etm2)
                        Ete = Ete2/norm
                        Etm = Etm2/norm
                        delta_phase = delta_phase2 + lut_TIR[m, n, 3] # phase shift due to TIR (TM-TE)
                        gap_x = lut_gap[m, n, 6] # assign the propagation distance
                        gap_y = lut_gap[m, n, 7]
                        x += gap_x
                        y += gap_y
                        ener = ener2 * 1.0
                        region_state = 5
                    elif rand_val <= efficiency1+efficiency2+efficiency3 and ener3 > threshold:
                        if is_inside_or_on_edge_4d(x, y, eff_reg_FOV, m, n):
                            xmin = eff_reg_FOV_range[m, n, 0]
                            xmax = eff_reg_FOV_range[m, n, 1]
                            ymin = eff_reg_FOV_range[m, n, 2]
                            ymax = eff_reg_FOV_range[m, n, 3]
                            add_to_EB_atomic_val(matrix_EB, m, n, x, y, xmin, xmax, ymin, ymax, 1.0)
                            return
                        else:
                            return
                    else:
                        return
                    break
            if not hit:
                return


@cuda.jit
def process_rays_kernel_pro_fullColor( # probability tracing
    x_v, y_v, gap_x_v, gap_y_v, pol_v, azi_v, m_v, n_v, lmd_num, te_v, tm_v, delta_phase_v,
    rng_states,
    IC, FC, FC_offset, OC, OC_offset, n_g,
    eff_reg1, eff_reg2, eff_reg_FOV, eff_reg_FOV_range,
    lut_ic1, lut_ic2, lut_ic3, lut_fc1, lut_fc2, lut_oc1, lut_oc2,
    lut_TIR, lut_gap,
    matrix_EB):

    idx = cuda.grid(1)
    N = x_v.shape[0]

    if idx >= N:
        return
    
    # Read ray property in vectors
    x = x_v[idx] # position of ray
    y = y_v[idx]
    gap_x = gap_x_v[idx] # propagation distance for each TIR
    gap_y = gap_y_v[idx]
    theta = pol_v[idx] # polar and azimuth angle of ray (global coordinates)
    phi = azi_v[idx]
    m = int(m_v[idx]) # LUT position
    n = int(n_v[idx])
    lmd_num = int(lmd_num[idx])
    Ete = te_v[idx] # E-field
    Etm = tm_v[idx]
    delta_phase = delta_phase_v[idx] # phase difference between te and tm (phi_tm - phi_te)  range: (-pi, pi]

    ener = 1
    threshold = 1e-15

    # have not interact with IC yet
    # check LUT and change propagation direction and electrical field here (TIR phase shift considered)
    # the parameters need to change: Es, Ep, delta_phase, theta, phi, gap_x, gap_y, energy
    # order to FC direction
    Ete1, Etm1, delta_phase1 = E_field_cal(
        Ete, Etm, delta_phase,
        lut_ic1[lmd_num, m, n, 13], lut_ic1[lmd_num, m, n, 18], # unit TE in, TE&TM out (-1st tran) (to FC)
        lut_ic1[lmd_num, m, n, 33], lut_ic1[lmd_num, m, n, 38]) # unit TM in, TE&TM out (-1st tran) (to FC)
    Ete2, Etm2, delta_phase2 = E_field_cal(
        Ete, Etm, delta_phase,
        lut_ic1[lmd_num, m, n, 15], lut_ic1[lmd_num, m, n, 20], # unit TE in, TE&TM out (+1st tran) (opposite to FC)
        lut_ic1[lmd_num, m, n, 35], lut_ic1[lmd_num, m, n, 40]) # unit TM in, TE&TM out (+1st tran) (opposite to FC)
    efficiency1 = (Ete1 * Ete1 + Etm1 * Etm1) * math.cos(lut_ic2[lmd_num, m, n, 0].real) / math.cos(lut_ic1[lmd_num, m, n, 0].real) * n_g
    efficiency2 = (Ete2 * Ete2 + Etm2 * Etm2) * math.cos(lut_ic3[lmd_num, m, n, 0].real) / math.cos(lut_ic1[lmd_num, m, n, 0].real) * n_g
    rand_val = get_uniform_random_number(rng_states, idx)
    if rand_val <= efficiency1:
        theta = lut_ic2[lmd_num, m, n, 0]
        phi = lut_ic2[lmd_num, m, n, 1]
        norm = math.sqrt(Ete1*Ete1 + Etm1*Etm1)
        Ete = Ete1/norm
        Etm = Etm1/norm
        delta_phase = delta_phase1 + lut_TIR[lmd_num, m, n, 0] # phase shift due to TIR (TM-TE)
        gap_x = lut_gap[lmd_num, m, n, 0] # assign the propagation distance
        gap_y = lut_gap[lmd_num, m, n, 1]
        x += gap_x
        y += gap_y
        ener *= efficiency1
        if not is_inside_or_on_edge(x, y, IC, 0, IC.shape[0]): # two situations will occur: 1. exist IC region 2. does not exist IC yet
            region_state = 2
        else:
            region_state = 0
    elif rand_val <= efficiency1+efficiency2:
        theta = lut_ic3[lmd_num, m, n, 0]
        phi = lut_ic3[lmd_num, m, n, 1]
        norm = math.sqrt(Ete2*Ete2 + Etm2*Etm2)
        Ete = Ete2/norm
        Etm = Etm2/norm
        delta_phase = delta_phase2 + lut_TIR[lmd_num, m, n, 2] # phase shift due to TIR (TM-TE)
        gap_x = lut_gap[lmd_num, m, n, 4] # assign the propagation distance
        gap_y = lut_gap[lmd_num, m, n, 5]
        x += gap_x
        y += gap_y
        ener *= efficiency2
        if not is_inside_or_on_edge(x, y, IC, 0, IC.shape[0]): # two situations will occur: 1. exist IC region 2. does not exist IC yet
            return
        else:
            region_state = 1
    else:
        return
    
    for _ in range(1e5):
        if not is_inside_or_on_edge(x, y, eff_reg1, 0, eff_reg1.shape[0]):
            return
        
        if region_state == 0:
            Ete1, Etm1, delta_phase1 = E_field_cal(
                Ete, Etm, delta_phase,
                lut_ic2[lmd_num, m, n, 4], lut_ic2[lmd_num, m, n, 9], # unit TE in, TE&TM out (0th ref) (to FC)
                lut_ic2[lmd_num, m, n, 24], lut_ic2[lmd_num, m, n, 29]) # unit TM in, TE&TM out (0th ref) (to FC)
            Ete2, Etm2, delta_phase2 = E_field_cal(
                Ete, Etm, delta_phase,
                lut_ic2[lmd_num, m, n, 6], lut_ic2[lmd_num, m, n, 11], # unit TE in, TE&TM out (+2nd ref) (opposite to FC)
                lut_ic2[lmd_num, m, n, 26], lut_ic2[lmd_num, m, n, 31]) # unit TM in, TE&TM out (+2nd ref) (opposite to FC)
            efficiency1 = (Ete1 * Ete1 + Etm1 * Etm1) * math.cos(lut_ic2[lmd_num, m, n, 0].real) / math.cos(theta.real) # 0th ref
            efficiency2 = (Ete2 * Ete2 + Etm2 * Etm2) * math.cos(lut_ic3[lmd_num, m, n, 0].real) / math.cos(theta.real) # +2nd ref
            rand_val = get_uniform_random_number(rng_states, idx)
            if rand_val <= efficiency1:
                theta = lut_ic2[lmd_num, m, n, 0]
                phi = lut_ic2[lmd_num, m, n, 1]
                norm = math.sqrt(Ete1*Ete1 + Etm1*Etm1)
                Ete = Ete1/norm
                Etm = Etm1/norm
                delta_phase = delta_phase1 + lut_TIR[lmd_num, m, n, 0] # phase shift due to TIR (TM-TE)
                gap_x = lut_gap[lmd_num, m, n, 0] # assign the propagation distance
                gap_y = lut_gap[lmd_num, m, n, 1]
                x += gap_x
                y += gap_y
                ener *= efficiency1
                if not is_inside_or_on_edge(x, y, IC, 0, IC.shape[0]): # two situations will occur: 1. exist IC region 2. does not exist IC yet
                    region_state = 2
                else:
                    region_state = 0
            elif rand_val <= efficiency1+efficiency2:
                theta = lut_ic3[lmd_num, m, n, 0]
                phi = lut_ic3[lmd_num, m, n, 1]
                norm = math.sqrt(Ete2*Ete2 + Etm2*Etm2)
                Ete = Ete2/norm
                Etm = Etm2/norm
                delta_phase = delta_phase2 + lut_TIR[lmd_num, m, n, 2] # phase shift due to TIR (TM-TE)
                gap_x = lut_gap[lmd_num, m, n, 4] # assign the propagation distance
                gap_y = lut_gap[lmd_num, m, n, 5]
                x += gap_x
                y += gap_y
                ener *= efficiency2
                if not is_inside_or_on_edge(x, y, IC, 0, IC.shape[0]): # two situations will occur: 1. exist IC region 2. does not exist IC yet
                    return
                else:
                    region_state = 1
            else:
                return
        
        elif region_state == 1:
            Ete1, Etm1, delta_phase1 = E_field_cal(
                Ete, Etm, delta_phase,
                lut_ic3[lmd_num, m, n, 2], lut_ic3[lmd_num, m, n, 22], # unit TE in, TE&TM out (-2nd ref) (to FC)
                lut_ic3[lmd_num, m, n, 7], lut_ic3[lmd_num, m, n, 27]) # unit TM in, TE&TM out (-2nd ref) (to FC)
            Ete2, Etm2, delta_phase2 = E_field_cal(
                Ete, Etm, delta_phase,
                lut_ic3[lmd_num, m, n, 4], lut_ic3[lmd_num, m, n, 9], # unit TE in, TE&TM out (0th ref) (opposite to FC)
                lut_ic3[lmd_num, m, n, 24], lut_ic3[lmd_num, m, n, 29]) # unit TM in, TE&TM out (0th ref) (opposite to FC)
            efficiency1 = (Ete1 * Ete1 + Etm1 * Etm1) * math.cos(lut_ic2[lmd_num, m, n, 0].real) / math.cos(theta.real)
            efficiency2 = (Ete2 * Ete2 + Etm2 * Etm2) * math.cos(lut_ic3[lmd_num, m, n, 0].real) / math.cos(theta.real)
            rand_val = get_uniform_random_number(rng_states, idx)
            if rand_val <= efficiency1:
                theta = lut_ic2[lmd_num, m, n, 0]
                phi = lut_ic2[lmd_num, m, n, 1]
                norm = math.sqrt(Ete1*Ete1 + Etm1*Etm1)
                Ete = Ete1/norm
                Etm = Etm1/norm
                delta_phase = delta_phase1 + lut_TIR[lmd_num, m, n, 0] # phase shift due to TIR (TM-TE)
                gap_x = lut_gap[lmd_num, m, n, 0] # assign the propagation distance
                gap_y = lut_gap[lmd_num, m, n, 1]
                x += gap_x
                y += gap_y
                ener *= efficiency1
                if not is_inside_or_on_edge(x, y, IC, 0, IC.shape[0]): # two situations will occur: 1. exist IC region 2. does not exist IC yet
                    region_state = 2
                else:
                    region_state = 0
            elif rand_val <= efficiency1+efficiency2:
                theta = lut_ic3[lmd_num, m, n, 0]
                phi = lut_ic3[lmd_num, m, n, 1]
                norm = math.sqrt(Ete2*Ete2 + Etm2*Etm2)
                Ete = Ete2/norm
                Etm = Etm2/norm
                delta_phase = delta_phase2 + lut_TIR[lmd_num, m, n, 2] # phase shift due to TIR (TM-TE)
                gap_x = lut_gap[lmd_num, m, n, 4] # assign the propagation distance
                gap_y = lut_gap[lmd_num, m, n, 5]
                x += gap_x
                y += gap_y
                ener *= efficiency2
                if not is_inside_or_on_edge(x, y, IC, 0, IC.shape[0]): # two situations will occur: 1. exist IC region 2. does not exist IC yet
                    return
                else:
                    region_state = 1
            else:
                return
        
        elif region_state == 2: # exist IC and interact with FC (to FC direction)
            hit = False
            for i in range(len(FC_offset)-1):
                start_FC = FC_offset[i]
                end_FC = FC_offset[i+1]
                if is_inside_or_on_edge(x, y, FC, start_FC, end_FC):
                    hit = True
                    Ete1, Etm1, delta_phase1 = E_field_cal(
                        Ete, Etm, delta_phase,
                        lut_fc1[i, lmd_num, m, n, 3], lut_fc1[i, lmd_num, m, n, 6],   # unit TE in, TE&TM out (0th ref)
                        lut_fc1[i, lmd_num, m, n, 15], lut_fc1[i, lmd_num, m, n, 18]) # unit TM in, TE&TM out (0th ref)
                    Ete2, Etm2, delta_phase2 = E_field_cal(
                        Ete, Etm, delta_phase,
                        lut_fc1[i, lmd_num, m, n, 2], lut_fc1[i, lmd_num, m, n, 5],   # unit TE in, TE&TM out (-1st ref)
                        lut_fc1[i, lmd_num, m, n, 14], lut_fc1[i, lmd_num, m, n, 17]) # unit TM in, TE&TM out (-1st ref)
                    efficiency1 = (Ete1 * Ete1 + Etm1 * Etm1) * math.cos(lut_fc1[i, lmd_num, m, n, 0].real) / math.cos(theta.real)
                    efficiency2 = (Ete2 * Ete2 + Etm2 * Etm2) * math.cos(lut_fc2[i, lmd_num, m, n, 0].real) / math.cos(theta.real)
                    ener1 = ener * efficiency1
                    ener2 = ener * efficiency2
                    rand_val = get_uniform_random_number(rng_states, idx)
                    if rand_val <= efficiency1 and ener1 > threshold:
                        theta = lut_fc1[i, lmd_num, m, n, 0]
                        phi = lut_fc1[i, lmd_num, m, n, 1]
                        norm = math.sqrt(Ete1*Ete1 + Etm1*Etm1)
                        Ete = Ete1/norm
                        Etm = Etm1/norm
                        delta_phase = delta_phase1 + lut_TIR[lmd_num, m, n, 0] # phase shift due to TIR (TM-TE)
                        gap_x = lut_gap[lmd_num, m, n, 0] # assign the propagation distance
                        gap_y = lut_gap[lmd_num, m, n, 1]
                        x += gap_x
                        y += gap_y
                        ener = ener1 * 1.0
                        region_state = 2
                    elif rand_val <= efficiency1+efficiency2 and ener2 > threshold:
                        theta = lut_fc2[i, lmd_num, m, n, 0]
                        phi = lut_fc2[i, lmd_num, m, n, 1]
                        norm = math.sqrt(Ete2*Ete2 + Etm2*Etm2)
                        Ete = Ete2/norm
                        Etm = Etm2/norm
                        delta_phase = delta_phase2 + lut_TIR[lmd_num, m, n, 1] # phase shift due to TIR (TM-TE)
                        gap_x = lut_gap[lmd_num, m, n, 2] # assign the propagation distance
                        gap_y = lut_gap[lmd_num, m, n, 3]
                        x += gap_x
                        y += gap_y
                        ener = ener2 * 1.0
                        region_state = 3
                    else:
                        return
                    break
            if not hit:
                x += gap_x
                y += gap_y
                delta_phase += 2*lut_TIR[lmd_num, m, n, 0]
        
        elif region_state == 3: # exist IC and interact with FC (to OC direction)
            hit = False
            for i in range(len(FC_offset)-1):
                start_FC = FC_offset[i]
                end_FC = FC_offset[i+1]
                if is_inside_or_on_edge(x, y, FC, start_FC, end_FC):
                    hit = True
                    Ete1, Etm1, delta_phase1 = E_field_cal(
                        Ete, Etm, delta_phase,
                        lut_fc2[i, lmd_num, m, n, 4], lut_fc2[i, lmd_num, m, n, 7],   # unit TE in, TE&TM out (+1st ref)
                        lut_fc2[i, lmd_num, m, n, 16], lut_fc2[i, lmd_num, m, n, 19]) # unit TM in, TE&TM out (+1st ref)
                    Ete2, Etm2, delta_phase2 = E_field_cal(
                        Ete, Etm, delta_phase,
                        lut_fc2[i, lmd_num, m, n, 3], lut_fc2[i, lmd_num, m, n, 6],   # unit TE in, TE&TM out (0th ref)
                        lut_fc2[i, lmd_num, m, n, 15], lut_fc2[i, lmd_num, m, n, 18]) # unit TM in, TE&TM out (0th ref)
                    efficiency1 = (Ete1 * Ete1 + Etm1 * Etm1) * math.cos(lut_fc1[i, lmd_num, m, n, 0].real) / math.cos(theta.real)
                    efficiency2 = (Ete2 * Ete2 + Etm2 * Etm2) * math.cos(lut_fc2[i, lmd_num, m, n, 0].real) / math.cos(theta.real)
                    ener1 = ener * efficiency1
                    ener2 = ener * efficiency2
                    rand_val = get_uniform_random_number(rng_states, idx)
                    if rand_val <= efficiency1 and ener1 > threshold:
                        theta = lut_fc1[i, lmd_num, m, n, 0]
                        phi = lut_fc1[i, lmd_num, m, n, 1]
                        norm = math.sqrt(Ete1*Ete1 + Etm1*Etm1)
                        Ete = Ete1/norm
                        Etm = Etm1/norm
                        delta_phase = delta_phase1 + lut_TIR[lmd_num, m, n, 0] # phase shift due to TIR (TM-TE)
                        gap_x = lut_gap[lmd_num, m, n, 0] # assign the propagation distance
                        gap_y = lut_gap[lmd_num, m, n, 1]
                        x += gap_x
                        y += gap_y
                        ener = ener1 * 1.0
                        region_state = 2
                    elif rand_val <= efficiency1+efficiency2 and ener2 > threshold:
                        theta = lut_fc2[i, lmd_num, m, n, 0]
                        phi = lut_fc2[i, lmd_num, m, n, 1]
                        norm = math.sqrt(Ete2*Ete2 + Etm2*Etm2)
                        Ete = Ete2/norm
                        Etm = Etm2/norm
                        delta_phase = delta_phase2 + lut_TIR[lmd_num, m, n, 1] # phase shift due to TIR (TM-TE)
                        gap_x = lut_gap[lmd_num, m, n, 2] # assign the propagation distance
                        gap_y = lut_gap[lmd_num, m, n, 3]
                        x += gap_x
                        y += gap_y
                        ener = ener2 * 1.0
                        region_state = 3
                    else:
                        return
                    break
            if not hit:
                if not is_inside_or_on_edge(x, y, eff_reg2, 0, eff_reg2.shape[0]):
                    region_state = 4
                else:
                    x += gap_x
                    y += gap_y
                    delta_phase += 2*lut_TIR[lmd_num, m, n, 1]

        elif region_state == 4: # exist FC and interact with OC (to OC direction)
            hit = False
            for i in range(len(OC_offset)-1):
                start_OC = OC_offset[i]
                end_OC = OC_offset[i+1]
                if is_inside_or_on_edge(x, y, OC, start_OC, end_OC):
                    hit = True
                    Ete1, Etm1, delta_phase1 = E_field_cal(
                        Ete, Etm, delta_phase,
                        lut_oc1[i, lmd_num, m, n, 4],  lut_oc1[i, lmd_num, m, n, 9],   # 0th ref
                        lut_oc1[i, lmd_num, m, n, 24], lut_oc1[i, lmd_num, m, n, 29])  # 0th ref
                    Ete2, Etm2, delta_phase2 = E_field_cal(
                        Ete, Etm, delta_phase,
                        lut_oc1[i, lmd_num, m, n, 2], lut_oc1[i, lmd_num, m, n, 7],   # -2nd ref
                        lut_oc1[i, lmd_num, m, n, 22], lut_oc1[i, lmd_num, m, n, 27]) # -2nd ref
                    Ete3, Etm3, delta_phase3 = E_field_cal(
                        Ete, Etm, delta_phase,
                        lut_oc1[i, lmd_num, m, n, 13], lut_oc1[i, lmd_num, m, n, 18],  # -1st tran
                        lut_oc1[i, lmd_num, m, n, 33], lut_oc1[i, lmd_num, m, n, 38])  # -1st tran
                    efficiency1 = (Ete1 * Ete1 + Etm1 * Etm1) * math.cos(lut_oc1[i, lmd_num, m, n, 0].real) / math.cos(theta.real)
                    efficiency2 = (Ete2 * Ete2 + Etm2 * Etm2) * math.cos(lut_oc2[i, lmd_num, m, n, 0].real) / math.cos(theta.real)
                    efficiency3 = (Ete3 * Ete3 + Etm3 * Etm3) * math.cos(lut_ic1[lmd_num, m, n, 0].real) / math.cos(theta.real) / n_g
                    ener1 = ener * efficiency1
                    ener2 = ener * efficiency2
                    ener3 = ener * efficiency3
                    rand_val = get_uniform_random_number(rng_states, idx)
                    if rand_val <= efficiency1 and ener1 > threshold:
                        theta = lut_oc1[i, lmd_num, m, n, 0]
                        phi = lut_oc1[i, lmd_num, m, n, 1]
                        norm = math.sqrt(Ete1*Ete1 + Etm1*Etm1)
                        Ete = Ete1/norm
                        Etm = Etm1/norm
                        delta_phase = delta_phase1 + lut_TIR[lmd_num, m, n, 1] # phase shift due to TIR (TM-TE)
                        gap_x = lut_gap[lmd_num, m, n, 2] # assign the propagation distance
                        gap_y = lut_gap[lmd_num, m, n, 3]
                        x += gap_x
                        y += gap_y
                        ener = ener1 * 1.0
                        region_state = 4
                    elif rand_val <= efficiency1+efficiency2 and ener2 > threshold:
                        theta = lut_oc2[i, lmd_num, m, n, 0]
                        phi = lut_oc2[i, lmd_num, m, n, 2]
                        norm = math.sqrt(Ete2*Ete2 + Etm2*Etm2)
                        Ete = Ete2/norm
                        Etm = Etm2/norm
                        delta_phase = delta_phase2 + lut_TIR[lmd_num, m, n, 3] # phase shift due to TIR (TM-TE)
                        gap_x = lut_gap[lmd_num, m, n, 6] # assign the propagation distance
                        gap_y = lut_gap[lmd_num, m, n, 7]
                        x += gap_x
                        y += gap_y
                        ener = ener2 * 1.0
                        region_state = 5
                    elif rand_val <= efficiency1+efficiency2+efficiency3 and ener3 > threshold:
                        if is_inside_or_on_edge_4d(x, y, eff_reg_FOV, m, n):
                            xmin = eff_reg_FOV_range[m, n, 0]
                            xmax = eff_reg_FOV_range[m, n, 1]
                            ymin = eff_reg_FOV_range[m, n, 2]
                            ymax = eff_reg_FOV_range[m, n, 3]
                            add_to_EB_atomic_val(matrix_EB[lmd_num,:,:,:,:], m, n, x, y, xmin, xmax, ymin, ymax, 1.0)
                            return
                        else:
                            return
                    else:
                        return
                    break
            if not hit:
                x += gap_x
                y += gap_y
                delta_phase += 2*lut_TIR[lmd_num, m, n, 1]

        elif region_state == 5: # exist FC and interact with OC (opposite OC direction)
            hit = False
            for i in range(len(OC_offset)-1):
                start_OC = OC_offset[i]
                end_OC = OC_offset[i+1]
                if is_inside_or_on_edge(x, y, OC, start_OC, end_OC):
                    hit = True
                    Ete1, Etm1, delta_phase1 = E_field_cal(
                        Ete, Etm, delta_phase,
                        lut_oc2[i, lmd_num, m, n, 6], lut_oc2[i, lmd_num, m, n, 11],   # +2nd ref
                        lut_oc2[i, lmd_num, m, n, 26], lut_oc2[i, lmd_num, m, n, 31])  # +2nd ref
                    Ete2, Etm2, delta_phase2 = E_field_cal(
                        Ete, Etm, delta_phase,
                        lut_oc2[i, lmd_num, m, n, 4],  lut_oc2[i, lmd_num, m, n, 9],   # 0th ref
                        lut_oc2[i, lmd_num, m, n, 24], lut_oc2[i, lmd_num, m, n, 29])  # 0th ref
                    Ete3, Etm3, delta_phase3 = E_field_cal(
                        Ete, Etm, delta_phase,
                        lut_oc2[i, lmd_num, m, n, 15], lut_oc2[i, lmd_num, m, n, 20],  # +1st tran
                        lut_oc2[i, lmd_num, m, n, 35], lut_oc2[i, lmd_num, m, n, 40])  # +1st tran
                    efficiency1 = (Ete1 * Ete1 + Etm1 * Etm1) * math.cos(lut_oc1[i, lmd_num, m, n, 0].real) / math.cos(theta.real)
                    efficiency2 = (Ete2 * Ete2 + Etm2 * Etm2) * math.cos(lut_oc2[i, lmd_num, m, n, 0].real) / math.cos(theta.real)
                    efficiency3 = (Ete3 * Ete3 + Etm3 * Etm3) * math.cos(lut_ic1[lmd_num, m, n, 0].real) / math.cos(theta.real) / n_g
                    ener1 = ener * efficiency1
                    ener2 = ener * efficiency2
                    ener3 = ener * efficiency3
                    rand_val = get_uniform_random_number(rng_states, idx)
                    if rand_val <= efficiency1 and ener1 > threshold:
                        theta = lut_oc1[i, lmd_num, m, n, 0]
                        phi = lut_oc1[i, lmd_num, m, n, 1]
                        norm = math.sqrt(Ete1*Ete1 + Etm1*Etm1)
                        Ete = Ete1/norm
                        Etm = Etm1/norm
                        delta_phase = delta_phase1 + lut_TIR[lmd_num, m, n, 1] # phase shift due to TIR (TM-TE)
                        gap_x = lut_gap[lmd_num, m, n, 2] # assign the propagation distance
                        gap_y = lut_gap[lmd_num, m, n, 3]
                        x += gap_x
                        y += gap_y
                        ener = ener1 * 1.0
                        region_state = 4
                    elif rand_val <= efficiency1+efficiency2 and ener2 > threshold:
                        theta = lut_oc2[i, lmd_num, m, n, 0]
                        phi = lut_oc2[i, lmd_num, m, n, 2]
                        norm = math.sqrt(Ete2*Ete2 + Etm2*Etm2)
                        Ete = Ete2/norm
                        Etm = Etm2/norm
                        delta_phase = delta_phase2 + lut_TIR[lmd_num, m, n, 3] # phase shift due to TIR (TM-TE)
                        gap_x = lut_gap[lmd_num, m, n, 6] # assign the propagation distance
                        gap_y = lut_gap[lmd_num, m, n, 7]
                        x += gap_x
                        y += gap_y
                        ener = ener2 * 1.0
                        region_state = 5
                    elif rand_val <= efficiency1+efficiency2+efficiency3 and ener3 > threshold:
                        if is_inside_or_on_edge_4d(x, y, eff_reg_FOV, m, n):
                            xmin = eff_reg_FOV_range[m, n, 0]
                            xmax = eff_reg_FOV_range[m, n, 1]
                            ymin = eff_reg_FOV_range[m, n, 2]
                            ymax = eff_reg_FOV_range[m, n, 3]
                            add_to_EB_atomic_val(matrix_EB[lmd_num,:,:,:,:], m, n, x, y, xmin, xmax, ymin, ymax, 1.0)
                            return
                        else:
                            return
                    else:
                        return
                    break
            if not hit:
                return


@cuda.jit
def process_rays_fc_kernel_pro( # probability tracing
    x_v, y_v, gap_x_v, gap_y_v, pol_v, azi_v, m_v, n_v, te_v, tm_v, delta_phase_v,
    rng_states,
    IC, FC, FC_offset, n_g,
    eff_reg1, eff_reg2,
    lut_ic1, lut_ic2, lut_ic3, lut_fc1, lut_fc2,
    lut_TIR, lut_gap,
    matrix_FOV):

    idx = cuda.grid(1)
    N = x_v.shape[0]

    if idx >= N:
        return
    
    # Read ray property in vectors
    x = x_v[idx] # position of ray
    y = y_v[idx]
    gap_x = gap_x_v[idx] # propagation distance for each TIR
    gap_y = gap_y_v[idx]
    theta = pol_v[idx] # polar and azimuth angle of ray (global coordinates)
    phi = azi_v[idx]
    m = int(m_v[idx]) # LUT position
    n = int(n_v[idx])
    Ete = te_v[idx] # E-field
    Etm = tm_v[idx]
    delta_phase = delta_phase_v[idx] # phase difference between te and tm (phi_tm - phi_te)  range: (-pi, pi]

    ener = 1
    threshold = 1e-15

    # have not interact with IC yet
    # check LUT and change propagation direction and electrical field here (TIR phase shift considered)
    # the parameters need to change: Es, Ep, delta_phase, theta, phi, gap_x, gap_y, energy
    # order to FC direction
    Ete1, Etm1, delta_phase1 = E_field_cal(
        Ete, Etm, delta_phase,
        lut_ic1[m, n, 13], lut_ic1[m, n, 18], # unit TE in, TE&TM out (-1st tran) (to FC)
        lut_ic1[m, n, 33], lut_ic1[m, n, 38]) # unit TM in, TE&TM out (-1st tran) (to FC)
    Ete2, Etm2, delta_phase2 = E_field_cal(
        Ete, Etm, delta_phase,
        lut_ic1[m, n, 15], lut_ic1[m, n, 20], # unit TE in, TE&TM out (+1st tran) (opposite to FC)
        lut_ic1[m, n, 35], lut_ic1[m, n, 40]) # unit TM in, TE&TM out (+1st tran) (opposite to FC)
    efficiency1 = (Ete1 * Ete1 + Etm1 * Etm1) * math.cos(lut_ic2[m, n, 0].real) / math.cos(lut_ic1[m, n, 0].real) * n_g
    efficiency2 = (Ete2 * Ete2 + Etm2 * Etm2) * math.cos(lut_ic3[m, n, 0].real) / math.cos(lut_ic1[m, n, 0].real) * n_g
    rand_val = get_uniform_random_number(rng_states, idx)
    if rand_val <= efficiency1:
        theta = lut_ic2[m, n, 0]
        phi = lut_ic2[m, n, 1]
        norm = math.sqrt(Ete1*Ete1 + Etm1*Etm1)
        Ete = Ete1/norm
        Etm = Etm1/norm
        delta_phase = delta_phase1 + lut_TIR[m, n, 0] # phase shift due to TIR (TM-TE)
        gap_x = lut_gap[m, n, 0] # assign the propagation distance
        gap_y = lut_gap[m, n, 1]
        x += gap_x
        y += gap_y
        ener *= efficiency1
        if not is_inside_or_on_edge(x, y, IC, 0, IC.shape[0]): # two situations will occur: 1. exist IC region 2. does not exist IC yet
            region_state = 2
            cuda.atomic.add(matrix_FOV, (n, m, 0), 1.0)
        else:
            region_state = 0
    elif rand_val <= efficiency1+efficiency2:
        theta = lut_ic3[m, n, 0]
        phi = lut_ic3[m, n, 1]
        norm = math.sqrt(Ete2*Ete2 + Etm2*Etm2)
        Ete = Ete2/norm
        Etm = Etm2/norm
        delta_phase = delta_phase2 + lut_TIR[m, n, 2] # phase shift due to TIR (TM-TE)
        gap_x = lut_gap[m, n, 4] # assign the propagation distance
        gap_y = lut_gap[m, n, 5]
        x += gap_x
        y += gap_y
        ener *= efficiency2
        if not is_inside_or_on_edge(x, y, IC, 0, IC.shape[0]): # two situations will occur: 1. exist IC region 2. does not exist IC yet
            return
        else:
            region_state = 1
    else:
        return
    
    for _ in range(1e5):
        if not is_inside_or_on_edge(x, y, eff_reg1, 0, eff_reg1.shape[0]):
            return
        
        if region_state == 0:
            Ete1, Etm1, delta_phase1 = E_field_cal(
                Ete, Etm, delta_phase,
                lut_ic2[m, n, 4], lut_ic2[m, n, 9], # unit TE in, TE&TM out (0th ref) (to FC)
                lut_ic2[m, n, 24], lut_ic2[m, n, 29]) # unit TM in, TE&TM out (0th ref) (to FC)
            Ete2, Etm2, delta_phase2 = E_field_cal(
                Ete, Etm, delta_phase,
                lut_ic2[m, n, 6], lut_ic2[m, n, 11], # unit TE in, TE&TM out (+2nd ref) (opposite to FC)
                lut_ic2[m, n, 26], lut_ic2[m, n, 31]) # unit TM in, TE&TM out (+2nd ref) (opposite to FC)
            efficiency1 = (Ete1 * Ete1 + Etm1 * Etm1) * math.cos(lut_ic2[m, n, 0].real) / math.cos(theta.real) # 0th ref
            efficiency2 = (Ete2 * Ete2 + Etm2 * Etm2) * math.cos(lut_ic3[m, n, 0].real) / math.cos(theta.real) # +2nd ref
            rand_val = get_uniform_random_number(rng_states, idx)
            if rand_val <= efficiency1:
                theta = lut_ic2[m, n, 0]
                phi = lut_ic2[m, n, 1]
                norm = math.sqrt(Ete1*Ete1 + Etm1*Etm1)
                Ete = Ete1/norm
                Etm = Etm1/norm
                delta_phase = delta_phase1 + lut_TIR[m, n, 0] # phase shift due to TIR (TM-TE)
                gap_x = lut_gap[m, n, 0] # assign the propagation distance
                gap_y = lut_gap[m, n, 1]
                x += gap_x
                y += gap_y
                ener *= efficiency1
                if not is_inside_or_on_edge(x, y, IC, 0, IC.shape[0]): # two situations will occur: 1. exist IC region 2. does not exist IC yet
                    region_state = 2
                    cuda.atomic.add(matrix_FOV, (n, m, 0), 1.0)
                else:
                    region_state = 0
            elif rand_val <= efficiency1+efficiency2:
                theta = lut_ic3[m, n, 0]
                phi = lut_ic3[m, n, 1]
                norm = math.sqrt(Ete2*Ete2 + Etm2*Etm2)
                Ete = Ete2/norm
                Etm = Etm2/norm
                delta_phase = delta_phase2 + lut_TIR[m, n, 2] # phase shift due to TIR (TM-TE)
                gap_x = lut_gap[m, n, 4] # assign the propagation distance
                gap_y = lut_gap[m, n, 5]
                x += gap_x
                y += gap_y
                ener *= efficiency2
                if not is_inside_or_on_edge(x, y, IC, 0, IC.shape[0]): # two situations will occur: 1. exist IC region 2. does not exist IC yet
                    return
                else:
                    region_state = 1
            else:
                return
        
        elif region_state == 1:
            Ete1, Etm1, delta_phase1 = E_field_cal(
                Ete, Etm, delta_phase,
                lut_ic3[m, n, 2], lut_ic3[m, n, 22], # unit TE in, TE&TM out (-2nd ref) (to FC)
                lut_ic3[m, n, 7], lut_ic3[m, n, 27]) # unit TM in, TE&TM out (-2nd ref) (to FC)
            Ete2, Etm2, delta_phase2 = E_field_cal(
                Ete, Etm, delta_phase,
                lut_ic3[m, n, 4], lut_ic3[m, n, 9], # unit TE in, TE&TM out (0th ref) (opposite to FC)
                lut_ic3[m, n, 24], lut_ic3[m, n, 29]) # unit TM in, TE&TM out (0th ref) (opposite to FC)
            efficiency1 = (Ete1 * Ete1 + Etm1 * Etm1) * math.cos(lut_ic2[m, n, 0].real) / math.cos(theta.real)
            efficiency2 = (Ete2 * Ete2 + Etm2 * Etm2) * math.cos(lut_ic3[m, n, 0].real) / math.cos(theta.real)
            rand_val = get_uniform_random_number(rng_states, idx)
            if rand_val <= efficiency1:
                theta = lut_ic2[m, n, 0]
                phi = lut_ic2[m, n, 1]
                norm = math.sqrt(Ete1*Ete1 + Etm1*Etm1)
                Ete = Ete1/norm
                Etm = Etm1/norm
                delta_phase = delta_phase1 + lut_TIR[m, n, 0] # phase shift due to TIR (TM-TE)
                gap_x = lut_gap[m, n, 0] # assign the propagation distance
                gap_y = lut_gap[m, n, 1]
                x += gap_x
                y += gap_y
                ener *= efficiency1
                if not is_inside_or_on_edge(x, y, IC, 0, IC.shape[0]): # two situations will occur: 1. exist IC region 2. does not exist IC yet
                    region_state = 2
                    cuda.atomic.add(matrix_FOV, (n, m, 0), 1.0)
                else:
                    region_state = 0
            elif rand_val <= efficiency1+efficiency2:
                theta = lut_ic3[m, n, 0]
                phi = lut_ic3[m, n, 1]
                norm = math.sqrt(Ete2*Ete2 + Etm2*Etm2)
                Ete = Ete2/norm
                Etm = Etm2/norm
                delta_phase = delta_phase2 + lut_TIR[m, n, 2] # phase shift due to TIR (TM-TE)
                gap_x = lut_gap[m, n, 4] # assign the propagation distance
                gap_y = lut_gap[m, n, 5]
                x += gap_x
                y += gap_y
                ener *= efficiency2
                if not is_inside_or_on_edge(x, y, IC, 0, IC.shape[0]): # two situations will occur: 1. exist IC region 2. does not exist IC yet
                    return
                else:
                    region_state = 1
            else:
                return
        
        elif region_state == 2: # exist IC and interact with FC (to FC direction)
            hit = False
            for i in range(len(FC_offset)-1):
                start_FC = FC_offset[i]
                end_FC = FC_offset[i+1]
                if is_inside_or_on_edge(x, y, FC, start_FC, end_FC):
                    hit = True
                    Ete1, Etm1, delta_phase1 = E_field_cal(
                        Ete, Etm, delta_phase,
                        lut_fc1[i, m, n, 3], lut_fc1[i, m, n, 6],   # unit TE in, TE&TM out (0th ref)
                        lut_fc1[i, m, n, 15], lut_fc1[i, m, n, 18]) # unit TM in, TE&TM out (0th ref)
                    Ete2, Etm2, delta_phase2 = E_field_cal(
                        Ete, Etm, delta_phase,
                        lut_fc1[i, m, n, 2], lut_fc1[i, m, n, 5],   # unit TE in, TE&TM out (-1st ref)
                        lut_fc1[i, m, n, 14], lut_fc1[i, m, n, 17]) # unit TM in, TE&TM out (-1st ref)
                    efficiency1 = (Ete1 * Ete1 + Etm1 * Etm1) * math.cos(lut_fc1[i, m, n, 0].real) / math.cos(theta.real)
                    efficiency2 = (Ete2 * Ete2 + Etm2 * Etm2) * math.cos(lut_fc2[i, m, n, 0].real) / math.cos(theta.real)
                    ener1 = ener * efficiency1
                    ener2 = ener * efficiency2
                    rand_val = get_uniform_random_number(rng_states, idx)
                    if rand_val <= efficiency1 and ener1 > threshold:
                        theta = lut_fc1[i, m, n, 0]
                        phi = lut_fc1[i, m, n, 1]
                        norm = math.sqrt(Ete1*Ete1 + Etm1*Etm1)
                        Ete = Ete1/norm
                        Etm = Etm1/norm
                        delta_phase = delta_phase1 + lut_TIR[m, n, 0] # phase shift due to TIR (TM-TE)
                        gap_x = lut_gap[m, n, 0] # assign the propagation distance
                        gap_y = lut_gap[m, n, 1]
                        x += gap_x
                        y += gap_y
                        ener = ener1 * 1.0
                        region_state = 2
                    elif rand_val <= efficiency1+efficiency2 and ener2 > threshold:
                        theta = lut_fc2[i, m, n, 0]
                        phi = lut_fc2[i, m, n, 1]
                        norm = math.sqrt(Ete2*Ete2 + Etm2*Etm2)
                        Ete = Ete2/norm
                        Etm = Etm2/norm
                        delta_phase = delta_phase2 + lut_TIR[m, n, 1] # phase shift due to TIR (TM-TE)
                        gap_x = lut_gap[m, n, 2] # assign the propagation distance
                        gap_y = lut_gap[m, n, 3]
                        x += gap_x
                        y += gap_y
                        ener = ener2 * 1.0
                        region_state = 3
                    else:
                        return
                    break
            if not hit:
                x += gap_x
                y += gap_y
                delta_phase += 2*lut_TIR[m, n, 0]
        
        elif region_state == 3: # exist IC and interact with FC (to OC direction)
            hit = False
            for i in range(len(FC_offset)-1):
                start_FC = FC_offset[i]
                end_FC = FC_offset[i+1]
                if is_inside_or_on_edge(x, y, FC, start_FC, end_FC):
                    hit = True
                    Ete1, Etm1, delta_phase1 = E_field_cal(
                        Ete, Etm, delta_phase,
                        lut_fc2[i, m, n, 4], lut_fc2[i, m, n, 7],   # unit TE in, TE&TM out (+1st ref)
                        lut_fc2[i, m, n, 16], lut_fc2[i, m, n, 19]) # unit TM in, TE&TM out (+1st ref)
                    Ete2, Etm2, delta_phase2 = E_field_cal(
                        Ete, Etm, delta_phase,
                        lut_fc2[i, m, n, 3], lut_fc2[i, m, n, 6],   # unit TE in, TE&TM out (0th ref)
                        lut_fc2[i, m, n, 15], lut_fc2[i, m, n, 18]) # unit TM in, TE&TM out (0th ref)
                    efficiency1 = (Ete1 * Ete1 + Etm1 * Etm1) * math.cos(lut_fc1[i, m, n, 0].real) / math.cos(theta.real)
                    efficiency2 = (Ete2 * Ete2 + Etm2 * Etm2) * math.cos(lut_fc2[i, m, n, 0].real) / math.cos(theta.real)
                    ener1 = ener * efficiency1
                    ener2 = ener * efficiency2
                    rand_val = get_uniform_random_number(rng_states, idx)
                    if rand_val <= efficiency1 and ener1 > threshold:
                        theta = lut_fc1[i, m, n, 0]
                        phi = lut_fc1[i, m, n, 1]
                        norm = math.sqrt(Ete1*Ete1 + Etm1*Etm1)
                        Ete = Ete1/norm
                        Etm = Etm1/norm
                        delta_phase = delta_phase1 + lut_TIR[m, n, 0] # phase shift due to TIR (TM-TE)
                        gap_x = lut_gap[m, n, 0] # assign the propagation distance
                        gap_y = lut_gap[m, n, 1]
                        x += gap_x
                        y += gap_y
                        ener = ener1 * 1.0
                        region_state = 2
                    elif rand_val <= efficiency1+efficiency2 and ener2 > threshold:
                        theta = lut_fc2[i, m, n, 0]
                        phi = lut_fc2[i, m, n, 1]
                        norm = math.sqrt(Ete2*Ete2 + Etm2*Etm2)
                        Ete = Ete2/norm
                        Etm = Etm2/norm
                        delta_phase = delta_phase2 + lut_TIR[m, n, 1] # phase shift due to TIR (TM-TE)
                        gap_x = lut_gap[m, n, 2] # assign the propagation distance
                        gap_y = lut_gap[m, n, 3]
                        x += gap_x
                        y += gap_y
                        ener = ener2 * 1.0
                        region_state = 3
                    else:
                        return
                    break
            if not hit:
                if not is_inside_or_on_edge(x, y, eff_reg2, 0, eff_reg2.shape[0]):
                    cuda.atomic.add(matrix_FOV, (n, m, 1), 1.0)
                    return
                else:
                    x += gap_x
                    y += gap_y
                    delta_phase += 2*lut_TIR[m, n, 1]


@cuda.jit
def process_rays_IC_kernel(vectors,
                           MAX_STEPS, 
                           IC,
                           lut_ic1, lut_ic2,
                           lut_TIR, lut_gap,
                           matrix_FOV):

    idx = cuda.grid(1)
    N = vectors.shape[0]

    if idx >= N:
        return
    
    # Read ray property in vectors
    x = vectors[idx, 0] # position of ray
    y = vectors[idx, 1]
    gap_x = vectors[idx, 2] # propagation distance for each TIR
    gap_y = vectors[idx, 3]
    theta = vectors[idx, 4] # polar and azimuth angle of ray (global coordinates)
    phi = vectors[idx, 5]
    m = int(vectors[idx, 6]) # LUT position
    n = int(vectors[idx, 7])
    Ete = vectors[idx, 8] # E-field
    Etm = vectors[idx, 9]
    delta_phase = vectors[idx, 10] # phase difference between te and tm (phi_tm - phi_te)  range: (-pi, pi]
    region_state = vectors[idx, 11] # region state
    flag = vectors[idx, 12] # 1: useful, 0: useless

    # check LUT and change propagation direction and electrical field here (TIR phase shift considered)
    # the parameters need to change: Es, Ep, delta_phase, theta, phi, gap_x, gap_y, energy
    theta = lut_ic2[m, n, 0]
    phi = lut_ic2[m, n, 1]
    Ete, Etm, delta_phase = E_field_cal(Ete, Etm, delta_phase,
                                        lut_ic1[m, n, 8], lut_ic1[m, n, 11], # unit TE in, TE&TM out (+1st tran)
                                        lut_ic1[m, n, 20], lut_ic1[m, n, 23]) # unit TM in, TE&TM out (+1st tran)
    delta_phase += lut_TIR[m, n, 0] # phase shift due to TIR (TM-TE)
    gap_x = lut_gap[m, n, 0] # assign the propagation distance
    gap_y = lut_gap[m, n, 1]
    x += gap_x
    y += gap_y
    
    # enter in waveguide
    for _ in range(int(MAX_STEPS)):
        # check if the ray is still inside the IC.
        if not is_inside_or_on_edge(x, y, IC, 0, IC.shape[0]): # two situations will occur: 1. exist IC region 2. does not exist IC yet
            efficiency = (Ete * Ete + Etm * Etm) * math.cos(lut_ic2[m, n, 0].real) / math.cos(lut_ic1[m, n, 0].real) * 1.52
            cuda.atomic.add(matrix_FOV, (n, m), efficiency)
            return
        else: # still inside IC
            # check LUT, change propagation direction and electrical field here (TIR phase shift considered)
            # the parameters need to change: Es, Ep, delta_phase, theta, phi, gap_x, gap_y, energy
            Ete, Etm, delta_phase = E_field_cal(
                Ete, Etm, delta_phase,
                lut_ic2[m, n, 3], lut_ic2[m, n, 6],   # unit TE in, TE&TM out (0th ref)
                lut_ic2[m, n, 15], lut_ic2[m, n, 18]) # unit TM in, TE&TM out (0th ref)
            delta_phase += lut_TIR[m, n, 0] # phase shift due to TIR (TM-TE)
            x += gap_x
            y += gap_y


@cuda.jit
def process_rays_IC_kernel2(vectors, useful_count_in, d_total_ray_counter, 
                            IC,
                            lut_ic1, lut_ic2, lut_ic3,
                            lut_TIR, lut_gap,
                            matrix_FOV):

    idx = cuda.grid(1)
    N = int(useful_count_in)

    if idx >= N:
        return
    
    # Read ray property in vectors
    x = vectors[idx, 0] # position of ray
    y = vectors[idx, 1]
    gap_x = vectors[idx, 2] # propagation distance for each TIR
    gap_y = vectors[idx, 3]
    theta = vectors[idx, 4] # polar and azimuth angle of ray (global coordinates)
    phi = vectors[idx, 5]
    m = int(vectors[idx, 6]) # LUT position
    n = int(vectors[idx, 7])
    Ete = vectors[idx, 8] # E-field
    Etm = vectors[idx, 9]
    delta_phase = vectors[idx, 10] # phase difference between te and tm (phi_tm - phi_te)  range: (-pi, pi]
    region_state = vectors[idx, 11] # region state
    flag = vectors[idx, 12] # 1: useful, 0: useless
    threshold = 1e-8
    n_g = 1.52

    if region_state == 0: # have not interact with IC yet
        # check LUT and change propagation direction and electrical field here (TIR phase shift considered)
        # the parameters need to change: Es, Ep, delta_phase, theta, phi, gap_x, gap_y, energy
        # order to FC direction
        theta = lut_ic2[m, n, 0]
        phi = lut_ic2[m, n, 1]
        Ete1, Etm1, delta_phase1 = E_field_cal(Ete, Etm, delta_phase,
                                               lut_ic1[m, n, 13], lut_ic1[m, n, 18], # unit TE in, TE&TM out (-1st tran) (to FC)
                                               lut_ic1[m, n, 33], lut_ic1[m, n, 38]) # unit TM in, TE&TM out (-1st tran) (to FC)
        delta_phase1 += lut_TIR[m, n, 0] # phase shift due to TIR (TM-TE)
        gap_x = lut_gap[m, n, 0] # assign the propagation distance
        gap_y = lut_gap[m, n, 1]
        x1 = x + gap_x
        y1 = y + gap_y
        if not is_inside_or_on_edge(x1, y1, IC, 0, IC.shape[0]): # two situations will occur: 1. exist IC region 2. does not exist IC yet
            efficiency = (Ete1 * Ete1 + Etm1 * Etm1) * math.cos(lut_ic2[m, n, 0].real) / math.cos(lut_ic1[m, n, 0].real) * n_g
            cuda.atomic.add(matrix_FOV, (n, m), efficiency)
            vectors[idx, 12] = 0
        else: # still inside IC (order to FC)
            vectors[idx, 0] = x1 # position
            vectors[idx, 1] = y1
            vectors[idx, 2] = gap_x # propagation direction
            vectors[idx, 3] = gap_y 
            vectors[idx, 4] = theta.real # polar and azimuth angles
            vectors[idx, 5] = phi.real
            vectors[idx, 6] = m # FOV
            vectors[idx, 7] = n
            vectors[idx, 8] = Ete1 # E-field
            vectors[idx, 9] = Etm1
            vectors[idx, 10] = delta_phase1
            vectors[idx, 11] = 1 # region state
            vectors[idx, 12] = 1 # flag
        
        # order to opposite FC direction
        theta = lut_ic3[m, n, 0]
        phi = lut_ic3[m, n, 1]
        gap_x = lut_gap[m, n, 4] # assign the propagation distance
        gap_y = lut_gap[m, n, 5]
        x2 = x + gap_x
        y2 = y + gap_y
        if is_inside_or_on_edge(x2, y2, IC, 0, IC.shape[0]): # two situations will occur: 1. exist IC region 2. does not exist IC yet
            Ete2, Etm2, delta_phase2 = E_field_cal(Ete, Etm, delta_phase,
                                                lut_ic1[m, n, 15], lut_ic1[m, n, 20], # unit TE in, TE&TM out (+1st tran) (opposite to FC)
                                                lut_ic1[m, n, 35], lut_ic1[m, n, 40]) # unit TM in, TE&TM out (+1st tran) (opposite to FC)
            delta_phase2 += lut_TIR[m, n, 2] # phase shift due to TIR (TM-TE)
            new_ray_index = cuda.atomic.add(d_total_ray_counter, 0, 1)
            vectors[new_ray_index, 0] = x2 # position
            vectors[new_ray_index, 1] = y2
            vectors[new_ray_index, 2] = gap_x # propagation direction
            vectors[new_ray_index, 3] = gap_y 
            vectors[new_ray_index, 4] = theta.real # polar and azimuth angles
            vectors[new_ray_index, 5] = phi.real
            vectors[new_ray_index, 6] = m # FOV
            vectors[new_ray_index, 7] = n
            vectors[new_ray_index, 8] = Ete2 # E-field
            vectors[new_ray_index, 9] = Etm2
            vectors[new_ray_index, 10] = delta_phase2
            vectors[new_ray_index, 11] = 2 # region state
            vectors[new_ray_index, 12] = 1 # flag
            return

    if region_state == 1: # propagate to FC direction
        theta = lut_ic2[m, n, 0]
        phi = lut_ic2[m, n, 1]
        Ete1, Etm1, delta_phase1 = E_field_cal(Ete, Etm, delta_phase,
                                               lut_ic2[m, n, 4], lut_ic2[m, n, 9], # unit TE in, TE&TM out (0th ref) (to FC)
                                               lut_ic2[m, n, 24], lut_ic2[m, n, 29]) # unit TM in, TE&TM out (0th ref) (to FC)
        delta_phase1 += lut_TIR[m, n, 0] # phase shift due to TIR (TM-TE)
        gap_x = lut_gap[m, n, 0] # assign the propagation distance
        gap_y = lut_gap[m, n, 1]
        x1 = x + gap_x
        y1 = y + gap_y
        if not is_inside_or_on_edge(x1, y1, IC, 0, IC.shape[0]): # two situations will occur: 1. exist IC region 2. does not exist IC yet
            efficiency = (Ete1 * Ete1 + Etm1 * Etm1) * math.cos(lut_ic2[m, n, 0].real) / math.cos(lut_ic1[m, n, 0].real) * n_g
            cuda.atomic.add(matrix_FOV, (n, m), efficiency)
            vectors[idx, 12] = 0
        else: # still inside IC (order to FC)
            efficiency = (Ete1 * Ete1 + Etm1 * Etm1) * math.cos(lut_ic2[m, n, 0].real) / math.cos(lut_ic1[m, n, 0].real) * n_g
            if efficiency < threshold:
                vectors[idx, 12] = 0 # flag
            else:
                vectors[idx, 0] = x1 # position
                vectors[idx, 1] = y1
                vectors[idx, 2] = gap_x # propagation direction
                vectors[idx, 3] = gap_y 
                vectors[idx, 4] = theta.real # polar and azimuth angles
                vectors[idx, 5] = phi.real
                vectors[idx, 6] = m # FOV
                vectors[idx, 7] = n
                vectors[idx, 8] = Ete1 # E-field
                vectors[idx, 9] = Etm1
                vectors[idx, 10] = delta_phase1
                vectors[idx, 11] = 1 # region state
                vectors[idx, 12] = 1 # flag
        
        # order to opposite FC direction
        theta = lut_ic3[m, n, 0]
        phi = lut_ic3[m, n, 1]
        gap_x = lut_gap[m, n, 4] # assign the propagation distance
        gap_y = lut_gap[m, n, 5]
        x2 = x + gap_x
        y2 = y + gap_y
        if is_inside_or_on_edge(x2, y2, IC, 0, IC.shape[0]): # two situations will occur: 1. exist IC region 2. does not exist IC yet
            Ete2, Etm2, delta_phase2 = E_field_cal(Ete, Etm, delta_phase,
                                                   lut_ic2[m, n, 6], lut_ic2[m, n, 11], # unit TE in, TE&TM out (+2nd ref) (opposite to FC)
                                                   lut_ic2[m, n, 26], lut_ic2[m, n, 31]) # unit TM in, TE&TM out (+2nd ref) (opposite to FC)
            efficiency = (Ete2 * Ete2 + Etm2 * Etm2) * math.cos(lut_ic3[m, n, 0].real) / math.cos(lut_ic1[m, n, 0].real) * n_g
            if efficiency < threshold:
                return
            delta_phase2 += lut_TIR[m, n, 2] # phase shift due to TIR (TM-TE)
            new_ray_index = cuda.atomic.add(d_total_ray_counter, 0, 1)
            vectors[new_ray_index, 0] = x2 # position
            vectors[new_ray_index, 1] = y2
            vectors[new_ray_index, 2] = gap_x # propagation direction
            vectors[new_ray_index, 3] = gap_y 
            vectors[new_ray_index, 4] = theta.real # polar and azimuth angles
            vectors[new_ray_index, 5] = phi.real
            vectors[new_ray_index, 6] = m # FOV
            vectors[new_ray_index, 7] = n
            vectors[new_ray_index, 8] = Ete2 # E-field
            vectors[new_ray_index, 9] = Etm2
            vectors[new_ray_index, 10] = delta_phase2
            vectors[new_ray_index, 11] = 2 # region state
            vectors[new_ray_index, 12] = 1 # flag
            return
    
    if region_state == 2: # propagate to opposite FC direction
        theta = lut_ic3[m, n, 0]
        phi = lut_ic3[m, n, 1]
        Ete1, Etm1, delta_phase1 = E_field_cal(Ete, Etm, delta_phase,
                                               lut_ic3[m, n, 4], lut_ic3[m, n, 9], # unit TE in, TE&TM out (0th ref) (opposite to FC)
                                               lut_ic3[m, n, 24], lut_ic3[m, n, 29]) # unit TM in, TE&TM out (0th ref) (opposite to FC)
        delta_phase1 += lut_TIR[m, n, 2] # phase shift due to TIR (TM-TE)
        gap_x = lut_gap[m, n, 4] # assign the propagation distance
        gap_y = lut_gap[m, n, 5]
        x1 = x + gap_x
        y1 = y + gap_y
        if is_inside_or_on_edge(x1, y1, IC, 0, IC.shape[0]): # two situations will occur: 1. exist IC region 2. does not exist IC yet
            efficiency = (Ete1 * Ete1 + Etm1 * Etm1) * math.cos(lut_ic3[m, n, 0].real) / math.cos(lut_ic1[m, n, 0].real) * n_g
            if efficiency < threshold:
                vectors[idx, 12] = 0 # flag
            else:
                vectors[idx, 0] = x1 # position
                vectors[idx, 1] = y1
                vectors[idx, 2] = gap_x # propagation direction
                vectors[idx, 3] = gap_y 
                vectors[idx, 4] = theta.real # polar and azimuth angles
                vectors[idx, 5] = phi.real
                vectors[idx, 6] = m # FOV
                vectors[idx, 7] = n
                vectors[idx, 8] = Ete1 # E-field
                vectors[idx, 9] = Etm1
                vectors[idx, 10] = delta_phase1
                vectors[idx, 11] = 2 # region state
                vectors[idx, 12] = 1 # flag
        else:
            vectors[idx, 12] = 0 # flag
        
        # order to FC direction
        theta = lut_ic2[m, n, 0]
        phi = lut_ic2[m, n, 1]
        gap_x = lut_gap[m, n, 0] # assign the propagation distance
        gap_y = lut_gap[m, n, 1]
        x2 = x + gap_x
        y2 = y + gap_y
        if is_inside_or_on_edge(x2, y2, IC, 0, IC.shape[0]): # two situations will occur: 1. exist IC region 2. does not exist IC yet
            Ete2, Etm2, delta_phase2 = E_field_cal(Ete, Etm, delta_phase,
                                                   lut_ic3[m, n, 2], lut_ic3[m, n, 22], # unit TE in, TE&TM out (-2nd ref) (to FC)
                                                   lut_ic3[m, n, 7], lut_ic3[m, n, 27]) # unit TM in, TE&TM out (-2nd ref) (to FC)
            efficiency = (Ete2 * Ete2 + Etm2 * Etm2) * math.cos(lut_ic2[m, n, 0].real) / math.cos(lut_ic1[m, n, 0].real) * n_g
            if efficiency < threshold:
                return
            delta_phase2 += lut_TIR[m, n, 0] # phase shift due to TIR (TM-TE)
            new_ray_index = cuda.atomic.add(d_total_ray_counter, 0, 1)
            vectors[new_ray_index, 0] = x2 # position
            vectors[new_ray_index, 1] = y2
            vectors[new_ray_index, 2] = gap_x # propagation direction
            vectors[new_ray_index, 3] = gap_y 
            vectors[new_ray_index, 4] = theta.real # polar and azimuth angles
            vectors[new_ray_index, 5] = phi.real
            vectors[new_ray_index, 6] = m # FOV
            vectors[new_ray_index, 7] = n
            vectors[new_ray_index, 8] = Ete2 # E-field
            vectors[new_ray_index, 9] = Etm2
            vectors[new_ray_index, 10] = delta_phase2
            vectors[new_ray_index, 11] = 1 # region state
            vectors[new_ray_index, 12] = 1 # flag
            return
        else:
            Ete2, Etm2, delta_phase2 = E_field_cal(Ete, Etm, delta_phase,
                                                   lut_ic3[m, n, 2], lut_ic3[m, n, 22], # unit TE in, TE&TM out (-2nd ref) (to FC)
                                                   lut_ic3[m, n, 7], lut_ic3[m, n, 27]) # unit TM in, TE&TM out (-2nd ref) (to FC)
            efficiency = (Ete2 * Ete2 + Etm2 * Etm2) * math.cos(lut_ic2[m, n, 0].real) / math.cos(lut_ic1[m, n, 0].real) * n_g
            cuda.atomic.add(matrix_FOV, (n, m), efficiency)


@cuda.jit
def process_rays_IC_kernel_pro( # probability tracing
    vectors,
    IC,
    lut_ic1, lut_ic2, lut_ic3,
    lut_TIR, lut_gap,
    rng_states,
    matrix_FOV):

    idx = cuda.grid(1)
    N = vectors.shape[0]

    if idx >= N:
        return
    
    # Read ray property in vectors
    x = vectors[idx, 0] # position of ray
    y = vectors[idx, 1]
    gap_x = vectors[idx, 2] # propagation distance for each TIR
    gap_y = vectors[idx, 3]
    theta = vectors[idx, 4] # polar and azimuth angle of ray (global coordinates)
    phi = vectors[idx, 5]
    m = int(vectors[idx, 6]) # LUT position
    n = int(vectors[idx, 7])
    Ete = vectors[idx, 8] # E-field
    Etm = vectors[idx, 9]
    delta_phase = vectors[idx, 10] # phase difference between te and tm (phi_tm - phi_te)  range: (-pi, pi]
    n_g = 1.52

    # have not interact with IC yet
    # check LUT and change propagation direction and electrical field here (TIR phase shift considered)
    # the parameters need to change: Es, Ep, delta_phase, theta, phi, gap_x, gap_y, energy
    # order to FC direction
    Ete1, Etm1, delta_phase1 = E_field_cal(
        Ete, Etm, delta_phase,
        lut_ic1[m, n, 13], lut_ic1[m, n, 18], # unit TE in, TE&TM out (-1st tran) (to FC)
        lut_ic1[m, n, 33], lut_ic1[m, n, 38]) # unit TM in, TE&TM out (-1st tran) (to FC)
    Ete2, Etm2, delta_phase2 = E_field_cal(
        Ete, Etm, delta_phase,
        lut_ic1[m, n, 15], lut_ic1[m, n, 20], # unit TE in, TE&TM out (+1st tran) (opposite to FC)
        lut_ic1[m, n, 35], lut_ic1[m, n, 40]) # unit TM in, TE&TM out (+1st tran) (opposite to FC)
    efficiency1 = (Ete1 * Ete1 + Etm1 * Etm1) * math.cos(lut_ic2[m, n, 0].real) / math.cos(lut_ic1[m, n, 0].real) * n_g
    efficiency2 = (Ete2 * Ete2 + Etm2 * Etm2) * math.cos(lut_ic3[m, n, 0].real) / math.cos(lut_ic1[m, n, 0].real) * n_g
    rand_val = get_uniform_random_number(rng_states, idx)
    if rand_val <= efficiency1:
        state = 0
        theta = lut_ic2[m, n, 0]
        phi = lut_ic2[m, n, 1]
        norm = math.sqrt(Ete1*Ete1 + Etm1*Etm1)
        Ete = Ete1/norm
        Etm = Etm1/norm
        delta_phase = delta_phase1 + lut_TIR[m, n, 0] # phase shift due to TIR (TM-TE)
        gap_x = lut_gap[m, n, 0] # assign the propagation distance
        gap_y = lut_gap[m, n, 1]
        x += gap_x
        y += gap_y
        if not is_inside_or_on_edge(x, y, IC, 0, IC.shape[0]): # two situations will occur: 1. exist IC region 2. does not exist IC yet
            cuda.atomic.add(matrix_FOV, (n, m), 1)
            return
    elif rand_val <= efficiency1+efficiency2:
        state = 1
        theta = lut_ic3[m, n, 0]
        phi = lut_ic3[m, n, 1]
        norm = math.sqrt(Ete2*Ete2 + Etm2*Etm2)
        Ete = Ete2/norm
        Etm = Etm2/norm
        delta_phase = delta_phase2 + lut_TIR[m, n, 2] # phase shift due to TIR (TM-TE)
        gap_x = lut_gap[m, n, 4] # assign the propagation distance
        gap_y = lut_gap[m, n, 5]
        x += gap_x
        y += gap_y
        if not is_inside_or_on_edge(x, y, IC, 0, IC.shape[0]): # two situations will occur: 1. exist IC region 2. does not exist IC yet
            return
    else:
        return
    
    for _ in range(100):
        if state == 0:
            Ete1, Etm1, delta_phase1 = E_field_cal(
                Ete, Etm, delta_phase,
                lut_ic2[m, n, 4], lut_ic2[m, n, 9], # unit TE in, TE&TM out (0th ref) (to FC)
                lut_ic2[m, n, 24], lut_ic2[m, n, 29]) # unit TM in, TE&TM out (0th ref) (to FC)
            Ete2, Etm2, delta_phase2 = E_field_cal(
                Ete, Etm, delta_phase,
                lut_ic2[m, n, 6], lut_ic2[m, n, 11], # unit TE in, TE&TM out (+2nd ref) (opposite to FC)
                lut_ic2[m, n, 26], lut_ic2[m, n, 31]) # unit TM in, TE&TM out (+2nd ref) (opposite to FC)
            efficiency1 = (Ete1 * Ete1 + Etm1 * Etm1) * math.cos(lut_ic2[m, n, 0].real) / math.cos(theta.real)
            efficiency2 = (Ete2 * Ete2 + Etm2 * Etm2) * math.cos(lut_ic3[m, n, 0].real) / math.cos(theta.real)
            rand_val = get_uniform_random_number(rng_states, idx)
            if rand_val <= efficiency1:
                state = 0
                theta = lut_ic2[m, n, 0]
                phi = lut_ic2[m, n, 1]
                norm = math.sqrt(Ete1*Ete1 + Etm1*Etm1)
                Ete = Ete1/norm
                Etm = Etm1/norm
                delta_phase = delta_phase1 + lut_TIR[m, n, 0] # phase shift due to TIR (TM-TE)
                gap_x = lut_gap[m, n, 0] # assign the propagation distance
                gap_y = lut_gap[m, n, 1]
                x += gap_x
                y += gap_y
                if not is_inside_or_on_edge(x, y, IC, 0, IC.shape[0]): # two situations will occur: 1. exist IC region 2. does not exist IC yet
                    cuda.atomic.add(matrix_FOV, (n, m), 1)
                    return
            elif rand_val <= efficiency1+efficiency2:
                state = 1
                theta = lut_ic3[m, n, 0]
                phi = lut_ic3[m, n, 1]
                norm = math.sqrt(Ete2*Ete2 + Etm2*Etm2)
                Ete = Ete2/norm
                Etm = Etm2/norm
                delta_phase = delta_phase2 + lut_TIR[m, n, 2] # phase shift due to TIR (TM-TE)
                gap_x = lut_gap[m, n, 4] # assign the propagation distance
                gap_y = lut_gap[m, n, 5]
                x += gap_x
                y += gap_y
                if not is_inside_or_on_edge(x, y, IC, 0, IC.shape[0]): # two situations will occur: 1. exist IC region 2. does not exist IC yet
                    return
            else:
                return
        elif state == 1:
            Ete1, Etm1, delta_phase1 = E_field_cal(
                Ete, Etm, delta_phase,
                lut_ic3[m, n, 2], lut_ic3[m, n, 22], # unit TE in, TE&TM out (-2nd ref) (to FC)
                lut_ic3[m, n, 7], lut_ic3[m, n, 27]) # unit TM in, TE&TM out (-2nd ref) (to FC)
            Ete2, Etm2, delta_phase2 = E_field_cal(
                Ete, Etm, delta_phase,
                lut_ic3[m, n, 4], lut_ic3[m, n, 9], # unit TE in, TE&TM out (0th ref) (opposite to FC)
                lut_ic3[m, n, 24], lut_ic3[m, n, 29]) # unit TM in, TE&TM out (0th ref) (opposite to FC)
            efficiency1 = (Ete1 * Ete1 + Etm1 * Etm1) * math.cos(lut_ic2[m, n, 0].real) / math.cos(theta.real)
            efficiency2 = (Ete2 * Ete2 + Etm2 * Etm2) * math.cos(lut_ic3[m, n, 0].real) / math.cos(theta.real)
            rand_val = get_uniform_random_number(rng_states, idx)
            if rand_val <= efficiency1:
                state = 0
                theta = lut_ic2[m, n, 0]
                phi = lut_ic2[m, n, 1]
                norm = math.sqrt(Ete1*Ete1 + Etm1*Etm1)
                Ete = Ete1/norm
                Etm = Etm1/norm
                delta_phase = delta_phase1 + lut_TIR[m, n, 0] # phase shift due to TIR (TM-TE)
                gap_x = lut_gap[m, n, 0] # assign the propagation distance
                gap_y = lut_gap[m, n, 1]
                x += gap_x
                y += gap_y
                if not is_inside_or_on_edge(x, y, IC, 0, IC.shape[0]): # two situations will occur: 1. exist IC region 2. does not exist IC yet
                    cuda.atomic.add(matrix_FOV, (n, m), 1)
                    return
            elif rand_val <= efficiency1+efficiency2:
                state = 1
                theta = lut_ic3[m, n, 0]
                phi = lut_ic3[m, n, 1]
                norm = math.sqrt(Ete2*Ete2 + Etm2*Etm2)
                Ete = Ete2/norm
                Etm = Etm2/norm
                delta_phase = delta_phase2 + lut_TIR[m, n, 2] # phase shift due to TIR (TM-TE)
                gap_x = lut_gap[m, n, 4] # assign the propagation distance
                gap_y = lut_gap[m, n, 5]
                x += gap_x
                y += gap_y
                if not is_inside_or_on_edge(x, y, IC, 0, IC.shape[0]): # two situations will occur: 1. exist IC region 2. does not exist IC yet
                    return
            else:
                return


@cuda.jit
def process_rays_kernel_PVG_1d(
    vectors,
    MAX_STEPS, 
    IC, OC, OC_offset,
    eff_reg, eff_reg_FOV, eff_reg_FOV_range,
    lut_ic1, lut_ic2, lut_oc,
    lut_TIR, lut_gap,
    matrix_EB):

    idx = cuda.grid(1)
    N = vectors.shape[0]

    if idx >= N:
        return
    
    # Read ray property in vectors
    x = vectors[idx, 0] # position of ray
    y = vectors[idx, 1]
    gap_x = vectors[idx, 2] # propagation distance for each TIR
    gap_y = vectors[idx, 3]
    m = int(vectors[idx, 4]) # LUT position
    n = int(vectors[idx, 5])
    Ete = vectors[idx, 6] # E-field
    Etm = vectors[idx, 7]
    delta_phase = vectors[idx, 8] # phase difference between te and tm (phi_tm - phi_te)  range: (-pi, pi]
    rgb = int(vectors[idx, 9]) # rbg state (0: r, 1: g, 2: b)

    # interact with IC (1st interaction)
    Ete, Etm, delta_phase = E_field_cal(
        Ete, Etm, delta_phase,
        lut_ic1[rgb, m, n, 2], lut_ic1[rgb, m, n, 5], # unit TE in, TE&TM out (+1st ref)
        lut_ic1[rgb, m, n, 14], lut_ic1[rgb, m, n, 17]) # unit TM in, TE&TM out (+1st ref)
    delta_phase += lut_TIR[rgb, m, n] # phase shift due to TIR (TM-TE)
    gap_x = lut_gap[rgb, m, n, 0] # assign the propagation distance
    gap_y = lut_gap[rgb, m, n, 1]
    x += gap_x
    y += gap_y

    # interact with IC (multi interaction)
    for _ in range(int(MAX_STEPS)):
        if not is_inside_or_on_edge(x, y, IC, 0, IC.shape[0]): # already exist IC region and propagate to OC
            break
        else:
            # check LUT, change propagation direction and electrical field here (TIR phase shift considered)
            # the parameters need to change: Es, Ep, delta_phase, theta, phi, gap_x, gap_y, energy
            Ete, Etm, delta_phase = E_field_cal(
                Ete, Etm, delta_phase,
                lut_ic2[rgb, m, n, 3], lut_ic2[rgb, m, n, 6],   # unit TE in, TE&TM out (0th ref)
                lut_ic2[rgb, m, n, 15], lut_ic2[rgb, m, n, 18]) # unit TM in, TE&TM out (0th ref)
            delta_phase += lut_TIR[rgb, m, n] # phase shift due to TIR (TM-TE)
            x += gap_x
            y += gap_y

    # interact with OC
    for _ in range(int(MAX_STEPS)):
        if not is_inside_or_on_edge(x, y, eff_reg, 0, eff_reg.shape[0]): # if not inside the entire effective region
            return
        
        hit = False # 'hit' used to check if there is overlap with OC
        for i in range(len(OC_offset) - 1):
            start_OC = OC_offset[i]
            end_OC = OC_offset[i+1]
            if is_inside_or_on_edge(x, y, OC, start_OC, end_OC):
                hit = True
                # If inside the FOV region for this FOV_num, accumulate to EB
                if is_inside_or_on_edge_4d(x, y, eff_reg_FOV, m, n):
                    Ete_out, Etm_out, _ = E_field_cal(
                        Ete, Etm, delta_phase,
                        lut_oc[i, rgb, m, n, 2], lut_oc[i, rgb, m, n, 5],  # +1st ref, TE in -> TE/TM
                        lut_oc[i, rgb, m, n, 14], lut_oc[i, rgb, m, n, 17])  # +1st ref, TM in -> TE/TM
                    # squared magnitude (no sqrt)
                    efficiency = Ete_out * Ete_out + Etm_out * Etm_out
                    # Add atomically into the EB matrix bin for this FOV
                    xmin = eff_reg_FOV_range[m, n, 0]
                    xmax = eff_reg_FOV_range[m, n, 1]
                    ymin = eff_reg_FOV_range[m, n, 2]
                    ymax = eff_reg_FOV_range[m, n, 3]
                    add_to_EB_atomic_val(matrix_EB[rgb,:,:,:,:], m, n, x, y, xmin, xmax, ymin, ymax, efficiency)

                # Then continue propagation with 0th order
                Ete, Etm, delta_phase = E_field_cal(
                    Ete, Etm, delta_phase,
                    lut_oc[i, rgb, m, n, 3],  lut_oc[i, rgb, m, n, 6],   # 0th ref
                    lut_oc[i, rgb, m, n, 15], lut_oc[i, rgb, m, n, 18])  # 0th ref
                delta_phase += lut_TIR[rgb, m, n]
                x += gap_x
                y += gap_y

                break

        # have not enter OC yet, continue propagate
        if not hit:
            delta_phase += 2 * lut_TIR[rgb, m, n]
            x += gap_x
            y += gap_y


@cuda.jit
def process_rays_kernel_PVG_IC(
    vectors,
    MAX_STEPS, 
    IC,
    lut_ic1, lut_ic2,
    lut_TIR, lut_gap,
    matrix_FOV):

    idx = cuda.grid(1)
    N = vectors.shape[0]

    if idx >= N:
        return
    
    # Read ray property in vectors
    x = vectors[idx, 0] # position of ray
    y = vectors[idx, 1]
    gap_x = vectors[idx, 2] # propagation distance for each TIR
    gap_y = vectors[idx, 3]
    m = int(vectors[idx, 4]) # LUT position
    n = int(vectors[idx, 5])
    Ete = vectors[idx, 6] # E-field
    Etm = vectors[idx, 7]
    delta_phase = vectors[idx, 8] # phase difference between te and tm (phi_tm - phi_te)  range: (-pi, pi]
    rgb = int(vectors[idx, 9]) # rbg state (0: r, 1: g, 2: b)

    # interact with IC (1st interaction)
    Ete, Etm, delta_phase = E_field_cal(
        Ete, Etm, delta_phase,
        lut_ic1[rgb, m, n, 2], lut_ic1[rgb, m, n, 5], # unit TE in, TE&TM out (+1st ref)
        lut_ic1[rgb, m, n, 14], lut_ic1[rgb, m, n, 17]) # unit TM in, TE&TM out (+1st ref)
    delta_phase += lut_TIR[rgb, m, n] # phase shift due to TIR (TM-TE)
    gap_x = lut_gap[rgb, m, n, 0] # assign the propagation distance
    gap_y = lut_gap[rgb, m, n, 1]
    x += gap_x
    y += gap_y

    # interact with IC (multi interaction)
    for _ in range(int(MAX_STEPS)):
        if not is_inside_or_on_edge(x, y, IC, 0, IC.shape[0]): # already exist IC region and propagate to OC
            efficiency = (Ete * Ete + Etm * Etm) * (math.cos(lut_ic2[rgb, m, n, 0].real) / math.cos(lut_ic1[rgb, m, n, 0].real))
            cuda.atomic.add(matrix_FOV, (rgb, m, n), efficiency)
            return
        else:
            # check LUT, change propagation direction and electrical field here (TIR phase shift considered)
            # the parameters need to change: Es, Ep, delta_phase, theta, phi, gap_x, gap_y, energy
            Ete, Etm, delta_phase = E_field_cal(
                Ete, Etm, delta_phase,
                lut_ic2[rgb, m, n, 3], lut_ic2[rgb, m, n, 6],   # unit TE in, TE&TM out (0th ref)
                lut_ic2[rgb, m, n, 15], lut_ic2[rgb, m, n, 18]) # unit TM in, TE&TM out (0th ref)
            delta_phase += lut_TIR[rgb, m, n] # phase shift due to TIR (TM-TE)
            x += gap_x
            y += gap_y